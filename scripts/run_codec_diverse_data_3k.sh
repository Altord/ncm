#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
DATA="$BASE/codec-diverse-training-data"
OLD3K="$BASE/codec-continuous-teacher-validnorm-3k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
HOLDOUT="$BASE/codec-fresh-godot-holdout/methods.jsonl"
OUT="$BASE/codec-diverse-data-3k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
 echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF/checkpoints/step-00003000.pt" "$REF/logs/validation.jsonl" \
 "$DATA/train.jsonl" "$DATA/train.manifest.json" "$DATA/methods.jsonl" "$DATA/methods.manifest.json" "$HOLDOUT" \
 "$METHODS" "$EXTERNAL" "$OLD3K/methods-reference.jsonl" "$OLD3K/methods-reference.report.json" \
 "$OLD3K/external-reference.jsonl" "$OLD3K/external-reference.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/cache" "$OUT/checkpoints" "$OUT/artifact" "$OUT/methods.jsonl" "$OUT/external.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_merge_codec_training.py
.venv/bin/python evaluation/test_build_fresh_codec_methods_integration.py
.venv/bin/python - "$REF/config.yaml" "$DATA" "$HOLDOUT" "$OUT" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
from training.data import sha256_file
ref, data, holdout, out = sys.argv[1:]
cfg = OmegaConf.load(ref); OmegaConf.resolve(cfg)
merged = json.loads(Path(data,'train.manifest.json').read_text())
extra = json.loads(Path(data,'methods.manifest.json').read_text())
if merged['merged_sha256'] != sha256_file(Path(data,'train.jsonl')) or merged['original_sha256'] != sha256_file(cfg.data.input):
    raise SystemExit('Merged or reference training corpus hash differs')
if merged['supplemental_manifest_sha256'] != sha256_file(Path(data,'methods.manifest.json')) or extra['dataset_sha256'] != sha256_file(Path(data,'methods.jsonl')):
    raise SystemExit('Supplemental dataset or manifest hash differs')
if extra['source_sha256'].get(holdout) != sha256_file(holdout):
    raise SystemExit('Fresh holdout was not checked by the supplemental dataset builder')
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or int(cfg.vqvae.dec_layers) != 4 or
    len(cfg.vqvae.vqvae_vocab_size) != 64 or float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0):
    raise SystemExit('Unexpected reference architecture')
if not cfg.train.get('rare_token_weights', {}).get('enabled', False): raise SystemExit('Expected reference rarity weights')
if float(cfg.train.get('continuous_distillation', {}).get('weight', 0)) or float(cfg.train.get('distillation', {}).get('weight', 0)):
    raise SystemExit('Unexpected reference teacher settings')
reference_cache = str(cfg.data.cache_dir)
if not Path(reference_cache,'manifest.json').is_file(): raise SystemExit('Reference rarity cache missing')
cfg.train.rare_token_weights.reference_cache_dir = reference_cache
cfg.data.input = data+'/train.jsonl'
cfg.data.cache_dir = out+'/cache'
cfg.train.run_dir = out; cfg.train.artifact_dir = out+'/artifact'
cfg.train.eval_every = 1000; cfg.train.save_every = 1000
cfg.task = 'codec-diverse-data-3k'
Path(out).mkdir(parents=True, exist_ok=True)
OmegaConf.save(cfg, out+'/config.yaml')
print(f"Training rows: {merged['original_records']} original + {merged['supplemental_records']} supplemental.", flush=True)
print('Original rarity-frequency cache retained; unchanged 64-level model, optimizer, validation and schedule.', flush=True)
print('Fresh run capped at 3,000 updates; larger training corpus changes effective epochs and data order.', flush=True)
PY
.venv/bin/python -m training.prepare_data --config "$OUT/config.yaml" --stage tokenizer
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 3000
STEP="$OUT/checkpoints/step-00003000.pt"
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; else INPUT="$EXTERNAL"; fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" --checkpoint "$STEP" \
  --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name.jsonl"
 echo "===== $name: matched 3,000-step comparison ====="
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$OLD3K/$name-reference.jsonl" --candidate "$OUT/$name.jsonl"
done
.venv/bin/python - "$REF/logs/validation.jsonl" "$OUT/logs/validation.jsonl" "$OLD3K" "$OUT" <<'PY'
import json, sys
from pathlib import Path
for label, path in [('reference',sys.argv[1]), ('diverse',sys.argv[2])]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    match = [r for r in rows if r.get('step') == 3000]
    if not match: print(f'{label}: missing step-3000 validation'); continue
    r = match[-1]
    print(f"{label} validation: accuracy={r['val_token_accuracy']:.2%}; exact={r['val_exact_sequence_match_rate']:.2%}; CE={r['val_recon_ce']:.4f}")
for name in ('methods','external'):
    ref = json.loads((Path(sys.argv[3])/f'{name}-reference.report.json').read_text())
    new = json.loads((Path(sys.argv[4])/f'{name}.report.json').read_text())
    print(f"{name}: accuracy {ref['valid_token_accuracy']:.2%} -> {new['valid_token_accuracy']:.2%}; exact {ref['exact_sequence_matches']}/{ref['examples']} -> {new['exact_sequence_matches']}/{new['examples']}")
print('Fresh training run; same step count, but different corpus size and data order. No compilation or generation evaluation.')
print('Fresh project holdout was used for exclusion only and is not evaluated by this tuning script.')
PY
