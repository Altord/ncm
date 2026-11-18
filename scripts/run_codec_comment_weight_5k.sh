#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
OUT="$BASE/codec-comment-weight-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$METHODS" "$BASE/codec-method-boundary-rare-repeat/reference.jsonl"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/cache/manifest.json" "$OUT/cache/tokens.bin.tmp" "$OUT/checkpoints/latest.pt" "$OUT/artifact/tokenizer.pt" "$OUT/methods.jsonl"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_comment_codec_weights.py
.venv/bin/python evaluation/test_method_name_weighting.py
.venv/bin/python evaluation/test_repeated_identifier_weights.py
mkdir -p "$OUT"
.venv/bin/python - "$REF" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
ref, out = sys.argv[1:]
cfg = OmegaConf.load(ref + '/config.yaml')
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    int(cfg.vqvae.dec_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 64 or
    float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0):
    raise SystemExit('Unexpected reference codec configuration')
weights = cfg.train.get('reconstruction_token_weights')
if not weights or str(weights.get('mode', '')) != 'per_occurrence' or weights.get('comment') is not None:
    raise SystemExit('Expected per-occurrence reference with default comment weighting')
if cfg.train.get('error_teacher', {}).get('enabled', False) or float(cfg.train.get('distillation', {}).get('weight', 0.0)):
    raise SystemExit('Unexpected teacher settings in reference')
cfg.task = 'codec-comment-weight-5k'
cfg.data.cache_dir = out + '/cache'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
cfg.train.reconstruction_token_weights.comment = 2.0
OmegaConf.save(cfg, out + '/config.yaml')
print('Preparing a separate training cache: C# comment-token weight 2.0; other weights unchanged.', flush=True)
print('Fresh 64-level codec, 5,000 updates; validation remains unweighted.', flush=True)
PY
.venv/bin/python -m training.prepare_data --config "$OUT/config.yaml" --stage tokenizer
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$METHODS" --text-key target --code-only --batch-size 16 --output "$OUT/methods.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py \
  --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: report = json.load(f)
print('\n=== Comment-weight codec screen results to share ===')
print(f"Methods: accuracy={report['valid_token_accuracy']:.2%}; exact={report['exact_sequence_matches']}/{report['examples']}; similarity={report['mean_character_similarity']:.2%}")
print('Rare-only reference: 218/348 exact. See paired comparison above.')
print('Known-target codec reconstruction only; no compilation or generation evaluation.')
PY
