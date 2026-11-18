#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS=data/mycode/godot-csharp-method-pairs-ctx256-prompt64/train.jsonl
HELDOUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
VALIDATION=data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl
OUT="$BASE/codec-method-mixture-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$REF/validation.report.json" "$VALIDATION" "$METHODS" "$HELDOUT" "$BASE/codec-method-boundary-rare-repeat/reference.jsonl"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/mixture.jsonl" "$OUT/mixture.manifest.json" "$OUT/cache/manifest.json" "$OUT/checkpoints/latest.pt" "$OUT/artifact/tokenizer.pt" "$OUT/methods.jsonl"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_method_mixture.py
.venv/bin/python evaluation/test_rare_codec_weights.py
mkdir -p "$OUT"
.venv/bin/python -m training.build_codec_method_mixture \
  --reference-config "$REF/config.yaml" --methods "$METHODS" --heldout-methods "$HELDOUT" \
  --output "$OUT/mixture.jsonl" --method-fraction 0.25
.venv/bin/python - "$REF" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
ref, out = sys.argv[1:]
cfg = OmegaConf.load(ref + '/config.yaml')
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    int(cfg.vqvae.dec_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 64 or
    float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0):
    raise SystemExit('Unexpected reference codec configuration')
rare = cfg.train.get('rare_token_weights')
if not rare or not rare.get('enabled', False): raise SystemExit('Expected rare-token weighting')
if cfg.train.get('error_teacher', {}).get('enabled', False) or float(cfg.train.get('distillation', {}).get('weight', 0.0)):
    raise SystemExit('Unexpected teacher settings in reference')
rare.reference_cache_dir = cfg.data.cache_dir
cfg.data.input = out + '/mixture.jsonl'
cfg.data.cache_dir = out + '/cache'
cfg.data.text_key = 'text'
cfg.task = 'codec-method-mixture-5k'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
OmegaConf.save(cfg, out + '/config.yaml')
print('Fresh 64-level codec; mixed training data; reference rarity counts and validation unchanged.', flush=True)
PY
.venv/bin/python -m training.prepare_data --config "$OUT/config.yaml" --stage tokenizer
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$VALIDATION" --text-key text --code-only --batch-size 16 --output "$OUT/validation.jsonl"
.venv/bin/python -m evaluation.compare_codec_aggregate \
  --reference-report "$REF/validation.report.json" --candidate-report "$OUT/validation.report.json" \
  --reference-config "$REF/config.yaml" --candidate-config "$OUT/config.yaml"
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$HELDOUT" --text-key target --code-only --batch-size 16 --output "$OUT/methods.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py \
  --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: r = json.load(f)
print('\n=== Method-mixture codec screen results to share ===')
print(f"Methods: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; similarity={r['mean_character_similarity']:.2%}")
print('Reference rare-only codec: 218/348 exact methods. See paired comparison above.')
print('Known-target codec reconstruction only; no compilation or generation evaluation.')
PY
