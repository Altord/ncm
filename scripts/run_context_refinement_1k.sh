#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
DATASET="$BASE/codec-learned-project-context-copy-boost20-1k"
PREVIOUS="$BASE/codec-learned-project-context-copy-contextual-1k"
TRAIN="$DATASET/train-context.jsonl"
VAL="$BASE/codec-project-context-copyability/augmented.jsonl"
OUT="$BASE/codec-context-refinement-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A training process is already running; finish or stop it first.' >&2; exit 1
fi
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$TRAIN" "$VAL" "$DATASET/train-context.manifest.json" "$PREVIOUS/report.json"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/refinement_head.pt" ] || [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/report.json" ]; then
  echo "Existing results in $OUT; stopping to preserve them." >&2; exit 1
fi
.venv/bin/python evaluation/test_context_refinement_head.py
.venv/bin/python evaluation/test_context_refinement_integration.py
.venv/bin/python - "$PREVIOUS/report.json" "$DATASET/train-context.manifest.json" "$TRAIN" "$VAL" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
with open(sys.argv[1]) as f: old = json.load(f)
with open(sys.argv[2]) as f: manifest = json.load(f)
train, val, codec = (sha256_file(path) for path in sys.argv[3:])
if (old['train_sha256'] != train or old['validation_sha256'] != val or old['codec_sha256'] != codec or
    manifest['augmented_train_sha256'] != train or manifest['validation_augmented_sha256'] != val):
    raise SystemExit('Training data, validation data, or codec differs from the previous run')
print('Matched frozen codec and target-removed context data.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_context_refinement_screen \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --train "$TRAIN" --validation "$VAL" --output "$OUT" \
  --device cuda --steps 1000 --batch-size 8 --prefix-length 256 --error-boost 20 --trust-weight 0.05
.venv/bin/python - "$PREVIOUS/report.json" "$OUT/report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: old = json.load(f)
with open(sys.argv[2]) as f: new = json.load(f)
print('\n=== Refinement results to share ===')
for name, r in [('contextual pointer', old), ('vocabulary-wide refinement', new)]:
    print(f"{name}: accuracy={r['after_token_accuracy']:.2%}; exact={r['after_exact_text']}/{r['validation_examples']}; names={r['after_exact_names']}/{r['scorable_names']}; fixed={r['baseline_errors_fixed']}; new errors={r['new_token_errors']}")
print(f"Refinement fixed {new['fixed_target_absent_from_prompt']} errors whose correct token was absent from prompt/context.")
print('Frozen known-target codec; no generator, compile, or behavioral evaluation.')
PY
