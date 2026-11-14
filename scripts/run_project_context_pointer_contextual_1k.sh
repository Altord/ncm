#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
DATASET="$BASE/codec-learned-project-context-copy-boost20-1k"
PREVIOUS="$BASE/codec-learned-project-context-copy-confidence-aux-1k"
COVER="$BASE/codec-project-context-copyability"
OUT="$BASE/codec-learned-project-context-copy-contextual-1k"
TRAIN="$DATASET/train-context.jsonl"
VAL="$COVER/augmented.jsonl"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen)' >/dev/null; then
  echo 'A training process is already running; finish or stop it first.' >&2; exit 1
fi
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$TRAIN" "$VAL" "$DATASET/train-context.manifest.json" "$PREVIOUS/report.json"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/copy_head.pt" ] || [ -e "$OUT/report.json" ] || [ -e "$OUT/results.jsonl" ]; then
  echo "Existing results in $OUT; stopping to preserve them." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_prompt_copy_head.py
.venv/bin/python evaluation/test_prompt_copy_screen_integration.py
.venv/bin/python - "$PREVIOUS/report.json" "$DATASET/train-context.manifest.json" "$TRAIN" "$VAL" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
old, manifest = (json.load(open(path)) for path in sys.argv[1:3])
train, val, codec = (sha256_file(path) for path in sys.argv[3:])
if (old['train_sha256'] != train or old['validation_sha256'] != val or old['codec_sha256'] != codec or
    manifest['augmented_train_sha256'] != train or manifest['validation_augmented_sha256'] != val):
    raise SystemExit('Training data, validation data or codec differs from the previous context run')
print('Same frozen codec, target-removed training and validation data verified.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_prompt_copy_screen \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --train "$TRAIN" --validation "$VAL" --output "$OUT" \
  --device cuda --steps 1000 --batch-size 8 --prefix-length 256 --copyable-error-boost 20 \
  --attention-aux-weight 0.2 --gate-aux-weight 0.1 --gate-positive-weight 20 --confidence-aware-gate --prompt-context-layers 2 --prompt-context-heads 4
.venv/bin/python - "$PREVIOUS/report.json" "$OUT/report.json" <<'PY'
import json, sys
old, new = (json.load(open(path)) for path in sys.argv[1:])
print('\n=== Contextual pointer results to share ===')
for name, r in [('prior confidence pointer', old), ('contextual pointer', new)]:
    print(f"{name}: token accuracy {r['after_token_accuracy']:.4%}; exact text {r['after_exact_text']}/{r['validation_examples']}; "
          f"exact names {r['after_exact_names']}/{r['scorable_names']}; fixed errors {r['baseline_errors_fixed']}; "
          f"new errors {r['new_token_errors']}; mean copy gate {r['mean_copy_gate']:.4%}")
print('Same frozen codec and target-removed project context; known-target VQ round trip only.')
PY
