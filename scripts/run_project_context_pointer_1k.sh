#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
TRAIN=data/mycode/godot-csharp-method-pairs-ctx256-prompt64/train.jsonl
VAL="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
COVER="$BASE/codec-project-context-copyability"
SIG_REPORT="$BASE/codec-learned-prompt-copy-boost20-1k/report.json"
OUT="$BASE/codec-learned-project-context-copy-boost20-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen)' >/dev/null; then
  echo 'A training process is already running; finish or stop it first.' >&2; exit 1
fi
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$TRAIN" "$VAL" "$COVER/report.json" "$COVER/augmented.jsonl" "$SIG_REPORT"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/train-context.jsonl" ] || [ -e "$OUT/copy_head.pt" ] || [ -e "$OUT/report.json" ]; then
  echo "Existing data or results in $OUT; stopping to preserve them." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_project_context_copyability.py
.venv/bin/python evaluation/test_prepare_project_context_pairs.py
.venv/bin/python evaluation/test_prompt_copy_head.py
.venv/bin/python evaluation/test_prompt_copy_screen_integration.py
.venv/bin/python - "$SIG_REPORT" "$VAL" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
report = json.load(open(sys.argv[1]))
if report['validation_sha256'] != sha256_file(sys.argv[2]) or report['codec_sha256'] != sha256_file(sys.argv[3]):
    raise SystemExit('Signature-only result refers to a different validation dataset or codec')
PY
echo '===== Prepare training context and verify previous validation context ====='
.venv/bin/python -m training.prepare_project_context_pairs \
  --config "$CODEC/config.yaml" --checkpoint "$CODEC/artifact/tokenizer.pt" \
  --train "$TRAIN" --validation-original "$VAL" --validation-augmented "$COVER/augmented.jsonl" \
  --coverage-report "$COVER/report.json" --source-root data/reference \
  --output "$OUT/train-context.jsonl" --context-tokens 256
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
echo '===== Train the pointer head; frozen 64-level codec ====='
.venv/bin/python -m training.train_prompt_copy_screen \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --train "$OUT/train-context.jsonl" --validation "$COVER/augmented.jsonl" --output "$OUT" \
  --device cuda --steps 1000 --batch-size 8 --prefix-length 256 --copyable-error-boost 20
echo '===== Compare with signature-only pointer ====='
.venv/bin/python - "$OUT/report.json" "$SIG_REPORT" "$COVER/report.json" <<'PY'
import json, sys
context, signature, coverage = (json.load(open(path)) for path in sys.argv[1:])
if context['codec_sha256'] != signature['codec_sha256'] or context['codec_sha256'] != coverage['codec_sha256']:
    raise SystemExit('Codec hashes differ; cannot compare')
print('\n=== Context-trained pointer results to share ===')
for name, r in [('signature only', signature), ('project context', context)]:
    print(f"{name}: token accuracy {r['after_token_accuracy']:.4%}; exact text {r['after_exact_text']}/{r['validation_examples']}; "
          f"exact names {r['after_exact_names']}/{r['scorable_names']}; "
          f"fixed errors {r['baseline_errors_fixed']}; new errors {r['new_token_errors']}; "
          f"copyable baseline errors {r['baseline_errors_with_target_token_in_prompt']}")
print(f"Context-token coverage diagnostic expected {coverage['context_copyable']} copyable errors; "
      f"training evaluation observed {context['baseline_errors_with_target_token_in_prompt']}.")
print('Synthetic signature plus target-removed same-file context; VQ latents encode known target. No generator or semantic evaluation.')
PY
