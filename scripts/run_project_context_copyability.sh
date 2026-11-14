#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
DATA="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
PREVIOUS="$BASE/codec-learned-prompt-copy-boost20-1k/report.json"
OUT="$BASE/codec-project-context-copyability"
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$DATA" "$PREVIOUS"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
[ -d data/reference ] || { echo 'Missing source root: data/reference' >&2; exit 1; }
if [ -e "$OUT/report.json" ] || [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/augmented.jsonl" ]; then
  echo "Existing results in $OUT; stopping to preserve them." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_project_context_copyability.py
.venv/bin/python evaluation/test_project_context_copyability_integration.py
.venv/bin/python - "$PREVIOUS" "$DATA" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
report = json.load(open(sys.argv[1]))
if report['validation_sha256'] != sha256_file(sys.argv[2]) or report['codec_sha256'] != sha256_file(sys.argv[3]):
    raise SystemExit('Previous pointer report refers to a different dataset or codec')
print('Previous pointer dataset and codec hashes match.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PYTHONUNBUFFERED=1 .venv/bin/python evaluation/measure_project_context_copyability.py \
  --config "$CODEC/config.yaml" --checkpoint "$CODEC/artifact/tokenizer.pt" \
  --input "$DATA" --source-root data/reference --output "$OUT" \
  --device cuda --batch-size 8 --context-tokens 256
.venv/bin/python - "$OUT/report.json" "$PREVIOUS" <<'PY'
import json, sys
current, previous = (json.load(open(p)) for p in sys.argv[1:])
print('\n=== Project-context copyability results to share ===')
print(f"Nearby context added: {current['context_status'].get('context_added', 0)}/{current['examples']}")
print(f"Baseline errors: {current['baseline_errors']}; previous screen: {previous['baseline_token_errors']}")
print(f"Copyable errors: prompt={current['prompt_copyable']}; with context={current['context_copyable']}; "
      f"new={current['newly_copyable']}; no longer copyable={current['no_longer_copyable']}")
print(f"Optimistic token accuracy ceiling: prompt={current['ideal_prompt_pointer_token_accuracy']:.2%}; "
      f"with context={current['ideal_context_pointer_token_accuracy']:.2%}")
print('Availability ceiling only: no pointer training, compile test, or generated-code evaluation.')
PY
