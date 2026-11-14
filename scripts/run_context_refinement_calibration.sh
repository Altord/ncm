#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
HEAD="$BASE/codec-context-refinement-1k"
VAL="$BASE/codec-project-context-copyability/augmented.jsonl"
OUT="$BASE/codec-context-refinement-calibration"
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$HEAD/refinement_head.pt" "$HEAD/report.json" "$VAL"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/report.json" ]; then echo "Existing report in $OUT; stopping to preserve it." >&2; exit 1; fi
.venv/bin/python evaluation/test_context_refinement_calibration.py
.venv/bin/python - "$HEAD/report.json" "$VAL" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
with open(sys.argv[1]) as f: report = json.load(f)
if report['validation_sha256'] != sha256_file(sys.argv[2]) or report['codec_sha256'] != sha256_file(sys.argv[3]):
    raise SystemExit('Validation data or codec differs from the refinement screen')
print('Matched refinement screen data and codec.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.calibrate_context_refinement \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --head-checkpoint "$HEAD/refinement_head.pt" --validation "$VAL" \
  --output "$OUT/report.json" --device cuda --batch-size 8
