#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
ENSEMBLE="$BASE/codec-two-model-ensemble-methods/results.jsonl"
OUT="$BASE/codec-ensemble-signature-methods"
for f in "$INPUT" "$ENSEMBLE" "${ENSEMBLE%.jsonl}.report.json"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
if [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/results.report.json" ]; then echo 'Existing output; stopping to preserve it.' >&2; exit 1; fi
.venv/bin/python evaluation/test_apply_ensemble_signature.py
.venv/bin/python -m evaluation.apply_ensemble_signature --input "$INPUT" --ensemble "$ENSEMBLE" --output "$OUT/results.jsonl"
