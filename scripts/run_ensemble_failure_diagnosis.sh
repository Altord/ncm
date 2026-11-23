#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CONFIG="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k/config.yaml"
OUT="$BASE/codec-ensemble-failure-diagnosis"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
METHOD_ENSEMBLE="$BASE/codec-two-model-ensemble-methods/results.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
EXTERNAL_ENSEMBLE="$BASE/codec-external-methods/ensemble.jsonl"
for f in "$CONFIG" "$METHODS" "$METHOD_ENSEMBLE" "${METHOD_ENSEMBLE%.jsonl}.report.json" "$EXTERNAL" "$EXTERNAL_ENSEMBLE" "${EXTERNAL_ENSEMBLE%.jsonl}.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/methods.json" "$OUT/external.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_diagnose_ensemble_failures.py
mkdir -p "$OUT"
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; ENSEMBLE="$METHOD_ENSEMBLE"; else INPUT="$EXTERNAL"; ENSEMBLE="$EXTERNAL_ENSEMBLE"; fi
 echo "===== $name ====="
 .venv/bin/python -m evaluation.diagnose_ensemble_failures --input "$INPUT" --ensemble "$ENSEMBLE" --config "$CONFIG" --output "$OUT/$name.json"
done
