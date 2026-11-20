#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
SAVED_A="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
OUT="$BASE/codec-two-model-ensemble-methods"
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$B/methods.jsonl" "$B/methods.report.json" "$INPUT" "$SAVED_A" "${SAVED_A%.jsonl}.report.json"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
if [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/results.report.json" ]; then echo 'Existing output; stopping to preserve it.' >&2; exit 1; fi
.venv/bin/python evaluation/test_ensemble_codec_methods.py
.venv/bin/python evaluation/test_ensemble_codec_integration.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.ensemble_codec_methods \
  --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" --saved-a "$SAVED_A" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$B/methods.jsonl" \
  --input "$INPUT" --output "$OUT/results.jsonl" --batch-size 8 --device cuda
