#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
METHOD_BASE="$BASE/codec-two-model-ensemble-methods/results.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
EXT_BASE="$BASE/codec-external-methods/ensemble.jsonl"
OUT="$BASE/codec-ensemble-chunks128"
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$METHODS" "$METHOD_BASE" "${METHOD_BASE%.jsonl}.report.json" "$EXTERNAL" "$EXT_BASE" "${EXT_BASE%.jsonl}.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 [ ! -e "$OUT/$name.jsonl" ] && [ ! -e "$OUT/$name.report.json" ] || { echo "Existing output: $OUT/$name; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_ensemble_chunk_screen.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; BASELINE="$METHOD_BASE"; else INPUT="$EXTERNAL"; BASELINE="$EXT_BASE"; fi
 .venv/bin/python -m evaluation.ensemble_chunk_screen --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --input "$INPUT" --baseline "$BASELINE" \
  --chunk-size 128 --batch-size 4 --device cuda --output "$OUT/$name.jsonl"
done
