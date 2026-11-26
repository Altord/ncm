#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXTERNAL="$BASE/codec-external-methods"
OUT="$BASE/codec-identifier-table-screen"
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$METHODS/reference.report.json" "$EXTERNAL/methods.jsonl" "$EXTERNAL/a.jsonl" "$EXTERNAL/a.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 [ ! -e "$OUT/$name.jsonl" ] && [ ! -e "$OUT/$name.report.json" ] || { echo "Existing output: $OUT/$name; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_identifier_table.py
.venv/bin/python evaluation/test_codec_identifier_table_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; BASELINE="$METHODS/reference.jsonl"; COUNT=64; else INPUT="$EXTERNAL/methods.jsonl"; BASELINE="$EXTERNAL/a.jsonl"; COUNT=70; fi
 .venv/bin/python -m evaluation.codec_identifier_table_screen --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
  --input "$INPUT" --baseline "$BASELINE" --max-examples "$COUNT" --device cuda --output "$OUT/$name.jsonl"
done
