#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
INPUT="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
SAMPLE="$BASE/codec-ast-signature-method-screen/results.jsonl"
COPY="$BASE/codec-prompt-signature-fallback-rare2/methods.jsonl"
OUT="$BASE/codec-method-token-ranks"
for f in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$INPUT" "${INPUT%.jsonl}.report.json" "$SAMPLE" "$COPY"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
if [ -e "$OUT/report.json" ]; then echo 'Existing report; stopping.' >&2; exit 1; fi
.venv/bin/python evaluation/test_measure_method_token_ranks.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.measure_method_token_ranks \
  --config "$CODEC/config.yaml" --checkpoint "$CODEC/artifact/tokenizer.pt" \
  --input "$INPUT" --sample "$SAMPLE" --signature-copy "$COPY" --output "$OUT/report.json" --device cuda
