#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
OUT="$BASE/codec-stochastic-index-search"
for f in "$MODEL/config.yaml" "$MODEL/checkpoints/step-00005000.pt" "$MODEL/methods-candidate.jsonl" "$MODEL/external-candidate.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 for f in "$OUT/$name.jsonl" "$OUT/$name.report.json" "$OUT/$name.codes.pt"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
 done
done
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"; LIMIT=64; else INPUT="$BASE/codec-external-methods/methods.jsonl"; LIMIT=70; fi
 .venv/bin/python -m evaluation.codec_stochastic_index_search --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00005000.pt" \
  --input "$INPUT" --baseline "$MODEL/$name-candidate.jsonl" --output "$OUT/$name.jsonl" --batch-size 4 --max-examples "$LIMIT" --samples 4 --temperature 0.05 --top-k 2 --device cuda
done
