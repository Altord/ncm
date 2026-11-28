#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
HOLD="$BASE/codec-diverse-5k-fresh-holdout"
ADAPTER="$BASE/codec-diverse-latent-refiner-1k/adapter.pt"
OUT="$BASE/codec-diverse-adapter-holdout"
for f in "$MODEL/config.yaml" "$MODEL/checkpoints/step-00005000.pt" "$HOLD/methods.jsonl" "$HOLD/new.jsonl" "$HOLD/new.report.json" "$ADAPTER"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/results.jsonl" "$OUT/results.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$MODEL/config.yaml" \
 --checkpoint "$MODEL/checkpoints/step-00005000.pt" --adapter "$ADAPTER" --input "$HOLD/methods.jsonl" \
 --baseline "$HOLD/new.jsonl" --batch-size 8 --device cuda --output "$OUT/results.jsonl"
