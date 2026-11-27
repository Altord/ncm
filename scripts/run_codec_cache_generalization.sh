#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
ADAPTER="$BASE/codec-latent-refiner-1k/adapter.pt"
OUT="$BASE/codec-cache-generalization"
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$ADAPTER"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
[ ! -e "$OUT/report.json" ] || { echo "Existing output: $OUT/report.json; stopping to preserve it." >&2; exit 1; }
.venv/bin/python evaluation/test_codec_cache_generalization.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_cache_generalization --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
 --adapter "$ADAPTER" --train-limit 512 --seed 42 --batch-size 8 --device cuda --output "$OUT/report.json"
