#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
ADAPTER="$BASE/codec-latent-refiner-1k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXTERNAL="$BASE/codec-external-methods"
OUT="$BASE/codec-latent-refiner-gate"
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$ADAPTER/adapter.pt" \
 "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$METHODS/reference.report.json" "$ADAPTER/methods.jsonl" "$ADAPTER/methods.report.json" \
 "$EXTERNAL/methods.jsonl" "$EXTERNAL/a.jsonl" "$EXTERNAL/a.report.json" "$ADAPTER/external.jsonl" "$ADAPTER/external.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/methods.jsonl" "$OUT/external.jsonl" "$OUT/report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_latent_refiner_gate.py
.venv/bin/python evaluation/test_codec_latent_refiner_gate_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_gate --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
 --adapter "$ADAPTER/adapter.pt" --methods-input "$METHODS/heldout-methods.jsonl" --methods-baseline "$METHODS/reference.jsonl" \
 --methods-adapter "$ADAPTER/methods.jsonl" --external-input "$EXTERNAL/methods.jsonl" --external-baseline "$EXTERNAL/a.jsonl" \
 --external-adapter "$ADAPTER/external.jsonl" --batch-size 8 --device cuda --output-dir "$OUT"
