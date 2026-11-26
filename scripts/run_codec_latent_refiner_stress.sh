#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
ADAPTER="$BASE/codec-latent-refiner-1k/adapter.pt"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-latent-refiner-stress"
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$MODEL/stress.jsonl" "$MODEL/stress.report.json" "$ADAPTER" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/results.jsonl" "$OUT/results.report.json" "$OUT/compiled.jsonl" "$OUT/compiled.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_latent_refiner.py
.venv/bin/python evaluation/test_compile_codec_ensemble.py
.venv/bin/python evaluation/test_compile_codec_latent_refiner.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$MODEL/config.yaml" \
 --checkpoint "$MODEL/artifact/tokenizer.pt" --adapter "$ADAPTER" --input "$INPUT" --baseline "$MODEL/stress.jsonl" \
 --batch-size 8 --device cuda --benchmark --output "$OUT/results.jsonl"
.venv/bin/python -m evaluation.compile_codec_latent_refiner --input "$INPUT" --adapter-output "$OUT/results.jsonl" \
 --baseline "$MODEL/stress.jsonl" --output "$OUT/compiled.jsonl"
