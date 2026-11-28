#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
SAVED="$BASE/codec-diverse-data-5k-stress"
ADAPTER="$BASE/codec-diverse-latent-refiner-1k/adapter.pt"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-diverse-adapter-stress"
for f in "$MODEL/config.yaml" "$MODEL/checkpoints/step-00005000.pt" "$SAVED/candidate.jsonl" "$SAVED/candidate.report.json" "$SAVED/candidate-compiled.jsonl" "$SAVED/candidate-compiled.report.json" "$ADAPTER" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/results.jsonl" "$OUT/results.report.json" "$OUT/compiled.jsonl" "$OUT/compiled.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
.venv/bin/python evaluation/test_codec_latent_refiner.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$MODEL/config.yaml" \
 --checkpoint "$MODEL/checkpoints/step-00005000.pt" --adapter "$ADAPTER" --input "$INPUT" --baseline "$SAVED/candidate.jsonl" \
 --batch-size 8 --device cuda --benchmark --output "$OUT/results.jsonl"
.venv/bin/python -m evaluation.compile_diverse_adapter --input "$INPUT" --adapter-output "$OUT/results.jsonl" \
 --baseline "$SAVED/candidate.jsonl" --compiled-baseline "$SAVED/candidate-compiled.jsonl" --output "$OUT/compiled.jsonl"
