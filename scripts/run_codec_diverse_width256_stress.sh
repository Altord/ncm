#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/codec-diverse-data-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
SAVED="$BASE/codec-diverse-data-5k-stress"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-diverse-width256-stress"
for f in "$A/config.yaml" "$A/checkpoints/step-00005000.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$B/stress.jsonl" "$B/stress.report.json" "$SAVED/candidate.jsonl" "$SAVED/candidate.report.json" "$SAVED/candidate-compiled.jsonl" "$SAVED/candidate-compiled.report.json" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/results.jsonl" "$OUT/results.report.json" "$OUT/compiled.jsonl" "$OUT/compiled.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
.venv/bin/python evaluation/test_ensemble_codec_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.ensemble_codec_methods --config-a "$A/config.yaml" --checkpoint-a "$A/checkpoints/step-00005000.pt" --saved-a "$SAVED/candidate.jsonl" \
 --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$B/stress.jsonl" --input "$INPUT" --output "$OUT/results.jsonl" --batch-size 8 --device cuda
.venv/bin/python -m evaluation.compile_diverse_ensemble --input "$INPUT" --ensemble "$OUT/results.jsonl" --baseline "$SAVED/candidate.jsonl" \
 --compiled-baseline "$SAVED/candidate-compiled.jsonl" --second-baseline "$B/stress.jsonl" --output "$OUT/compiled.jsonl"
