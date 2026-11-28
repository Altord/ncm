#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
STRESS="$BASE/codec-diverse-data-5k-stress"
INITIAL="$BASE/codec-stochastic-index-16/stress.jsonl"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-stochastic-gradient-stress"
for f in "$MODEL/config.yaml" "$MODEL/checkpoints/step-00005000.pt" "$STRESS/candidate.jsonl" "$STRESS/candidate-compiled.jsonl" "$INITIAL" "${INITIAL%.jsonl}.codes.pt" "${INITIAL%.jsonl}.report.json" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/stress.jsonl" "$OUT/stress.report.json" "$OUT/stress-compiled.jsonl" "$OUT/stress-compiled.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
.venv/bin/python evaluation/test_codec_index_search.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_index_search --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00005000.pt" \
 --input "$INPUT" --baseline "$STRESS/candidate.jsonl" --initial-search "$INITIAL" --output "$OUT/stress.jsonl" \
 --max-examples 48 --budget 32 --rounds 4 --device cuda
.venv/bin/python -m evaluation.compile_diverse_adapter --input "$INPUT" --adapter-output "$OUT/stress.jsonl" --baseline "$STRESS/candidate.jsonl" \
 --compiled-baseline "$STRESS/candidate-compiled.jsonl" --output "$OUT/stress-compiled.jsonl"
