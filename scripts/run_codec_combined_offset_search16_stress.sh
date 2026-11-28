#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
CONTROL="$BASE/codec-deterministic-decoder-control"
FOLLOW="$BASE/codec-decoder-control-followup"
STOCH="$BASE/codec-decoder-control-search16/stress.jsonl"
OFFSET="$BASE/codec-prefix-offset-screen/stress.jsonl"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-combined-offset-search16-stress"
for f in "$MODEL/config.yaml" "$CONTROL/decoder.pt" "$STOCH" "$OFFSET" "$FOLLOW/stress.jsonl" "$FOLLOW/stress-compiled.jsonl" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/stress.jsonl" "$OUT/stress.report.json" "$OUT/stress.codes.pt" "$OUT/stress-compiled.jsonl" "$OUT/stress-compiled.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.combine_codec_encoder_search --config "$MODEL/config.yaml" --checkpoint "$CONTROL/decoder.pt" \
 --input "$INPUT" --first "$STOCH" --second "$OFFSET" --output "$OUT/stress.jsonl" --device cuda
.venv/bin/python -m evaluation.compile_diverse_adapter --input "$INPUT" --adapter-output "$OUT/stress.jsonl" --baseline "$FOLLOW/stress.jsonl" \
 --compiled-baseline "$FOLLOW/stress-compiled.jsonl" --output "$OUT/stress-compiled.jsonl"
