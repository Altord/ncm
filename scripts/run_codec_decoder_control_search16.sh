#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
CONTROL="$BASE/codec-deterministic-decoder-control"
FOLLOW="$BASE/codec-decoder-control-followup"
HOLD="$BASE/codec-diverse-5k-fresh-holdout"
OUT="$BASE/codec-decoder-control-search16"
for f in "$MODEL/config.yaml" "$CONTROL/decoder.pt" "$FOLLOW/holdout.jsonl" "$FOLLOW/stress.jsonl" "$FOLLOW/stress-compiled.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in holdout stress; do
 for f in "$OUT/$name.jsonl" "$OUT/$name.report.json" "$OUT/$name.codes.pt"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
 done
done
for f in "$OUT/stress-compiled.jsonl" "$OUT/stress-compiled.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in holdout stress; do
 if [ "$name" = holdout ]; then INPUT="$HOLD/methods.jsonl"; else INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl; fi
 .venv/bin/python -m evaluation.codec_stochastic_index_search --config "$MODEL/config.yaml" --checkpoint "$CONTROL/decoder.pt" \
  --input "$INPUT" --baseline "$FOLLOW/$name.jsonl" --output "$OUT/$name.jsonl" --batch-size 4 --samples 16 --temperature 0.05 --top-k 2 --seed 42 --device cuda
done
.venv/bin/python -m evaluation.compile_diverse_adapter --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
 --adapter-output "$OUT/stress.jsonl" --baseline "$FOLLOW/stress.jsonl" --compiled-baseline "$FOLLOW/stress-compiled.jsonl" --output "$OUT/stress-compiled.jsonl"
