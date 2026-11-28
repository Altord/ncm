#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
CONTROL="$BASE/codec-deterministic-decoder-control"
FOLLOW="$BASE/codec-decoder-control-followup"
HOLD="$BASE/codec-diverse-5k-fresh-holdout"
OUT="$BASE/codec-prefix-offset-screen"
for f in "$MODEL/config.yaml" "$CONTROL/decoder.pt" "$CONTROL/methods.jsonl" "$CONTROL/external.jsonl" "$FOLLOW/stress.jsonl" "$FOLLOW/stress-compiled.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external stress; do
 for f in "$OUT/$name.jsonl" "$OUT/$name.report.json" "$OUT/$name.codes.pt"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
 done
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external stress; do
 case "$name" in
  methods) INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"; SAVED="$CONTROL/methods.jsonl"; LIMIT=64 ;;
  external) INPUT="$BASE/codec-external-methods/methods.jsonl"; SAVED="$CONTROL/external.jsonl"; LIMIT=70 ;;
  stress) INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl; SAVED="$FOLLOW/stress.jsonl"; LIMIT=48 ;;
 esac
 .venv/bin/python -m evaluation.codec_prefix_offset_search --config "$MODEL/config.yaml" --checkpoint "$CONTROL/decoder.pt" \
  --input "$INPUT" --baseline "$SAVED" --output "$OUT/$name.jsonl" --max-examples "$LIMIT" --offsets 0 4 8 16 --device cuda
done
.venv/bin/python -m evaluation.compile_diverse_adapter --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
 --adapter-output "$OUT/stress.jsonl" --baseline "$FOLLOW/stress.jsonl" --compiled-baseline "$FOLLOW/stress-compiled.jsonl" --output "$OUT/stress-compiled.jsonl"
