#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-width256-encoder8-rare2-5k"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-width256-stochastic-stress"
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$MODEL/stress.jsonl" "$MODEL/stress.report.json" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/stress.jsonl" "$OUT/stress.report.json" "$OUT/stress.codes.pt"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_stochastic_index_search --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
 --input "$INPUT" --baseline "$MODEL/stress.jsonl" --output "$OUT/stress.jsonl" --batch-size 4 --samples 16 --temperature 0.05 --top-k 2 --seed 42 --device cuda
