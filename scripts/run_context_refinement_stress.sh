#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
HEAD="$BASE/codec-context-refinement-1k/refinement_head.pt"
CAL="$BASE/codec-context-refinement-calibration/report.json"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-context-refinement-stress"
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$CODEC/stress.jsonl" "$CODEC/stress.report.json" "$HEAD" "$CAL" "$INPUT"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing stress output in $OUT; stopping to preserve it." >&2; exit 1
fi
.venv/bin/python evaluation/test_refinement_stress.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.evaluate_refinement_stress \
  --config "$CODEC/config.yaml" --codec "$CODEC/artifact/tokenizer.pt" \
  --head "$HEAD" --calibration "$CAL" --input "$INPUT" --baseline "$CODEC/stress.jsonl" \
  --output "$OUT/stress.jsonl" --batch-size 16 --device cuda
