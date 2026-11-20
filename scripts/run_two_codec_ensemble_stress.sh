#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-two-model-ensemble-stress"
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$A/stress.jsonl" "$A/stress.report.json" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$B/stress.jsonl" "$B/stress.report.json" "$INPUT"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
.venv/bin/python evaluation/test_ensemble_codec_methods.py
.venv/bin/python evaluation/test_compile_codec_ensemble.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
if [ -f "$OUT/results.jsonl" ] && [ -f "$OUT/results.report.json" ]; then
  echo 'Using saved ensemble reconstructions.'
elif [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/results.report.json" ]; then
  echo 'Incomplete ensemble output exists; stopping to preserve it.' >&2; exit 1
else
  .venv/bin/python -m evaluation.ensemble_codec_methods \
    --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" --saved-a "$A/stress.jsonl" \
    --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$B/stress.jsonl" \
    --input "$INPUT" --output "$OUT/results.jsonl" --batch-size 8 --device cuda
fi
if [ -f "$OUT/compiled.jsonl" ] && [ -f "$OUT/compiled.report.json" ]; then
  echo 'Using saved compile report.'
  cat "$OUT/compiled.report.json"
elif [ -e "$OUT/compiled.jsonl" ] || [ -e "$OUT/compiled.report.json" ]; then
  echo 'Incomplete compile output exists; stopping to preserve it.' >&2; exit 1
else
  .venv/bin/python -m evaluation.compile_codec_ensemble \
    --input "$INPUT" --ensemble "$OUT/results.jsonl" --baseline-a "$A/stress.jsonl" \
    --baseline-b "$B/stress.jsonl" --output "$OUT/compiled.jsonl"
fi
