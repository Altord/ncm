#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
BASE=runs/context256-codec-comparison
CANDIDATES="$BASE/codec-signature-parameter-search-fixed/stress.jsonl"
OUT="$BASE/codec-pretrained-candidate-ranker"
for file in "$INPUT" "$CANDIDATES"; do [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }; done
if [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/results.report.json" ]; then echo 'Existing results; stopping.' >&2; exit 1; fi
.venv/bin/python evaluation/test_rank_codec_candidates_lm.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.rank_codec_candidates_lm --input "$INPUT" --candidates "$CANDIDATES" \
  --output "$OUT/results.jsonl" --device cuda
