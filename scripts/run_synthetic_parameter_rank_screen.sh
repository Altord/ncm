#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT="$BASE/codec-project-context-copyability/augmented.jsonl"
OUT="$BASE/codec-synthetic-parameter-ranking"
[ -f "$INPUT" ] || { echo "Missing: $INPUT" >&2; exit 1; }
if [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/results.report.json" ]; then echo 'Existing results; stopping.' >&2; exit 1; fi
.venv/bin/python evaluation/test_synthetic_parameter_rank_screen.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.synthetic_parameter_rank_screen \
  --input "$INPUT" --output "$OUT/results.jsonl" --max-examples 80 --device cuda
