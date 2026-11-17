#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
OUT="$BASE/codec-real-method-ranking-conservative"
[ -f "$INPUT" ] || { echo "Missing: $INPUT" >&2; exit 1; }
if [ -e "$OUT/results.jsonl" ] || [ -e "$OUT/results.report.json" ]; then echo 'Existing results; stopping.' >&2; exit 1; fi
.venv/bin/python evaluation/test_signature_parameter_search.py
.venv/bin/python evaluation/test_rank_codec_candidates_lm.py
.venv/bin/python evaluation/test_rank_real_codec_methods.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.rank_real_codec_methods \
  --input "$INPUT" --output "$OUT/results.jsonl" --max-examples 100 --gap 0.05 --device cuda
