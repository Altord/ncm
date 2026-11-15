#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
PREVIOUS="$BASE/codec-refinement-signature-rescue/stress.jsonl"
OUT="$BASE/codec-signature-parameter-search-fixed"
for file in "$INPUT" "$PREVIOUS"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
.venv/bin/python evaluation/test_signature_parameter_search.py
.venv/bin/python -m evaluation.signature_parameter_search \
  --input "$INPUT" --previous "$PREVIOUS" --output "$OUT/stress.jsonl" --max-attempts 12
