#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
HYBRID="$BASE/codec-repair-signature-hybrid/stress.jsonl"
PREVIOUS="$BASE/codec-context-refinement-stress/stress.jsonl"
HARD="$BASE/codec-context-refinement-hard-stress/stress.jsonl"
OUT="$BASE/codec-refinement-signature-rescue"
for file in "$INPUT" "$HYBRID" "$PREVIOUS" "$HARD"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
.venv/bin/python evaluation/test_refinement_signature_rescue.py
.venv/bin/python -m evaluation.refinement_signature_rescue \
  --input "$INPUT" --hybrid "$HYBRID" --previous-refinement "$PREVIOUS" \
  --hard-refinement "$HARD" --output "$OUT/stress.jsonl"
