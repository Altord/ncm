#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
CAND="$BASE/codec-teacher-error-weight-5k/methods.jsonl"
OUT="$BASE/codec-teacher-error-region-comparison"
for file in "$REF" "$CAND"; do [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }; done
if [ -e "$OUT/report.json" ]; then echo 'Existing report; stopping.' >&2; exit 1; fi
.venv/bin/python evaluation/test_compare_codec_error_regions.py
.venv/bin/python -m evaluation.compare_codec_error_regions --reference "$REF" --candidate "$CAND" --output "$OUT/report.json"
