#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
SAMPLE="$BASE/codec-ast-signature-method-screen/results.jsonl"
COPY="$BASE/codec-prompt-signature-fallback-rare2/methods.jsonl"
OUT="$BASE/codec-ast-failure-coverage"
for file in "$INPUT" "$SAMPLE" "$COPY"; do [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }; done
if [ -e "$OUT/report.json" ]; then echo 'Existing report; stopping.' >&2; exit 1; fi
.venv/bin/python evaluation/test_ast_codec_candidates.py
.venv/bin/python evaluation/test_ast_codec_coverage.py
.venv/bin/python -m evaluation.diagnose_ast_codec_coverage \
  --input "$INPUT" --sample "$SAMPLE" --signature-copy "$COPY" --output "$OUT/report.json"
