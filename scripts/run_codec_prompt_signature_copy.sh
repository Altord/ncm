#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
OUT="$BASE/codec-prompt-signature-copy-rare2"
[ -f "$INPUT" ] || { echo "Missing: $INPUT" >&2; exit 1; }
if [ -e "$OUT/methods.jsonl" ] || [ -e "$OUT/methods.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_codec_prompt_signature_copy.py
.venv/bin/python evaluation/codec_prompt_signature_copy.py --input "$INPUT" --output "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('\n=== Prompt signature copy results to share ===')
print(f"Exact text: {r['before_exact_text']} -> {r['after_exact_text']}/{r['examples']}; gained={r['new_exact_text']}; lost={r['lost_exact_text']}")
print(f"Exact names: {r['before_exact_names']} -> {r['after_exact_names']}/{r['scorable_names']}")
print(f"Similarity: {r['before_mean_character_similarity']:.2%} -> {r['after_mean_character_similarity']:.2%}")
print('Selections:', r['selection_counts'])
print('Signature is supplied in synthetic prompt; the VQ reconstruction encodes the known target. No generator or behavioral test.')
PY
