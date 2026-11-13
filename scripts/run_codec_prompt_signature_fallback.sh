#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
PRIOR="$BASE/codec-prompt-signature-copy-rare2/methods.jsonl"
OUT="$BASE/codec-prompt-signature-fallback-rare2"
for path in "$INPUT" "$PRIOR"; do
  [ -f "$path" ] || { echo "Missing: $path" >&2; exit 1; }
done
if [ -e "$OUT/methods.jsonl" ] || [ -e "$OUT/methods.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_codec_prompt_signature_copy.py
.venv/bin/python evaluation/codec_prompt_signature_copy.py \
  --input "$INPUT" --fallback --output "$OUT/methods.jsonl"
.venv/bin/python - "$PRIOR" "$OUT/methods.jsonl" "$INPUT" <<'PY'
import json, sys
from pathlib import Path
from evaluation.codec_prompt_signature_copy import copy_signature
def load(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    d = {r['id']: r for r in rows}
    if not d or len(d) != len(rows):
        raise SystemExit(f'Empty or duplicate IDs: {path}')
    return d
old, new, inputs = map(load, sys.argv[1:])
if (set(old) != set(new) or set(old) != set(inputs) or
    any(old[k]['before_exact'] != new[k]['before_exact'] or
        old[k]['selected'] != copy_signature(inputs[k]['prompt'], inputs[k]['reconstructed'])[0]
        for k in old)):
    raise SystemExit('Input IDs or baseline scoring differ')
print('\n=== Prompt signature fallback results to share ===')
print(f"Prior exact: {sum(r['after_exact'] for r in old.values())}/{len(old)}")
print(f"Fallback exact: {sum(r['after_exact'] for r in new.values())}/{len(new)}")
print(f"Newly exact: {sum(not old[k]['after_exact'] and new[k]['after_exact'] for k in old)}")
print(f"Lost exact: {sum(old[k]['after_exact'] and not new[k]['after_exact'] for k in old)}")
print(f"Fallback applied: {sum(r['selection'] == 'fallback_signature_copied' for r in new.values())}")
print(f"Method names exact: {sum(r['after_name_exact'] for r in old.values())} -> {sum(r['after_name_exact'] for r in new.values())}")
print('Synthetic prompt provides signature; VQ encodes the known target. No generator, compilation or behavioral tests.')
PY
