#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
HEAD="$BASE/codec-context-refinement-hard-1k/refinement_head.pt"
CAL="$BASE/codec-context-refinement-hard-1k/calibration.json"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-context-refinement-hard-stress"
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$CODEC/stress.jsonl" "$CODEC/stress.report.json" "$HEAD" "$CAL" "$INPUT" "$BASE/codec-repair-signature-hybrid/stress.jsonl"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing stress output in $OUT; stopping to preserve it." >&2; exit 1
fi
.venv/bin/python evaluation/test_refinement_stress.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.evaluate_refinement_stress \
  --config "$CODEC/config.yaml" --codec "$CODEC/artifact/tokenizer.pt" \
  --head "$HEAD" --calibration "$CAL" --input "$INPUT" --baseline "$CODEC/stress.jsonl" \
  --output "$OUT/stress.jsonl" --batch-size 16 --device cuda
.venv/bin/python - "$OUT/stress.jsonl" "$BASE/codec-repair-signature-hybrid/stress.jsonl" "$INPUT" <<'PY'
import json, sys
from pathlib import Path

def load(path):
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f'Missing comparison file: {p}')
    rows = [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {row['id']: row for row in rows}
    if not rows or len(rows) != len(result):
        raise SystemExit(f'Empty file or duplicate IDs: {p}')
    return result

gated, hybrid, source = (load(path) for path in sys.argv[1:])
if set(gated) != set(hybrid) or set(gated) != set(source):
    raise SystemExit('Cannot compare: stress IDs differ')
combined_compile = combined_exact = rescued = 0
for key in source:
    g, h, target = gated[key], hybrid[key], source[key]['target']
    if h['baseline']['text'] != g['baseline']:
        raise SystemExit(f'Baseline mismatch: {key}')
    earlier = h['select_strict']
    old_pass = earlier['compile_status'] == 'passed'
    new_pass = g['selected_compile'] == 'passed'
    if earlier['compile_status'] not in ('passed', 'failed', 'empty') or g['selected_compile'] not in ('passed', 'failed', 'empty'):
        raise SystemExit(f'Compile result unavailable: {key}')
    use_new = not old_pass and new_pass
    text = g['selected'] if use_new else earlier['text']
    combined_compile += old_pass or use_new
    combined_exact += text == target
    if use_new:
        rescued += 1
        print(f'Rescued: {key}; exact={text == target}')
        print(f'  Selected: {text}')
        if text != target:
            print(f'  Original: {target}')
print('\n=== Hard-refinement hybrid results to share ===')
print(f'Combined: {combined_compile}/{len(source)} compile; {combined_exact}/{len(source)} exact')
print(f'Additional compile passes vs earlier hybrid: {rescued}')
print('Candidate selection uses compile status; target text is used for offline scoring only.')
print('Known-target codec; synthetic target signatures; compilation does not prove behavior.')
PY
