#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
NEW="$BASE/codec-diverse-data-3k"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-diverse-data-3k-stress"
for f in "$REF/config.yaml" "$REF/checkpoints/step-00003000.pt" "$REF/stress.jsonl" "$REF/stress.report.json" \
 "$NEW/config.yaml" "$NEW/checkpoints/step-00003000.pt" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in reference candidate; do
 for suffix in '.jsonl' '.report.json' '-compiled.jsonl' '-compiled.report.json'; do
  [ ! -e "$OUT/$name$suffix" ] || { echo "Existing output: $OUT/$name$suffix; stopping to preserve it." >&2; exit 1; }
 done
done
.venv/bin/python evaluation/test_compile_saved_codec_candidate.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in reference candidate; do
 if [ "$name" = reference ]; then MODEL="$REF"; else MODEL="$NEW"; fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00003000.pt" \
  --input "$INPUT" --text-key target --batch-size 8 --output "$OUT/$name.jsonl"
 .venv/bin/python -m evaluation.compile_saved_codec_candidate --input "$INPUT" --candidate "$OUT/$name.jsonl" \
  --baseline "$REF/stress.jsonl" --output "$OUT/$name-compiled.jsonl"
done
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/reference.jsonl" --candidate "$OUT/candidate.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
def load(name):
    rows = [json.loads(s) for s in (root/f'{name}-compiled.jsonl').read_text().splitlines() if s.strip()]
    out = {r['id']:r for r in rows}
    if not rows or len(out) != len(rows): raise SystemExit(f'Empty or duplicate IDs: {name}')
    return out
a, b = load('reference'), load('candidate')
if a.keys() != b.keys(): raise SystemExit('IDs differ')
print('\n=== Matched 3,000-step stress comparison ===')
print(f"Compile: {sum(r['compile_status']=='passed' for r in a.values())}/{len(a)} -> {sum(r['compile_status']=='passed' for r in b.values())}/{len(b)}")
print(f"Exact text: {sum(r['exact_text'] for r in a.values())}/{len(a)} -> {sum(r['exact_text'] for r in b.values())}/{len(b)}")
print(f"Compile gains: {sum(a[k]['compile_status']!='passed' and b[k]['compile_status']=='passed' for k in a)}; losses: {sum(a[k]['compile_status']=='passed' and b[k]['compile_status']!='passed' for k in a)}")
print('Known-target codec reconstruction, reused stress data. Compilation does not establish behavior or generator quality.')
PY
