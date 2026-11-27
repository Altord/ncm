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
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then
 export DOTNET_ROOT="$HOME/.dotnet"
 export PATH="$DOTNET_ROOT:$PATH"
fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found on PATH or at ~/.dotnet/dotnet.' >&2; exit 1; }
.venv/bin/python evaluation/test_compile_saved_codec_candidate.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in reference candidate; do
 if [ "$name" = reference ]; then MODEL="$REF"; else MODEL="$NEW"; fi
 if [ -f "$OUT/$name.jsonl" ] && [ -f "$OUT/$name.report.json" ]; then
  .venv/bin/python - "$OUT/$name.report.json" "$INPUT" "$MODEL/checkpoints/step-00003000.pt" <<'PY'
import json, sys
from pathlib import Path
from training.data import sha256_file
r = json.loads(Path(sys.argv[1]).read_text())
if r['input_sha256'] != sha256_file(sys.argv[2]) or Path(r['checkpoint']).resolve() != Path(sys.argv[3]).resolve():
    raise SystemExit('Saved reconstruction does not match requested input or checkpoint')
PY
  echo "Using saved reconstruction: $OUT/$name.jsonl"
 elif [ -e "$OUT/$name.jsonl" ] || [ -e "$OUT/$name.report.json" ]; then
  echo "Incomplete reconstruction output: $OUT/$name; stopping to preserve it." >&2; exit 1
 else
  .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00003000.pt" \
   --input "$INPUT" --text-key target --batch-size 8 --output "$OUT/$name.jsonl"
 fi
 if [ -f "$OUT/$name-compiled.jsonl" ] && [ -f "$OUT/$name-compiled.report.json" ]; then
  .venv/bin/python - "$OUT/$name-compiled.report.json" "$INPUT" "$OUT/$name.jsonl" <<'PY'
import json, sys
from pathlib import Path
from training.data import sha256_file
r = json.loads(Path(sys.argv[1]).read_text())
if r['input_sha256'] != sha256_file(sys.argv[2]) or r['candidate_sha256'] != sha256_file(sys.argv[3]):
    raise SystemExit('Saved compilation does not match requested input or reconstruction')
PY
  echo "Using saved compilation: $OUT/$name-compiled.jsonl"
 elif [ -e "$OUT/$name-compiled.jsonl" ] || [ -e "$OUT/$name-compiled.report.json" ]; then
  echo "Incomplete compilation output: $OUT/$name-compiled; stopping to preserve it." >&2; exit 1
 else
  .venv/bin/python -m evaluation.compile_saved_codec_candidate --input "$INPUT" --candidate "$OUT/$name.jsonl" \
   --baseline "$REF/stress.jsonl" --output "$OUT/$name-compiled.jsonl"
 fi
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
