#!/usr/bin/env bash
set -euo pipefail
# Bell means the script ended; preserve its original success or failure status.
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
REPAIR="$BASE/codec-scope-repair-unique/selected.jsonl"
HYBRID="$BASE/codec-pointer-signature-hybrid/stress.jsonl"
POINTER="$BASE/codec-pointer-stress-signature/stress.jsonl"
OUT="$BASE/codec-repair-signature-hybrid"
for file in "$INPUT" "$REPAIR" "${REPAIR%.jsonl}.report.json" "$HYBRID" "${HYBRID%.jsonl}.report.json" "$POINTER" "${POINTER%.jsonl}.report.json"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_repair_signature_hybrid.py
PYTHONUNBUFFERED=1 .venv/bin/python evaluation/evaluate_repair_signature_hybrid.py \
  --input "$INPUT" --repair "$REPAIR" --hybrid "$HYBRID" --pointer "$POINTER" \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT/stress.report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('\n=== Repair + supplied signature results to share ===')
for name in ('baseline', 'repair', 'repair_strict', 'repair_fallback',
             'pointer_strict', 'pointer_fallback', 'select_strict', 'select_fallback'):
    x = r['results'][name]
    print(f"{name}: exact={x['exact_text']}/{r['examples']}; compile={x['compile_passes']}/{r['examples']}; "
          f"names={x['name_exact']}/{r['examples']}; similarity={x['mean_character_similarity']:.2%}; "
          f"exact gained/lost vs repair={x['new_exact_vs_repair']}/{x['lost_exact_vs_repair']}; "
          f"compile gained/lost vs repair={x['new_compiles_vs_repair']}/{x['lost_compiles_vs_repair']}")
print(f"New compiler calls={r['new_compiler_calls']}; elapsed={r['elapsed_seconds']:.1f}s")
print('Compiler-based selection is not a correctness guarantee; synthetic signature and known-target codec round trip.')
PY
