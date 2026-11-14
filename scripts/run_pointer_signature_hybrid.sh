#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
POINTER="$BASE/codec-pointer-stress-signature/stress.jsonl"
OUT="$BASE/codec-pointer-signature-hybrid"
for file in "$INPUT" "$POINTER" "${POINTER%.jsonl}.report.json"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_pointer_signature_hybrid.py
PYTHONUNBUFFERED=1 .venv/bin/python evaluation/evaluate_pointer_signature_hybrid.py \
  --input "$INPUT" --pointer-results "$POINTER" --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT/stress.report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('\n=== Pointer + signature-copy results to share ===')
for name in ('baseline', 'signature', 'contextual', 'signature_strict', 'signature_fallback', 'contextual_strict', 'contextual_fallback'):
    x = r['results'][name]
    print(f"{name}: exact={x['exact_text']}/{r['examples']}; compile={x['compile_passes']}/{r['examples']}; "
          f"names={x['name_exact']}/{r['examples']}; similarity={x['mean_character_similarity']:.2%}; "
          f"exact gained/lost={x['new_exact_text']}/{x['lost_exact_text']}; "
          f"compiles gained/lost={x['new_compiles']}/{x['lost_compiles']}")
print(f"New compile calls={r['new_compiler_calls']}; elapsed={r['elapsed_seconds']:.1f}s")
print('Prior compile-guided repair reference: 43/48 compiled, 40/48 exact text; separate system.')
print('Known-target codec round trip; synthetic signature supplied in prompt; no generator or behavioral test.')
PY
