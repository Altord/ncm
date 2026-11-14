#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
SIG="$BASE/codec-learned-prompt-copy-boost20-1k/copy_head.pt"
CONTEXT="$BASE/codec-learned-project-context-copy-contextual-1k/copy_head.pt"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-pointer-stress-signature"
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$CODEC/stress.jsonl" "$CODEC/stress.report.json" "$SIG" "$CONTEXT" "$INPUT"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/stress.jsonl" ] || [ -e "$OUT/stress.report.json" ]; then
  echo "Existing output in $OUT; stopping to preserve it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_evaluate_pointer_stress.py
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PYTHONUNBUFFERED=1 .venv/bin/python evaluation/evaluate_pointer_stress.py \
  --config "$CODEC/config.yaml" --codec "$CODEC/artifact/tokenizer.pt" \
  --input "$INPUT" --baseline "$CODEC/stress.jsonl" \
  --signature-head "$SIG" --context-head "$CONTEXT" \
  --output "$OUT/stress.jsonl" --batch-size 16 --device cuda
.venv/bin/python - "$OUT/stress.report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print('\n=== Pointer stress results to share ===')
for name in ('baseline', 'signature', 'contextual'):
    x = r['results'][name]
    print(f"{name}: accuracy={x['token_accuracy']:.2%}; exact text={x['exact_text']}/{r['examples']}; "
          f"names={x['name_exact']}/{r['examples']}; compile={x['compile_passes']}/{r['examples']}; "
          f"fixed tokens={x['fixed_tokens']}; new token errors={x['new_token_errors']}")
    if name != 'baseline':
        print(f"  vs baseline: exact gained={x['new_exact_text']}, lost={x['lost_exact_text']}; "
              f"compiles gained={x['new_compiles']}, lost={x['lost_compiles']}")
print(f"New compile calls: {r['new_compiler_calls']}; elapsed={r['elapsed_seconds']:.1f}s")
print('Known-target codec round trip; synthetic prompt supplies target signature. No project context, generation or behavioral test.')
PY
