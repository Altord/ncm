#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/packed-measurement
mkdir -p "$OUT"
for name in flat occurrence ast-occurrence; do
  suffix=""
  if [ "$name" != flat ]; then suffix="-$name"; fi
  echo "===== $name: stress set ====="
  .venv/bin/python evaluation/godot_csharp_codec_eval.py \
    --config "config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256${suffix}-weighted-30k.yaml" \
    --checkpoint "artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx256${suffix}-weighted-30k/tokenizer.pt" \
    --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
    --text-key target --exact-residual --measure-packed \
    --output "$OUT/$name-stress.jsonl"
done
echo "===== AST occurrence: held-out corpus ====="
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml \
  --checkpoint artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx256-ast-occurrence-weighted-30k/tokenizer.pt \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --exact-residual --measure-packed \
  --output "$OUT/ast-occurrence-validation.jsonl"
echo "===== Prompt copy opportunity: validation ====="
.venv/bin/python evaluation/inspect_prompt_copy_opportunities.py \
  --input examples/godot_csharp_generator_pilot/validation.jsonl \
  --output "$OUT/prompt-copy-validation.json"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Packed codec summary ===')
for name in ('flat-stress', 'occurrence-stress', 'ast-occurrence-stress', 'ast-occurrence-validation'):
    report = json.loads((root / (name + '.report.json')).read_text())
    item = report['packed_measurement']
    residual = report['exact_residual']
    print(f"{name}: {report['examples']} rows, VQ={item['vq_code_bytes']} B, corrections={item['residual_bytes']} B, total={item['total_bytes']} B, input={item['input_utf8_bytes']} B, ratio={item['total_to_input_ratio']:.2f}x, decode={item['decode_and_argmax_seconds']:.2f}s, exact text={residual['restored_text_exact']}/{report['examples']}")
copy = json.loads((root / 'prompt-copy-validation.json').read_text())
print(f"Prompt overlap: {copy['exact_prompt_overlap_occurrences']}/{copy['target_identifier_occurrences']} target identifier occurrences ({copy['exact_prompt_overlap_rate']:.1%}); {copy['examples']} pilot validation examples")
print(f"Reports: {root.resolve()}")
PY
