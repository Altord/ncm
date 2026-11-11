#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CFG=config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml
RUN=runs/godot-csharp-deepseek-coder-1.3b-strict/8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k
CKPT=artifacts/godot-csharp-deepseek-coder-1.3b-strict/8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k/tokenizer.pt
OUT=runs/context256-codec-comparison/compact-ast-8level
mkdir -p "$OUT"
echo '===== Training compact 8-level codec ====='
.venv/bin/python -m training.train_tokenizer --config "$CFG" --max-steps 30000
echo '===== Held-out AST validation and packed size ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --exact-residual --measure-packed \
  --output "$OUT/validation.jsonl"
echo '===== Shared compile stress set ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$CKPT" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --compile --exact-residual --measure-packed \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Compact codec experiment summary ===')
for name in ('validation', 'stress'):
    report = json.loads((root / (name + '.report.json')).read_text())
    packed = report['packed_measurement']
    print(f"{name}: token accuracy={report['valid_token_accuracy']:.2%}; exact={report['exact_sequence_matches']}/{report['examples']}; total/input={packed['total_to_input_ratio']:.2f}x; VQ bytes={packed['vq_code_bytes']}; correction bytes={packed['residual_bytes']}")
    if report['compile_requested']:
        print(f"  compile: original={report['original_compile_passes']}/{report['examples']}; raw reconstruction={report['reconstruction_compile_passes']}/{report['examples']}; corrected={report['exact_residual']['restored_compile_passes']}/{report['examples']}")
print('Prior AST 64-level validation: accuracy=96.32%; exact=92/462; total/input=13.59x.')
print('Prior AST 64-level stress compile: raw reconstruction=28/48.')
print(f'Reports: {root.resolve()}')
PY
