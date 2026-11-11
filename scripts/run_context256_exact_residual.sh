#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STRESS=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT=runs/context256-codec-comparison/exact-residual
mkdir -p "$OUT"

evaluate() {
  local name="$1" config="$2" checkpoint="$3"
  echo "===== $name: frozen codec plus exact correction sidecar ====="
  .venv/bin/python evaluation/godot_csharp_codec_eval.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --input "$STRESS" \
    --text-key target \
    --compile \
    --exact-residual \
    --output "$OUT/$name.jsonl"
}

evaluate ctx256-flat \
  config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256-weighted-30k.yaml \
  artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx256-weighted-30k/tokenizer.pt

evaluate ctx256-occurrence \
  config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256-occurrence-weighted-30k.yaml \
  artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx256-occurrence-weighted-30k/tokenizer.pt

evaluate ctx256-ast-occurrence \
  config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml \
  artifacts/godot-csharp-deepseek-coder-1.3b-strict/64level-code512-decoder4-ctx256-ast-occurrence-weighted-30k/tokenizer.pt

.venv/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("\n=== Exact residual comparison ===")
print("run                    raw compile  corrected  corrections  sidecar bytes  sidecar/input")
for name in ("ctx256-flat", "ctx256-occurrence", "ctx256-ast-occurrence"):
    report = json.loads((root / f"{name}.report.json").read_text(encoding="utf-8"))
    residual = report["exact_residual"]
    print(
        f"{name:23}"
        f"{report['reconstruction_compile_passes']:>2}/{report['examples']:<8}"
        f"{residual['restored_compile_passes']:>2}/{report['examples']:<8}"
        f"{residual['changed_tokens']:>10}"
        f"{residual['binary_bytes']:>15}"
        f"{residual['residual_to_input_byte_ratio']:>14.1%}"
    )
print(f"Detailed rows and reports: {root.resolve()}")
print("Corrected compilation is expected for intact inputs; compare correction cost.")
print("This sidecar works for known-input round trips and cannot repair generated code alone.")
PY
