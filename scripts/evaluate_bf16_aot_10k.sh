#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/bf16-aot-10k
CFG="$OUT/config.yaml"
CKPT="$OUT/artifact/tokenizer.pt"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
PRIOR=runs/context256-codec-comparison/compact-ast-16level/stress.report.json
if [ ! -f "$CFG" ] || [ ! -f "$CKPT" ]; then
  echo "Missing BF16 config or selected checkpoint; complete the 10k run first." >&2
  exit 1
fi
ARGS=()
if [ -f "$PRIOR" ]; then ARGS=(--compare-report "$PRIOR"); fi
echo '===== Evaluate the selected BF16/AOTriton checkpoint on 48 development methods ====='
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$CKPT" --input "$INPUT" --text-key target \
  --compile --exact-residual --measure-packed --progress "${ARGS[@]}" \
  --output "$OUT/stress-10k.jsonl"
.venv/bin/python - "$OUT/stress-10k.report.json" <<'PY'
import json, statistics, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
print('\n=== 10k BF16/AOT stress summary ===')
print(f"Original compiles: {r['original_compile_passes']}/{r['examples']}")
print(f"Raw reconstruction compiles: {r['reconstruction_compile_passes']}/{r['examples']}")
print(f"Exact token sequences: {r['exact_sequence_matches']}/{r['examples']}")
print(f"Token accuracy: {r['valid_token_accuracy']:.2%}; character similarity: {r['mean_character_similarity']:.2%}")
print(f"VQ plus corrections / input bytes: {r['packed_measurement']['total_to_input_ratio']:.2f}x")
print(f"Corrected reconstruction compiles: {r['exact_residual']['restored_compile_passes']}/{r['examples']} (requires original tokens)")
log = Path(sys.argv[1]).parent / 'logs/train.jsonl'
if log.exists():
    rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
    speeds = [row['examples_per_second'] for row in rows[-3:] if row.get('examples_per_second') is not None]
    if speeds:
        rate = statistics.median(speeds)
        print(f"Recent training speed: {rate:.1f} examples/s (~{rate/16:.2f} steps/s at batch size 16)")
print('Prior 16-level zero-block FP32 at 30k: raw compiles 0/48; prior 64-level AST: 28/48.')
print(f"Report: {Path(sys.argv[1]).resolve()}")
PY
