#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=runs/godot-csharp-deepseek-coder-1.3b-strict/8level-code512-decoder4-ctx256-encoder2-ast-occurrence-30k
ARTIFACT=artifacts/godot-csharp-deepseek-coder-1.3b-strict/8level-code512-decoder4-ctx256-encoder2-ast-occurrence-30k
OUT=runs/context256-codec-comparison/compact-encoder2
CFG="$OUT/config.yaml"
mkdir -p "$OUT"
if [ -e "$RUN/checkpoints/latest.pt" ]; then
  echo "A checkpoint already exists in $RUN; stopping to avoid overwriting an earlier run."
  exit 1
fi
.venv/bin/python - "$CFG" "$RUN" "$ARTIFACT" <<'PY'
import sys
from omegaconf import OmegaConf
source = 'config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml'
cfg = OmegaConf.load(source)
cfg.task = 'godot-csharp-ctx256-8level-encoder2-ast-occurrence-30k'
cfg.vqvae.enc_layers = 2
cfg.train.run_dir = sys.argv[2]
cfg.train.artifact_dir = sys.argv[3]
OmegaConf.save(cfg, sys.argv[1])
print('Encoder depth 2 configuration:', sys.argv[1], flush=True)
print('VQ positions per 256-token chunk:', sum(cfg.vqvae.vector_quantizer_config.levels.manual_levels), flush=True)
PY
echo '===== Train encoder depth 2 codec ====='
.venv/bin/python -m training.train_tokenizer --config "$CFG" --max-steps 30000
echo '===== Held-out validation ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$ARTIFACT/tokenizer.pt" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --exact-residual --measure-packed \
  --output "$OUT/validation.jsonl"
echo '===== Shared compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$ARTIFACT/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --compile --exact-residual --measure-packed \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Encoder depth 2 codec experiment summary ===')
for name in ('validation', 'stress'):
    report = json.loads((root / (name + '.report.json')).read_text())
    packed = report['packed_measurement']
    print(f"{name}: token accuracy={report['valid_token_accuracy']:.2%}; exact={report['exact_sequence_matches']}/{report['examples']}; total/input={packed['total_to_input_ratio']:.2f}x; VQ={packed['vq_code_bytes']} B; corrections={packed['residual_bytes']} B")
    if report['compile_requested']:
        print(f"  compile: original={report['original_compile_passes']}/{report['examples']}; raw={report['reconstruction_compile_passes']}/{report['examples']}; corrected={report['exact_residual']['restored_compile_passes']}/{report['examples']}")
print('Encoder depth 0, 8-level reference: validation accuracy=79.58%, total/input=0.83x; raw stress compile=0/48.')
print(f'Reports: {root.resolve()}')
PY
