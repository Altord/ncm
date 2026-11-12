#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
OLD=runs/godot-csharp-deepseek-coder-1.3b-strict/16level-code512-decoder4-ctx256-encoder4-ast-occurrence-30k
SOURCE_CFG=runs/context256-codec-comparison/compact-16level-encoder4/config.yaml
RUN=runs/godot-csharp-deepseek-coder-1.3b-strict/16level-code512-decoder4-ctx256-encoder4-fast-resume-30k
ARTIFACT=artifacts/godot-csharp-deepseek-coder-1.3b-strict/16level-code512-decoder4-ctx256-encoder4-fast-resume-30k
OUT=runs/context256-codec-comparison/compact-16level-encoder4-fast-resume
CFG="$OUT/config.yaml"
CHECKPOINT="$OLD/checkpoints/latest.pt"
if [ ! -f "$CHECKPOINT" ] || [ ! -f "$SOURCE_CFG" ]; then
  echo "Original checkpoint or run config missing; cannot resume." >&2
  exit 1
fi
if [ -e "$RUN/checkpoints/latest.pt" ]; then
  echo "Fast-resume checkpoint already exists; stopping to preserve it." >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$SOURCE_CFG" "$CFG" "$RUN" "$ARTIFACT" "$CHECKPOINT" <<'PY'
import sys, torch
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
cfg.train.run_dir, cfg.train.artifact_dir = sys.argv[3], sys.argv[4]
OmegaConf.save(cfg, sys.argv[2])
ckpt = torch.load(sys.argv[5], map_location='cpu', weights_only=False)
print(f"Resuming checkpoint: {sys.argv[5]}, stored step={ckpt.get('step')}", flush=True)
PY
echo "ROCm experimental AOTriton: $TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"
.venv/bin/python -m training.train_tokenizer --config "$CFG" --resume "$CHECKPOINT" --max-steps 30000
echo '===== Held-out validation ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$CFG" --checkpoint "$ARTIFACT/tokenizer.pt" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --exact-residual --measure-packed --output "$OUT/validation.jsonl"
echo '===== Shared compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$CFG" --checkpoint "$ARTIFACT/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --compile --exact-residual --measure-packed --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Fast-resume codec summary ===')
for name in ('validation', 'stress'):
    r = json.loads((root / (name + '.report.json')).read_text())
    print(f"{name}: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; total/input={r['packed_measurement']['total_to_input_ratio']:.2f}x")
    if r['compile_requested']:
        print(f"  raw compile={r['reconstruction_compile_passes']}/{r['examples']}; corrected compile={r['exact_residual']['restored_compile_passes']}/{r['examples']}")
print(f'Reports: {root.resolve()}')
PY
