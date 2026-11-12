#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
SOURCE="$BASE/compact-ar64-5k/config.yaml"
OUT="$BASE/compact-parallel64-5k"
CFG="$OUT/config.yaml"
mkdir -p "$OUT"
if [ ! -f "$SOURCE" ]; then
  echo "Missing prior experiment config: $SOURCE" >&2
  exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/tokenizer.pt" ]; then
  echo "Existing checkpoint at $OUT; stopping to protect it." >&2
  exit 1
fi
.venv/bin/python - "$SOURCE" "$CFG" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
source, dest, out = sys.argv[1:]
cfg = OmegaConf.load(source)
if (int(cfg.vqvae.compression_factor) != 4 or
    list(cfg.vqvae.vector_quantizer_config.levels.manual_levels) != [64] or
    not cfg.vqvae.get('autoregressive_decoder', False)):
    raise ValueError('Prior config is not the expected autoregressive 64-index experiment')
cfg.task = 'compact-parallel64-5k'
cfg.vqvae.autoregressive_decoder = False
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
OmegaConf.save(cfg, dest)
print('Same 64-index layout and training settings as compact-ar64-5k; parallel decoder.')
print('Config:', dest, flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$CFG" --stop-after-steps 5000
echo '===== Held-out validation ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --batch-size 16 --exact-residual --measure-packed \
  --output "$OUT/validation.jsonl"
echo '===== Shared compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --batch-size 16 --compile --progress --exact-residual --measure-packed \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Parallel 64-index 5k results to share ===')
for name in ('validation', 'stress'):
    r = json.loads((root / f'{name}.report.json').read_text())
    p = r['packed_measurement']
    print(f"{name}: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; VQ={p['vq_code_bytes']} B; corrections={p['residual_bytes']} B; total/input={p['total_to_input_ratio']:.2f}x; decode={p['decode_and_argmax_seconds']:.2f}s")
    if r['compile_requested']:
        print(f"  raw compile={r['reconstruction_compile_passes']}/{r['examples']}; original compile={r['original_compile_passes']}/{r['examples']}")
print('Autoregressive 64-index 5k reference: validation 45.46%; stress 28.99%, 0/48 raw compile, 1.37x total/input.')
print('Exact correction sidecars require original input and are unavailable for unknown generated targets.')
PY
