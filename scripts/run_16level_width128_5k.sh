#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder4-nodrop-width128-5k"
OUT="$BASE/bf16-aot-16level-encoder4-nodrop-width128-5k"
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.]train_tokenizer' >/dev/null; then
  echo 'A tokenizer trainer is already running; finish or stop it first.' >&2
  exit 1
fi
if [ ! -f "$REF/config.yaml" ]; then
  echo "Missing reference configuration: $REF/config.yaml" >&2
  exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/tokenizer.pt" ]; then
  echo "Existing checkpoint in $OUT; stopping to protect it." >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$REF/config.yaml" "$CFG" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
source, dest, out = sys.argv[1:]
cfg = OmegaConf.load(source)
if (int(cfg.vqvae.embed_dim) != 128 or int(cfg.vqvae.enc_layers) != 4 or
    int(cfg.vqvae.dec_layers) != 4 or
    len(cfg.vqvae.vqvae_vocab_size) != 64 or
    float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0 or
    float(cfg.train.get('vq_loss_weight', 1.0)) != 1.0):
    raise ValueError('Reference config does not match expected 64-level no-drop model')
cfg.task = 'ctx256-16level-encoder4-nodrop-width128-5k'
cfg.vqvae.vqvae_vocab_size = [512] * 16
cfg.vqvae.vector_quantizer_config.levels.manual_levels = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
OmegaConf.save(cfg, dest)
print('Changed 64 levels to 16; width 128, no-drop, encoder4, decoder4 and loss unchanged.')
print('Validation every 1000 updates; selected checkpoint uses validation token accuracy.')
print('Fixed-width VQ index count is 1,423 per chunk, down from 8,194.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$CFG" --stop-after-steps 5000
echo '===== Held-out validation reconstruction ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --batch-size 16 --exact-residual --measure-packed \
  --output "$OUT/validation.jsonl"
echo '===== Compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --batch-size 16 --compile --progress --exact-residual --measure-packed \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== 16-level width 128 screen results to share ===')
for name in ('validation', 'stress'):
    r = json.loads((root / f'{name}.report.json').read_text())
    p = r['packed_measurement']
    print(f"{name}: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; total/input={p['total_to_input_ratio']:.2f}x; VQ={p['vq_code_bytes']} B; corrections={p['residual_bytes']} B")
    if r['compile_requested']:
        print(f"  raw compiles={r['reconstruction_compile_passes']}/{r['examples']}; original compiles={r['original_compile_passes']}/{r['examples']}")
print('Width 128 reference: stress accuracy=98.41%; exact=33/48; raw compile=36/48.')
print('Validation uses FP32 evaluation; model size, memory and runtime are not included in packed bytes.')
PY
