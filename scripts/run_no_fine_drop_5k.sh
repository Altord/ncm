#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/bf16-aot-no-fine-drop-5k
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.]train_tokenizer' >/dev/null; then
  echo 'A tokenizer trainer is already running; finish or stop it first.' >&2
  exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ]; then
  echo "A checkpoint already exists in $OUT; stopping to preserve it." >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$CFG" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
source = 'config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml'
cfg = OmegaConf.load(source)
cfg.task = 'ctx256-16level-encoder4-no-fine-drop-5k-screen'
cfg.vqvae.enc_layers = 4
cfg.vqvae.vqvae_vocab_size = [512] * 16
cfg.vqvae.vector_quantizer_config.levels.manual_levels = [1,2,4,8,16,32,48,64,80,96,112,128,160,192,224,256]
cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob = 0.0
cfg.train.precision = 'bf16'
cfg.train.max_steps = 30000
cfg.train.run_dir = sys.argv[2]
cfg.train.artifact_dir = sys.argv[2] + '/artifact'
cfg.train.save_every = 1000
cfg.train.eval_every = 5000
OmegaConf.save(cfg, sys.argv[1])
print('New run: 16 VQ levels; 4 encoder blocks; BF16; fine-level dropout=0; 5,000 updates.', flush=True)
PY
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python -m training.train_tokenizer --config "$CFG" --stop-after-steps 5000
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / 'logs/validation.jsonl').read_text().splitlines() if line.strip()]
row = rows[-1]
print('\n=== No fine-drop 5k experiment summary ===')
print(f"step {row['step']}: accuracy={row['val_token_accuracy']:.2%}, exact={row['val_exact_sequence_match_rate']:.2%}, CE={row['val_recon_ce']:.4f}")
print('Prior 5k reference (fine-drop=0.25): accuracy=90.72%, exact=5.84%, CE=0.6991.')
print('The reference was resumed during training, so data order and attention settings were not fully matched.')
print(f'Checkpoint: {out / "checkpoints/latest.pt"}')
PY
