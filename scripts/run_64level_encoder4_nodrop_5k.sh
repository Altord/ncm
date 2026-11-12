#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/bf16-aot-64level-encoder4-nodrop-5k
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.]train_tokenizer' >/dev/null; then
  echo 'A tokenizer trainer is running; finish or stop it first.' >&2
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
source = 'config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml'
cfg = OmegaConf.load(source)
cfg.task = 'ctx256-64level-encoder4-bf16-nodrop-5k-screen'
cfg.vqvae.enc_layers = 4
cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob = 0.0
cfg.train.precision = 'bf16'
cfg.train.max_steps = 30000
cfg.train.run_dir = sys.argv[2]
cfg.train.artifact_dir = sys.argv[2] + '/artifact'
cfg.train.save_every = 1000
cfg.train.eval_every = 5000
OmegaConf.save(cfg, sys.argv[1])
print('New run: 64 VQ levels; 4 encoder blocks; BF16; 5,000 updates.', flush=True)
print('Fine-level dropout is disabled; schedule horizon is 30k.', flush=True)
PY
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python -m training.train_tokenizer --config "$CFG" --stop-after-steps 5000
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / 'logs/validation.jsonl').read_text().splitlines() if line.strip()]
row = rows[-1]
print('\n=== 64-level encoder4 no-drop 5k screen ===')
print(f"step {row['step']}: accuracy={row['val_token_accuracy']:.2%}, exact={row['val_exact_sequence_match_rate']:.2%}, CE={row['val_recon_ce']:.4f}")
print('Compare this result with the 64-level encoder4 baseline at the same step.')
print('No-drop changes the training objective and may reduce generalization.')
print('No compile test was run. The 64-level representation is substantially larger.')
print(f'Checkpoint: {out / "checkpoints/latest.pt"}')
PY
