#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE=runs/context256-codec-comparison/precision-check-1k/bf16
SOURCE_CFG=runs/context256-codec-comparison/precision-check-1k/bf16.yaml
OUT=runs/context256-codec-comparison/bf16-screen-3k
CFG="$OUT/config.yaml"
CHECKPOINT="$SOURCE/checkpoints/latest.pt"
if [ ! -f "$CHECKPOINT" ] || [ ! -f "$SOURCE_CFG" ]; then
  echo "The 1,000-step BF16 checkpoint or config is missing; cannot continue." >&2
  exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ]; then
  echo "The 3,000-step screen already has a checkpoint; stopping to preserve it." >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$SOURCE_CFG" "$CFG" "$OUT" "$CHECKPOINT" <<'PY'
import sys
import torch
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
ckpt = torch.load(sys.argv[4], map_location='cpu', weights_only=False)
if int(ckpt.get('step', -1)) != 999:
    raise SystemExit(f"Expected a 1,000-update checkpoint; stored step is {ckpt.get('step')}")
if str(cfg.train.precision).lower() != 'bf16' or int(cfg.train.max_steps) != 30000:
    raise SystemExit('Expected BF16 with a 30,000-step learning-rate schedule')
cfg.train.run_dir = sys.argv[3]
cfg.train.artifact_dir = sys.argv[3] + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
OmegaConf.save(cfg, sys.argv[2])
print('Continuing BF16 from 1,000 to 3,000 updates; validation and checkpoints every 1,000.', flush=True)
PY
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 \
  .venv/bin/python -m training.train_tokenizer \
  --config "$CFG" --resume "$CHECKPOINT" --stop-after-steps 3000
.venv/bin/python - "$SOURCE" "$OUT" <<'PY'
import json, sys
from pathlib import Path
source, out = map(Path, sys.argv[1:])
print('\n=== BF16 early continuation summary ===')
for name, root in (('original', source), ('continued', out)):
    path = root / 'logs/validation.jsonl'
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        print(f"{name} step {row['step']}: accuracy={row['val_token_accuracy']:.2%}, exact={row['val_exact_sequence_match_rate']:.2%}, CE={row['val_recon_ce']:.4f}")
print('The 30,000-step learning-rate schedule is preserved; data-loader order restarts on resume.')
print(f'Latest checkpoint: {out / "checkpoints/latest.pt"}')
PY
