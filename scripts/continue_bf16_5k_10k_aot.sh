#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/bf16-aot-10k
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.]train_tokenizer' >/dev/null; then
  echo 'A tokenizer trainer is already running. Let it finish or stop it before starting this run.' >&2
  exit 1
fi
if [ -f "$OUT/checkpoints/latest.pt" ] && [ -f "$CFG" ]; then
  SOURCE_CFG="$CFG"
  CHECKPOINT="$OUT/checkpoints/latest.pt"
elif [ -f runs/context256-codec-comparison/bf16-screen-3k/checkpoints/latest.pt ]; then
  SOURCE_CFG=runs/context256-codec-comparison/bf16-screen-3k/config.yaml
  CHECKPOINT=runs/context256-codec-comparison/bf16-screen-3k/checkpoints/latest.pt
else
  SOURCE_CFG=runs/context256-codec-comparison/precision-check-1k/bf16.yaml
  CHECKPOINT=runs/context256-codec-comparison/precision-check-1k/bf16/checkpoints/latest.pt
fi
if [ ! -f "$CHECKPOINT" ] || [ ! -f "$SOURCE_CFG" ]; then
  echo "BF16 checkpoint or config missing: $CHECKPOINT / $SOURCE_CFG" >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$SOURCE_CFG" "$CFG" "$OUT" "$CHECKPOINT" <<'PY'
import sys
import torch
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
ckpt = torch.load(sys.argv[4], map_location='cpu', weights_only=False)
completed = int(ckpt.get('step', -1)) + 1
if str(cfg.train.precision).lower() != 'bf16' or int(cfg.train.max_steps) != 30000:
    raise SystemExit('Expected BF16 and a 30,000-step learning-rate schedule')
if completed < 1000 or completed >= 10000:
    raise SystemExit(f'Checkpoint has {completed} updates; expected 1,000 to 9,999')
if int(cfg.vqvae.enc_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 16:
    raise SystemExit('Checkpoint configuration is not the 16-level, four-encoder-block experiment')
cfg.train.run_dir = sys.argv[3]
cfg.train.artifact_dir = sys.argv[3] + '/artifact'
cfg.train.eval_every = 5000
cfg.train.save_every = 1000
OmegaConf.save(cfg, sys.argv[2])
print(f'Checkpoint: {sys.argv[4]} ({completed} completed updates)', flush=True)
print('Target: 10,000 total updates; validation at 5,000 and 10,000; checkpoints every 1,000.', flush=True)
print('BF16 and ROCm experimental AOTriton are enabled; schedule horizon remains 30,000.', flush=True)
PY
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python -m training.train_tokenizer \
  --config "$CFG" --resume "$CHECKPOINT" --stop-after-steps 10000
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print('\n=== BF16 + AOTriton 5k/10k summary ===')
for line in (out / 'logs/validation.jsonl').read_text().splitlines():
    if line.strip():
        row = json.loads(line)
        print(f"step {row['step']}: accuracy={row['val_token_accuracy']:.2%}, exact={row['val_exact_sequence_match_rate']:.2%}, CE={row['val_recon_ce']:.4f}")
print('These are early codec validation metrics; the raw compile test was not run.')
print(f'Latest checkpoint: {out / "checkpoints/latest.pt"}')
print('Prior zero-block, FP32 16-level run at 10k: accuracy=85.91%, exact=4.98%, CE=1.0434; encoder and precision differ.')
PY
