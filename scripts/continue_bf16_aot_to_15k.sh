#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SOURCE=runs/context256-codec-comparison/bf16-aot-10k
OUT=runs/context256-codec-comparison/bf16-aot-15k
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.]train_tokenizer' >/dev/null; then
  echo 'A tokenizer trainer is already running; finish or stop it before starting this continuation.' >&2
  exit 1
fi
if [ -f "$OUT/checkpoints/latest.pt" ] && [ -f "$CFG" ]; then
  SOURCE_CFG="$CFG"
  CHECKPOINT="$OUT/checkpoints/latest.pt"
else
  SOURCE_CFG="$SOURCE/config.yaml"
  CHECKPOINT="$SOURCE/checkpoints/latest.pt"
fi
if [ ! -f "$CHECKPOINT" ] || [ ! -f "$SOURCE_CFG" ]; then
  echo "Missing source checkpoint or configuration: $CHECKPOINT / $SOURCE_CFG" >&2
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
if int(cfg.vqvae.enc_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 16:
    raise SystemExit('Expected the 16-level, four-encoder-block configuration')
if completed < 10000 or completed >= 15000:
    raise SystemExit(f'Checkpoint has {completed} completed updates; expected 10,000 to 14,999')
cfg.train.run_dir = sys.argv[3]
cfg.train.artifact_dir = sys.argv[3] + '/artifact'
cfg.train.eval_every = 5000
cfg.train.save_every = 1000
OmegaConf.save(cfg, sys.argv[2])
print(f'Resuming {sys.argv[4]} at {completed} completed updates; stopping at 15,000.', flush=True)
print('BF16 and AOTriton enabled; validation at 15,000; checkpoints every 1,000.', flush=True)
PY
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python -m training.train_tokenizer \
  --config "$CFG" --resume "$CHECKPOINT" --stop-after-steps 15000
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print('\n=== BF16 + AOTriton 15k continuation summary ===')
for line in (out / 'logs/validation.jsonl').read_text().splitlines():
    if line.strip():
        row = json.loads(line)
        print(f"step {row['step']}: accuracy={row['val_token_accuracy']:.2%}, exact={row['val_exact_sequence_match_rate']:.2%}, CE={row['val_recon_ce']:.4f}")
print('10k reference: accuracy=94.48%, exact=11.69%, CE=0.4398.')
print('Training validation uses BF16 autocast; this does not test generated code.')
print(f'Latest checkpoint: {out / "checkpoints/latest.pt"}')
PY
