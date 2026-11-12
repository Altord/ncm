#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/bf16-aot-64level-encoder4-nodrop-id3-5k
CACHE=data/cache/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/tokenizer-id3
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.]train_tokenizer' >/dev/null; then
  echo 'A tokenizer trainer is already running; finish or stop it first.' >&2
  exit 1
fi
mkdir -p "$OUT"
if [ ! -f "$CFG" ]; then
  .venv/bin/python - "$CFG" "$OUT" "$CACHE" <<'PY'
import sys
from omegaconf import OmegaConf
source = 'config/train/godot-csharp-deepseek-coder-1.3b-strict-64level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml'
cfg = OmegaConf.load(source)
cfg.task = 'ctx256-64level-encoder4-nodrop-identifier3-5k'
cfg.vqvae.enc_layers = 4
cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob = 0.0
cfg.train.precision = 'bf16'
cfg.train.reconstruction_token_weights.identifier = 3.0
cfg.train.max_steps = 30000
cfg.train.run_dir = sys.argv[2]
cfg.train.artifact_dir = sys.argv[2] + '/artifact'
cfg.train.save_every = 1000
cfg.train.eval_every = 5000
cfg.data.cache_dir = sys.argv[3]
OmegaConf.save(cfg, sys.argv[1])
print('64 levels, encoder4, no fine-drop; identifier/literal weight 3.0 instead of 1.5.', flush=True)
PY
fi
if [ ! -f "$CACHE/manifest.json" ]; then
  echo '===== Prepare separate training cache with identifier weight 3.0 ====='
  .venv/bin/python -m training.prepare_data --config "$CFG" --stage tokenizer
else
  echo "Using existing dedicated cache: $CACHE"
fi
RESUME=()
if [ -f "$OUT/checkpoints/latest.pt" ]; then
  RESUME=(--resume "$OUT/checkpoints/latest.pt")
  echo 'Resuming the existing identifier-weight experiment checkpoint.'
fi
echo '===== Train 5,000-step identifier-weight experiment ====='
env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  .venv/bin/python -m training.train_tokenizer --config "$CFG" "${RESUME[@]}" --stop-after-steps 5000
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / 'logs/validation.jsonl').read_text().splitlines() if line.strip()]
row = rows[-1]
print('\n=== Identifier-weight 5k summary ===')
print(f"step {row['step']}: accuracy={row['val_token_accuracy']:.2%}, exact={row['val_exact_sequence_match_rate']:.2%}, CE={row['val_recon_ce']:.4f}")
print('Prior no-drop reference: accuracy=96.81%, exact=22.73%, CE=0.3085.')
print('Training weight changed for identifiers AND literals; validation metrics remain unweighted.')
print(f'Checkpoint: {out / "checkpoints/latest.pt"}')
PY
