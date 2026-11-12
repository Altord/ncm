#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison/speed-screen
mkdir -p "$BASE"
if [ -e "$BASE/fp32/checkpoints/latest.pt" ] || [ -e "$BASE/aot/checkpoints/latest.pt" ] || [ -e "$BASE/bf16/checkpoints/latest.pt" ]; then
  echo "Speed-screen checkpoints already exist; stopping to preserve earlier results."
  exit 1
fi
.venv/bin/python - "$BASE" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
root = Path(sys.argv[1])
source = 'config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml'
for name in ('fp32', 'aot', 'bf16'):
    cfg = OmegaConf.load(source)
    cfg.task = 'codec-speed-screen-' + name
    cfg.vqvae.enc_layers = 4
    cfg.vqvae.vqvae_vocab_size = [512] * 16
    cfg.vqvae.vector_quantizer_config.levels.manual_levels = [1,2,4,8,16,32,48,64,80,96,112,128,160,192,224,256]
    cfg.train.precision = 'bf16' if name == 'bf16' else 'fp32'
    cfg.train.max_steps = 30000
    cfg.train.run_dir = str(root / name)
    cfg.train.artifact_dir = str(root / name / 'artifact')
    OmegaConf.save(cfg, root / (name + '.yaml'))
PY
if [ $? -ne 0 ]; then exit 1; fi

for name in fp32 aot bf16; do
  echo "===== $name: 300-step speed screen ====="
  flag=0
  if [ "$name" = aot ]; then flag=1; fi
  if env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="$flag" \
    .venv/bin/python -m training.train_tokenizer \
      --config "$BASE/$name.yaml" --stop-after-steps 300; then
    echo "$name: completed"
  else
    echo "$name: failed; check the error above"
  fi
done

.venv/bin/python - "$BASE" <<'PY'
import json, math, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== GPU speed screen summary ===')
for name in ('fp32', 'aot', 'bf16'):
    log = root / name / 'logs/train.jsonl'
    val = root / name / 'logs/validation.jsonl'
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()] if log.exists() else []
    recent = [r['examples_per_second'] for r in rows if r.get('step') in (200, 300) and math.isfinite(r.get('examples_per_second', float('nan')))]
    validation = [json.loads(x) for x in val.read_text().splitlines() if x.strip()] if val.exists() else []
    if not recent or not validation or validation[-1].get('step') != 300:
        print(f'{name}: incomplete or failed; inspect output above')
        continue
    rate = statistics.median(recent)
    v = validation[-1]
    print(f"{name}: {rate:.1f} examples/s, ~{rate/16:.2f} steps/s; validation accuracy={v['val_token_accuracy']:.2%}; CE={v['val_recon_ce']:.4f}")
print('300 steps measure speed and startup stability; they cannot establish final model quality.')
print('Time estimates exclude startup, validation, checkpoints, and later evaluations.')
print(f'Logs: {root.resolve()}')
PY
