#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison/precision-check-1k
SOURCE=config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml
mkdir -p "$BASE"
if [ -e "$BASE/fp32/checkpoints/latest.pt" ] || [ -e "$BASE/bf16/checkpoints/latest.pt" ]; then
  echo "Precision-check checkpoint already exists; stopping to preserve results." >&2
  exit 1
fi
.venv/bin/python - "$BASE" "$SOURCE" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
root, source = Path(sys.argv[1]), sys.argv[2]
for name in ('fp32', 'bf16'):
    cfg = OmegaConf.load(source)
    cfg.task = 'codec-precision-check-1k-' + name
    cfg.vqvae.enc_layers = 4
    cfg.vqvae.vqvae_vocab_size = [512] * 16
    cfg.vqvae.vector_quantizer_config.levels.manual_levels = [1,2,4,8,16,32,48,64,80,96,112,128,160,192,224,256]
    cfg.train.precision = name
    cfg.train.max_steps = 30000
    cfg.train.run_dir = str(root / name)
    cfg.train.artifact_dir = str(root / name / 'artifact')
    OmegaConf.save(cfg, root / (name + '.yaml'))
PY
for name in fp32 bf16; do
  echo "===== $name: 1,000-step matched screen ====="
  env TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 \
    .venv/bin/python -m training.train_tokenizer \
    --config "$BASE/$name.yaml" --stop-after-steps 1000
done
.venv/bin/python - "$BASE" <<'PY'
import json, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Matched 1,000-step precision check ===')
for name in ('fp32', 'bf16'):
    rows = [json.loads(x) for x in (root/name/'logs/train.jsonl').read_text().splitlines() if x.strip()]
    validation = [json.loads(x) for x in (root/name/'logs/validation.jsonl').read_text().splitlines() if x.strip()][-1]
    recent = [r['examples_per_second'] for r in rows if r.get('step', 0) >= 600]
    rate = statistics.median(recent)
    print(f"{name}: {rate:.1f} examples/s ({rate/16:.2f} steps/s); validation accuracy={validation['val_token_accuracy']:.2%}; exact={validation['val_exact_sequence_match_rate']:.2%}; CE={validation['val_recon_ce']:.4f}")
print('Early matched comparison only; long-run accuracy and compilation still require evaluation.')
print(f'Reports and checkpoints: {root.resolve()}')
PY
