#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
TRAIN=data/mycode/godot-csharp-method-pairs-ctx256-prompt64/train.jsonl
VALID="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
OUT="$BASE/codec-learned-prompt-copy-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen)' >/dev/null; then
  echo 'A training process is already running; finish or stop it first.' >&2; exit 1
fi
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$TRAIN" "$VALID"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/copy_head.pt" ] || [ -e "$OUT/report.json" ] || [ -e "$OUT/results.jsonl" ]; then
  echo "Existing results in $OUT; stopping to protect them." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_prompt_copy_head.py
.venv/bin/python evaluation/test_prompt_copy_screen_integration.py
.venv/bin/python - "$CODEC/config.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
if len(cfg.vqvae.vector_quantizer_config.levels.manual_levels) != 64 or int(cfg.vqvae.embed_dim) != 192:
    raise SystemExit('Unexpected codec; refusing to run')
print('64-level width-192 codec preflight passed.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_prompt_copy_screen \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --train "$TRAIN" --validation "$VALID" --output "$OUT" \
  --device cuda --steps 1000 --batch-size 8 --prefix-length 64
