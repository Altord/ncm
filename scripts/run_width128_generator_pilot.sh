#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/width128-generator-pilot-20step
CODEC=runs/context256-codec-comparison/bf16-aot-64level-encoder4-nodrop-width128-5k
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2
  exit 1
fi
if [ ! -f "$CODEC/config.yaml" ] || [ ! -f "$CODEC/artifact/tokenizer.pt" ]; then
  echo "Missing codec config or artifact at $CODEC" >&2
  exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/generator.pt" ]; then
  echo "Existing generator checkpoint in $OUT; stopping to protect it." >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$CODEC" "$CFG" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
from training.data import load_base_tokenizer, iter_records, encode_text
codec_dir, dest, out = sys.argv[1:]
codec = OmegaConf.load(codec_dir + '/config.yaml')
cfg = OmegaConf.load('config/train/godot-csharp-generator-pilot.yaml')
cfg.tokenizer = codec.tokenizer
cfg.vqvae = codec.vqvae
cfg.vqvae.checkpoint_path = codec_dir + '/artifact/tokenizer.pt'
cfg.model.context_length = int(codec.vqvae.context_length)
cfg.model.in_vocab_size = int(codec.vqvae.in_vocab_size)
cfg.model.vocab_size = list(codec.vqvae.vqvae_vocab_size)[1:]
cfg.model.predict_level0 = True
cfg.model.query_level0 = False
tok = load_base_tokenizer(cfg.tokenizer)
cfg.model.prefix_pad_token_id = int(tok.pad_token_id)
cfg.data.cache_dir = out + '/cache-train'
cfg.data.encode_batch_size = 4
cfg.train.validation_cache_dir = out + '/cache-validation'
cfg.train.precision = 'bf16'
cfg.train.batch_size = 1
cfg.train.max_steps = 20
cfg.train.warmup_steps = 5
cfg.train.log_every = 1
cfg.train.save_every = 20
cfg.train.eval_every = 20
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
for split in ('train', 'validation', 'test'):
    path = f'examples/godot_csharp_generator_pilot/{split}.jsonl'
    rows = list(iter_records(path, 'jsonl'))
    long_prompts = sum(len(tok.encode(row['prompt'])) > int(cfg.model.prefix_len) for row in rows)
    long_targets = sum(len(encode_text(tok, row['target'], True)) > int(cfg.model.context_length) for row in rows)
    print(f'{split}: {len(rows)} examples; truncated prompts={long_prompts}; truncated targets={long_targets}', flush=True)
    if long_targets:
        raise SystemExit('Targets exceed codec context; stopping before data preparation')
OmegaConf.save(cfg, dest)
print('20-step pipeline pilot; level 0 is predicted from prompt; outputs are not a coding-quality claim.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
echo '===== Prepare training data ====='
.venv/bin/python -m training.prepare_data --config "$CFG" --stage generator
echo '===== Prepare validation data ====='
.venv/bin/python -m training.prepare_data --config "$CFG" --stage generator \
  --input examples/godot_csharp_generator_pilot/validation.jsonl \
  --cache-dir "$OUT/cache-validation"
echo '===== Train 20 steps ====='
.venv/bin/python -m training.train_generator --config "$CFG"
echo '===== Generate and compile six pilot examples ====='
.venv/bin/python evaluation/godot_csharp_generator_pilot.py \
  --config "$CFG" --checkpoint "$OUT/artifact/generator.pt" \
  --input examples/godot_csharp_generator_pilot/test.jsonl \
  --output "$OUT/predictions.jsonl" --compile
echo '===== Result to share ====='
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
for name in ('logs/train.jsonl', 'logs/validation.jsonl'):
    path = out / name
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if rows:
            row = rows[-1]
            print(f"{name}: step={row.get('step')}, loss={row.get('loss', row.get('val_loss'))}, examples/s={row.get('examples_per_second')}")
report = out / 'predictions.report.json'
if report.exists():
    print(report.read_text())
else:
    print('Generation finished; inspect predictions and log:', out)
print('Tiny 20-step pipeline test; not an independent code-quality assessment.')
PY
