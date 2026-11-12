#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/method-generator-100step
DATA=data/mycode/godot-csharp-method-pairs-ctx256-prompt64
SOURCE=runs/context256-codec-comparison/width128-generator-pilot-20step/config.yaml
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
if [ ! -f "$SOURCE" ] || [ ! -f "$DATA/train.jsonl" ] || [ ! -f "$DATA/validation.jsonl" ]; then
  echo 'Missing pilot config or method-pair dataset.' >&2; exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/generator.pt" ]; then
  echo "Existing checkpoint at $OUT; stopping to protect it." >&2; exit 1
fi
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
echo '===== Verify actual token lengths and dataset ambiguity ====='
.venv/bin/python evaluation/audit_method_pairs.py --dataset "$DATA" --codec-config "$SOURCE" \
  --prefix-length 64 --target-length 256 --output "$OUT/dataset-audit.json"
.venv/bin/python - "$SOURCE" "$CFG" "$OUT" "$DATA" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
source, dest, out, data = sys.argv[1:]
audit = json.loads((Path(out) / 'dataset-audit.json').read_text())
if audit['source_paths_in_both_splits']:
    raise SystemExit('Source paths overlap between train and validation; stopping')
for split, stats in audit['splits'].items():
    if stats['prompt_truncations'] or stats['target_truncations']:
        raise SystemExit(f'{split} has truncated examples with the actual tokenizer; stopping')
cfg = OmegaConf.load(source)
cfg.task = 'csharp-method-signature-generator-100step'
cfg.model.prefix_len = 64
cfg.data.input = data + '/train.jsonl'
cfg.data.cache_dir = out + '/cache-train'
cfg.train.validation_cache_dir = out + '/cache-validation'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.max_steps = 100
cfg.train.eval_every = 100
cfg.train.save_every = 100
cfg.train.log_every = 5
cfg.train.warmup_steps = 5
cfg.train.batch_size = 1
OmegaConf.save(cfg, dest)
rows = [json.loads(line) for line in (Path(data) / 'validation.jsonl').read_text().splitlines() if line.strip()]
sample = Path(out) / 'validation-sample.jsonl'
sample_text = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows[:4])
if sample.exists() and sample.read_text(encoding='utf-8') != sample_text:
    raise SystemExit('Existing validation sample differs; stopping to protect it')
if not sample.exists():
    sample.write_text(sample_text, encoding='utf-8')
print('Fresh 100-step generator run; 64-token prompts; full validation loss; first four validation methods generated.')
print('Identical signatures may have multiple valid bodies; this is signature completion, not natural-language coding.')
PY
echo '===== Prepare training cache ====='
.venv/bin/python -m training.prepare_data --config "$CFG" --stage generator
echo '===== Prepare full validation cache ====='
.venv/bin/python -m training.prepare_data --config "$CFG" --stage generator \
  --input "$DATA/validation.jsonl" --cache-dir "$OUT/cache-validation"
echo '===== Train 100 steps ====='
.venv/bin/python -m training.train_generator --config "$CFG"
echo '===== Generate first four held-out-file methods ====='
.venv/bin/python evaluation/godot_csharp_generator_pilot.py --config "$CFG" \
  --checkpoint "$OUT/artifact/generator.pt" --input "$OUT/validation-sample.jsonl" \
  --output "$OUT/validation-sample-predictions.jsonl"
echo '===== Results to share ====='
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
audit = json.loads((out / 'dataset-audit.json').read_text())
print(f"Dataset: {audit['splits']['train']['examples']} train, {audit['splits']['validation']['examples']} validation; "
      f"ambiguous exact training prompts={audit['ambiguous_training_prompts']}; "
      f"builder token count mismatches={audit['splits']['train']['builder_token_count_mismatches'] + audit['splits']['validation']['builder_token_count_mismatches']}")
for name in ('logs/train.jsonl', 'logs/validation.jsonl'):
    path = out / name
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    row = rows[-1]
    print(f"{name}: step={row.get('step')}; loss={row.get('loss', row.get('val_loss'))}; examples/s={row.get('examples_per_second')}")
print((out / 'validation-sample-predictions.report.json').read_text())
print('Four validation methods are a diagnostic sample; no acceptance or compile tests were run.')
PY
