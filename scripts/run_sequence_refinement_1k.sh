#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
DATA="$BASE/codec-learned-project-context-copy-boost20-1k"
TRAIN="$DATA/train-context.jsonl"
VAL="$BASE/codec-project-context-copyability/augmented.jsonl"
PRIOR="$BASE/codec-context-refinement-hard-1k"
OUT="$BASE/codec-sequence-refinement-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$TRAIN" "$VAL" "$DATA/train-context.manifest.json" "$PRIOR/report.json" "$PRIOR/calibration.json"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/refinement_head.pt" "$OUT/results.jsonl" "$OUT/report.json" "$OUT/calibration.json"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_context_refinement_head.py
.venv/bin/python evaluation/test_context_refinement_integration.py
.venv/bin/python evaluation/test_context_refinement_calibration.py
.venv/bin/python - "$PRIOR/report.json" "$DATA/train-context.manifest.json" "$TRAIN" "$VAL" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
with open(sys.argv[1]) as f: old = json.load(f)
with open(sys.argv[2]) as f: manifest = json.load(f)
train, val, codec = (sha256_file(path) for path in sys.argv[3:])
if (old['train_sha256'] != train or old['validation_sha256'] != val or old['codec_sha256'] != codec or
    manifest['augmented_train_sha256'] != train or manifest['validation_augmented_sha256'] != val):
    raise SystemExit('Training data, validation data, or codec differs from previous experiment')
print('Matched frozen codec and data; sequence attention is the changed architecture.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_context_refinement_screen \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --train "$TRAIN" --validation "$VAL" --output "$OUT" --device cuda \
  --steps 1000 --batch-size 8 --prefix-length 256 --error-boost 20 --trust-weight 0.05 \
  --hard-example-probability 0.5 --sequence-layers 1
.venv/bin/python -m evaluation.calibrate_context_refinement \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --head-checkpoint "$OUT/refinement_head.pt" --validation "$VAL" \
  --output "$OUT/calibration.json" --device cuda --batch-size 8
.venv/bin/python - "$PRIOR/calibration.json" "$OUT/calibration.json" "$OUT/report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: old = json.load(f)
with open(sys.argv[2]) as f: new = json.load(f)
with open(sys.argv[3]) as f: train = json.load(f)
if old['validation_sha256'] != new['validation_sha256'] or old['codec_sha256'] != new['codec_sha256'] or old['split'] != new['split']:
    raise SystemExit('Cannot compare calibration dataset, codec, or source split')
print('\n=== Sequence refinement results to share ===')
for name, report in [('previous hard-example head', old), ('sequence-attention head', new)]:
    r = report['evaluation'][report['selected_index']]
    print(f"{name}: threshold={report['selected_margin']}; exact={r['exact']}/{r['examples']}; token accuracy={r['correct']/r['tokens']:.4%}; fixed={r['fixed']}; new errors={r['introduced']}")
print(f"Sequence head: parameters={train['head_parameters']:,}; scan={train['hard_example_scan_seconds']:.1f}s; training={train['training_seconds']:.1f}s; peak GPU GiB={train['peak_gpu_allocated_gib']}")
print('Thresholds selected on calibration files. Reused evaluation data; target encoded by frozen codec, synthetic signature supplied; not a generator test.')
PY
