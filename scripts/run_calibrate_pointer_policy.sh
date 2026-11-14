#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
CODEC="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
DATASET="$BASE/codec-learned-project-context-copy-boost20-1k"
COVER="$BASE/codec-project-context-copyability"
HEAD="$BASE/codec-learned-project-context-copy-confidence-aux-1k"
OUT="$BASE/codec-calibrated-pointer-policy"
for file in "$CODEC/config.yaml" "$CODEC/artifact/tokenizer.pt" "$DATASET/train-context.jsonl" "$COVER/augmented.jsonl" "$HEAD/copy_head.pt" "$HEAD/report.json"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
if [ -e "$OUT/report.json" ] || [ -e "$OUT/results.jsonl" ]; then
  echo "Existing results in $OUT; stopping to preserve them." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_calibrate_pointer_policy.py
.venv/bin/python evaluation/test_calibrate_pointer_policy_integration.py
.venv/bin/python - "$HEAD/report.json" "$DATASET/train-context.jsonl" "$COVER/augmented.jsonl" "$CODEC/artifact/tokenizer.pt" <<'PY'
import json, sys
from training.data import sha256_file
r = json.load(open(sys.argv[1]))
if (r['train_sha256'] != sha256_file(sys.argv[2]) or
    r['validation_sha256'] != sha256_file(sys.argv[3]) or
    r['codec_sha256'] != sha256_file(sys.argv[4]) or not r.get('confidence_aware_gate')):
    raise SystemExit('Pointer report does not match the data or codec')
print('Pointer, codec and both datasets match.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python evaluation/calibrate_pointer_policy.py \
  --codec-config "$CODEC/config.yaml" --codec-checkpoint "$CODEC/artifact/tokenizer.pt" \
  --head "$HEAD/copy_head.pt" --train "$DATASET/train-context.jsonl" \
  --validation "$COVER/augmented.jsonl" --output "$OUT" \
  --sample-size 512 --seed 2027 --batch-size 8 --prefix-length 256 --device cuda
.venv/bin/python - "$OUT/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
m = r['metrics']
print('\n=== Calibrated pointer results to share ===')
print('Policy selected on pointer-training sample:', r['selected_policy'])
for key in ('base', 'pointer', 'policy'):
    print(f"{key}: accuracy={m[key+'_token_accuracy']:.4%}; "
          f"exact text={m[key+'_exact_text']}/{r['validation_examples']}; "
          f"exact names={m[key+'_exact_names']}/{m['scorable_names']}")
print(f"Policy fixed={m['policy_fixed']}; regressed={m['policy_regressed']}; "
      f"new exact={m['policy_new_exact']}; lost exact={m['policy_lost_exact']}")
print('Calibration uses pointer-training rows; validation was not used to choose thresholds.')
print('Known-target VQ codec diagnostic only, not a generator or semantic test.')
PY
