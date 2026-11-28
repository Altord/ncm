#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/codec-diverse-data-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
OUT="$BASE/codec-confidence-gate-screen"
for f in "$A/config.yaml" "$A/checkpoints/step-00005000.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$A/methods-candidate.jsonl" "$A/external-candidate.jsonl" "$B/methods.jsonl" "$BASE/codec-external-methods/b.jsonl" "$METHODS" "$EXTERNAL"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/methods.jsonl" "$OUT/methods.report.json" "$OUT/external.jsonl" "$OUT/external.report.json" "$OUT/calibration.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python -m evaluation.test_codec_confidence_gate_screen
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; SAVED="$B/methods.jsonl"; else INPUT="$EXTERNAL"; SAVED="$BASE/codec-external-methods/b.jsonl"; fi
 .venv/bin/python -m evaluation.codec_confidence_gate_screen --config-a "$A/config.yaml" --checkpoint-a "$A/checkpoints/step-00005000.pt" --saved-a "$A/$name-candidate.jsonl" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$SAVED" \
  --input "$INPUT" --output "$OUT/$name.jsonl" --batch-size 8 --device cuda
done
.venv/bin/python - "$METHODS" "$EXTERNAL" "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
from evaluation.ensemble_codec_methods import load_rows
from training.data import sha256_file
methods, external, out = map(Path, sys.argv[1:])
source, m, e = load_rows(methods), load_rows(out/'methods.jsonl'), load_rows(out/'external.jsonl')
if source.keys() != m.keys(): raise SystemExit('Method IDs differ')
for path, input_path in ((out/'methods.report.json',methods), (out/'external.report.json',external)):
 if json.loads(path.read_text())['input_sha256'] != sha256_file(input_path): raise SystemExit('Input hash mismatch')
split = {k: int(hashlib.sha256(row['source'].encode()).hexdigest()[:8],16) % 2 == 0 for k,row in source.items()}
names = ('equal','margin_half','margin_one','margin_two')
scores = {name: {'calibration_exact': sum(row['exact'][name] for k,row in m.items() if split[k]),
                 'evaluation_exact': sum(row['exact'][name] for k,row in m.items() if not split[k]),
                 'external_exact': sum(row['exact'][name] for row in e.values())} for name in names}
# Only the source-file-grouped calibration results select the mode. Prefer equal on ties.
chosen = max(names, key=lambda name:(scores[name]['calibration_exact'], name == 'equal'))
result = {'selected': chosen, 'calibration_count': sum(split.values()), 'evaluation_count': sum(not v for v in split.values()),
          'scores': scores, 'scope':'Known-target codec; target labels select gate only on calibration split.',
          'limitations':'All sets reused; only three fixed ratios screened. No compiler, behavior or generation test; separate latent streams.'}
(out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
PY
