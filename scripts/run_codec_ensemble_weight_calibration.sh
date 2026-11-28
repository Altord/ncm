#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/codec-diverse-data-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
OUT="$BASE/codec-ensemble-weight-calibration"
for f in "$A/config.yaml" "$A/checkpoints/step-00005000.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$A/methods-candidate.jsonl" "$A/external-candidate.jsonl" "$B/methods.jsonl" "$BASE/codec-external-methods/b.jsonl" "$METHODS" "$EXTERNAL"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/quarter.jsonl" "$OUT/half.jsonl" "$OUT/threequarter.jsonl" "$OUT/calibration.json" "$OUT/external.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_ensemble_codec_methods.py
.venv/bin/python evaluation/test_ensemble_codec_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for spec in quarter:0.25 half:0.5 threequarter:0.75; do
 name="${spec%:*}"; weight="${spec#*:}"
 .venv/bin/python -m evaluation.ensemble_codec_methods --config-a "$A/config.yaml" --checkpoint-a "$A/checkpoints/step-00005000.pt" --saved-a "$A/methods-candidate.jsonl" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$B/methods.jsonl" \
  --input "$METHODS" --output "$OUT/$name.jsonl" --batch-size 8 --device cuda --weight-a "$weight"
done
.venv/bin/python - "$METHODS" "$OUT" <<'PY'
import hashlib, json, sys
from pathlib import Path
from training.data import sha256_file
source, out = Path(sys.argv[1]), Path(sys.argv[2])
def load(p):
    rows = [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
    d = {r['id']:r for r in rows}
    if not rows or len(d) != len(rows): raise SystemExit('Empty or duplicate IDs')
    return d
original = load(source)
split = {k: int(hashlib.sha256(r['source'].encode()).hexdigest()[:8],16) % 2 == 0 for k,r in original.items()}
if not all(any(v == x for v in split.values()) for x in (True, False)): raise SystemExit('Empty calibration or evaluation split')
results = {}
for name in ('quarter','half','threequarter'):
    path = out/(name+'.jsonl'); rows = load(path)
    report = json.loads(path.with_suffix('.report.json').read_text())
    if rows.keys() != original.keys() or report['input_sha256'] != sha256_file(source): raise SystemExit('Input mismatch')
    results[name] = {'weight': report['weight_a'], 'calibration_exact': sum(rows[k]['selected'] == original[k]['target'] for k in rows if split[k]),
                     'evaluation_exact': sum(rows[k]['selected'] == original[k]['target'] for k in rows if not split[k])}
selected = max(results, key=lambda n: (results[n]['calibration_exact'], -abs(results[n]['weight']-.5)))
summary = {'selected': selected, 'weight_a': results[selected]['weight'], 'calibration_count': sum(split.values()),
           'evaluation_count': sum(not v for v in split.values()), 'results': results, 'input_sha256': sha256_file(source),
           'scope': 'Fixed weights selected by exact reconstruction on source-file-grouped calibration split; evaluation and external labels never select weight.',
           'limitations': 'Development methods and external sets repeatedly used; file grouping does not preclude semantic duplication. Known-target two-codec round trip; not generation.'}
(out/'calibration.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)
PY
WEIGHT=$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["weight_a"])' "$OUT/calibration.json")
.venv/bin/python -m evaluation.ensemble_codec_methods --config-a "$A/config.yaml" --checkpoint-a "$A/checkpoints/step-00005000.pt" --saved-a "$A/external-candidate.jsonl" \
 --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$BASE/codec-external-methods/b.jsonl" \
 --input "$EXTERNAL" --output "$OUT/external.jsonl" --batch-size 8 --device cuda --weight-a "$WEIGHT"
