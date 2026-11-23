#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
C="$BASE/codec-literal-weight-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXT="$BASE/codec-external-methods"
OUT="$BASE/codec-three-model-ensemble"
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$C/config.yaml" "$C/artifact/tokenizer.pt" \
 "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$METHODS/reference.report.json" "$B/methods.jsonl" "$B/methods.report.json" "$C/methods.jsonl" "$C/methods.report.json" \
 "$EXT/methods.jsonl" "$EXT/a.jsonl" "$EXT/a.report.json" "$EXT/b.jsonl" "$EXT/b.report.json" "$C/external.jsonl" "$C/external.report.json" \
 "$BASE/codec-two-model-ensemble-methods/results.jsonl" "$BASE/codec-two-model-ensemble-methods/results.report.json" "$EXT/ensemble.jsonl" "$EXT/ensemble.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 [ ! -e "$OUT/$name.jsonl" ] && [ ! -e "$OUT/$name.report.json" ] || { echo "Existing output: $OUT/$name; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_ensemble_codec_methods.py
.venv/bin/python evaluation/test_ensemble_codec_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; SA="$METHODS/reference.jsonl"; SB="$B/methods.jsonl"; else INPUT="$EXT/methods.jsonl"; SA="$EXT/a.jsonl"; SB="$EXT/b.jsonl"; fi
 .venv/bin/python -m evaluation.ensemble_codec_methods \
  --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" --saved-a "$SA" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$SB" \
  --config-c "$C/config.yaml" --checkpoint-c "$C/artifact/tokenizer.pt" --saved-c "$C/$name.jsonl" \
  --input "$INPUT" --output "$OUT/$name.jsonl" --batch-size 4 --device cuda
done
.venv/bin/python - "$OUT" "$BASE/codec-two-model-ensemble-methods/results.jsonl" "$EXT/ensemble.jsonl" "$EXT/methods.jsonl" <<'PY'
import json, sys
from pathlib import Path
from collections import defaultdict
out, methods, external, source = sys.argv[1:]
def load(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(rows) != len(result): raise SystemExit(f'Empty or duplicate IDs: {path}')
    return result
for name, old_path in [('methods', methods), ('external', external)]:
    old, new = load(old_path), load(Path(out)/(name+'.jsonl'))
    old_report = json.loads(Path(old_path).with_suffix('.report.json').read_text())
    new_report = json.loads((Path(out)/(name+'.report.json')).read_text())
    for field in ('input_sha256', 'checkpoint_a_sha256', 'checkpoint_b_sha256'):
        if old_report[field] != new_report[field]: raise SystemExit(f'{name}: mismatched {field}')
    if old.keys() != new.keys(): raise SystemExit(f'{name}: IDs differ')
    print(f'\n=== {name}: two versus three codecs ===')
    print(f"Exact: {sum(r['ensemble_exact'] for r in old.values())}/{len(old)} -> {sum(r['ensemble_exact'] for r in new.values())}/{len(new)}")
    print(f"Gained: {sum(not old[k]['ensemble_exact'] and new[k]['ensemble_exact'] for k in old)}; lost: {sum(old[k]['ensemble_exact'] and not new[k]['ensemble_exact'] for k in old)}")
    if name == 'external':
        rows = load(source)
        if rows.keys() != old.keys(): raise SystemExit('External IDs differ')
        groups = defaultdict(lambda: [0,0,0])
        for k, row in rows.items():
            g = groups[row['source'].split('/')[0]]
            g[0] += 1; g[1] += old[k]['ensemble_exact']; g[2] += new[k]['ensemble_exact']
        for project, (n,a,b) in sorted(groups.items()): print(f'{project}: {a}/{n} -> {b}/{n}')
print('Known-target reconstruction; 3 independent codec streams; no training, signatures, compilation, behavior or generator test.')
print('Reused datasets; tokenizer configuration matches but tokenizer file identity is not independently verified.')
PY
