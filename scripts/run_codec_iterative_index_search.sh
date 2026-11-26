#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXTERNAL="$BASE/codec-external-methods"
PRIOR="$BASE/codec-index-search-screen"
OUT="$BASE/codec-iterative-index-search"
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$METHODS/reference.report.json" "$EXTERNAL/methods.jsonl" "$EXTERNAL/a.jsonl" "$EXTERNAL/a.report.json" "$PRIOR/methods.jsonl" "$PRIOR/methods.report.json" "$PRIOR/external.jsonl" "$PRIOR/external.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 [ ! -e "$OUT/$name.jsonl" ] && [ ! -e "$OUT/$name.report.json" ] || { echo "Existing output: $OUT/$name; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_index_search.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; BASELINE="$METHODS/reference.jsonl"; COUNT=64; else INPUT="$EXTERNAL/methods.jsonl"; BASELINE="$EXTERNAL/a.jsonl"; COUNT=70; fi
 .venv/bin/python -m evaluation.codec_index_search --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
  --input "$INPUT" --baseline "$BASELINE" --max-examples "$COUNT" --budget 4 --rounds 3 --device cuda --output "$OUT/$name.jsonl"
done
.venv/bin/python - "$PRIOR" "$OUT" <<'PY'
import json, sys
from pathlib import Path
prior, current = map(Path, sys.argv[1:])
def load(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    data = {r['id']: r for r in rows}
    if not rows or len(data) != len(rows): raise SystemExit(f'Empty or duplicate IDs: {path}')
    return data
for name in ('methods', 'external'):
    a, b = (json.loads((path / f'{name}.report.json').read_text()) for path in (prior, current))
    if any(a[key] != b[key] for key in ('input_sha256', 'checkpoint_sha256', 'examples')):
        raise SystemExit(f'{name}: report inputs differ')
    old, new = load(prior/f'{name}.jsonl'), load(current/f'{name}.jsonl')
    if old.keys() != new.keys(): raise SystemExit(f'{name}: example IDs differ')
    print(f'\n=== {name}: one round versus up to three ===')
    print(f"Exact: {a['selected_exact']}/{a['examples']} -> {b['selected_exact']}/{b['examples']}")
    print(f"Exact gains: {sum(not old[k]['selected_exact'] and new[k]['selected_exact'] for k in old)}; losses: {sum(old[k]['selected_exact'] and not new[k]['selected_exact'] for k in old)}")
    print(f"Token errors: {a['selected_token_errors']} -> {b['selected_token_errors']}; proposals {a['attempts']} -> {b['attempts']}")
    print(f"Screen seconds: {a['seconds']:.2f} -> {b['seconds']:.2f}; excludes model loading and is not end-to-end timing")
print('Known-source encoder search; full discrete indices are saved. No training, compilation, behavior or generation test.')
PY
