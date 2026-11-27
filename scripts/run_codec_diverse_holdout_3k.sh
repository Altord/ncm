#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
NEW="$BASE/codec-diverse-data-3k"
DATA="$BASE/codec-diverse-training-data"
HOLDOUT="$BASE/codec-fresh-godot-holdout/methods.jsonl"
OUT="$BASE/codec-diverse-data-3k-holdout"
for f in "$REF/config.yaml" "$REF/checkpoints/step-00003000.pt" "$NEW/config.yaml" "$NEW/checkpoints/step-00003000.pt" \
 "$NEW/cache/manifest.json" "$DATA/train.jsonl" "$DATA/train.manifest.json" "$DATA/methods.manifest.json" "$HOLDOUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/reference.jsonl" "$OUT/reference.report.json" "$OUT/candidate.jsonl" "$OUT/candidate.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python - "$REF/config.yaml" "$REF/checkpoints/step-00003000.pt" "$NEW/config.yaml" "$NEW/checkpoints/step-00003000.pt" "$NEW/cache/manifest.json" "$DATA" "$HOLDOUT" <<'PY'
import json, sys, torch
from pathlib import Path
from omegaconf import OmegaConf
from training.data import sha256_file
a, b = OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[3])
OmegaConf.resolve(a); OmegaConf.resolve(b)
for key in ('tokenizer','vqvae','model'):
    if OmegaConf.to_container(a[key], resolve=True) != OmegaConf.to_container(b[key], resolve=True):
        raise SystemExit(f'Reference and candidate {key} configurations differ')
for path in (sys.argv[2],sys.argv[4]):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if int(ckpt.get('step', -1)) != 2999: raise SystemExit(f'Expected step-3000 checkpoint: {path}')
    del ckpt
cache = json.loads(Path(sys.argv[5]).read_text()); data = Path(sys.argv[6]); holdout = sys.argv[7]
merged = json.loads((data/'train.manifest.json').read_text())
supplement = json.loads((data/'methods.manifest.json').read_text())
if (cache.get('source_sha256') != sha256_file(data/'train.jsonl') or merged['merged_sha256'] != cache['source_sha256'] or
    merged['supplemental_manifest_sha256'] != sha256_file(data/'methods.manifest.json') or
    supplement['source_sha256'].get(holdout) != sha256_file(holdout)):
    raise SystemExit('Training or protected holdout provenance differs')
print('Matched step-3000 architecture and training/holdout provenance checks passed.', flush=True)
PY
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for kind in reference candidate; do
 if [ "$kind" = reference ]; then MODEL="$REF"; else MODEL="$NEW"; fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00003000.pt" \
  --input "$HOLDOUT" --text-key target --code-only --batch-size 8 --output "$OUT/$kind.jsonl"
done
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/reference.jsonl" --candidate "$OUT/candidate.jsonl"
.venv/bin/python - "$HOLDOUT" "$OUT/reference.jsonl" "$OUT/candidate.jsonl" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path
def load(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(result) != len(rows): raise SystemExit(f'Empty or duplicate IDs: {path}')
    return result
source, reference, candidate = map(load, sys.argv[1:])
if source.keys() != reference.keys() or source.keys() != candidate.keys(): raise SystemExit('IDs differ')
groups = defaultdict(lambda: [0,0,0,0,0,0,0])
for key, row in source.items():
    a, b = reference[key], candidate[key]
    if a['target'] != row['target'] or b['target'] != row['target']: raise SystemExit(f'Target differs: {key}')
    old, new = a['token_exact_match'], b['token_exact_match']
    g = groups[row['source'].split('/')[0]]; g[0] += 1; g[1] += old; g[2] += new; g[3] += not old and new; g[4] += old and not new
    g[5] += a['reconstructed'] == row['target']; g[6] += b['reconstructed'] == row['target']
print('\n=== Fresh holdout: matched 3,000-step project breakdown ===')
for project, (n, old, new, gained, lost, old_text, new_text) in sorted(groups.items()):
    print(f'{project}: token-exact {old}/{n} -> {new}/{n}; gained={gained}; lost={lost}; text-exact {old_text}/{n} -> {new_text}/{n}')
print('One preselected broader-data candidate versus the original reference; known-target codec only.')
print('Both checkpoints are at step 3,000; larger training corpus changes sample order and effective epochs.')
print('Overlap filtering is lexical and does not rule out semantic or pretraining overlap; no compile, behavior or generation tests.')
PY
