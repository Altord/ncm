#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
NEW="$BASE/codec-diverse-data-5k"
DATA="$BASE/codec-diverse-training-data"
METHODS=data/mycode/godot-csharp-method-pairs-ctx256-prompt64
STRESS=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
PRIOR="$BASE/codec-external-methods/methods.jsonl"
HOLDOUT="$BASE/codec-fresh-godot-holdout/methods.jsonl"
ROOT=data/external-codec-holdout-b-2026-11-26
OUT="$BASE/codec-diverse-5k-fresh-holdout"
for f in "$A/config.yaml" "$A/checkpoints/step-00005000.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" \
 "$NEW/config.yaml" "$NEW/checkpoints/step-00005000.pt" "$DATA/methods.jsonl" "$DATA/methods.manifest.json" \
 "$METHODS/train.jsonl" "$METHODS/validation.jsonl" "$STRESS" "$PRIOR" "$HOLDOUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$ROOT" "$OUT/methods.jsonl" "$OUT/methods.manifest.json" "$OUT/a.jsonl" "$OUT/b.jsonl" "$OUT/new.jsonl" "$OUT/old-ensemble.jsonl" "$OUT/new-ensemble.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_build_fresh_codec_methods_integration.py
.venv/bin/python evaluation/test_ensemble_codec_integration.py
mkdir -p "$ROOT" "$OUT"
fetch_repo() {
 local name="$1" url="$2" commit="$3" path="$ROOT/$1"
 mkdir -p "$path"
 git -C "$path" init -q
 git -C "$path" remote add origin "$url"
 git -C "$path" -c protocol.version=2 fetch --depth 1 --filter=blob:none origin "$commit"
 git -C "$path" sparse-checkout init --no-cone
 git -C "$path" sparse-checkout set --no-cone '/*.cs' '**/*.cs' '/LICENSE*' '/README*'
 git -C "$path" checkout --detach FETCH_HEAD
 [ "$(git -C "$path" rev-parse HEAD)" = "$commit" ] || { echo "Unexpected revision: $path" >&2; exit 1; }
}
fetch_repo alpheratz https://github.com/Ark2000/world_of_alpheratz.git 4b10365f32eef30c693976e9db4ef9691fdc2b56
fetch_repo project2d https://github.com/Valks-Games/Project2D.git ecad6939c3aea5f2d8e6a906e0d4b49f0c0d863b
fetch_repo squiggles https://github.com/QueenOfSquiggles/Squiggles-Core-4X.git 77d168472fd64039bb2c16e0f1872a0cbd2ce932
.venv/bin/python -m evaluation.build_fresh_codec_methods --root "$ROOT" --reference-config "$A/config.yaml" --comparison-config "$B/config.yaml" \
 --method-train "$METHODS/train.jsonl" --method-validation "$METHODS/validation.jsonl" --stress "$STRESS" \
 --extra-protected "$PRIOR" --extra-protected "$HOLDOUT" --extra-protected "$DATA/methods.jsonl" \
 --max-per-repository 40 --max-per-file 3 --output "$OUT/methods.jsonl"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in a b new; do
 if [ "$name" = a ]; then MODEL="$A"; CKPT="$A/checkpoints/step-00005000.pt"; elif [ "$name" = b ]; then MODEL="$B"; CKPT="$B/artifact/tokenizer.pt"; else MODEL="$NEW"; CKPT="$NEW/checkpoints/step-00005000.pt"; fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$CKPT" \
  --input "$OUT/methods.jsonl" --text-key target --code-only --batch-size 8 --output "$OUT/$name.jsonl"
done
for kind in old new; do
 if [ "$kind" = old ]; then MODEL="$A"; CKPT="$A/checkpoints/step-00005000.pt"; SAVED="$OUT/a.jsonl"; else MODEL="$NEW"; CKPT="$NEW/checkpoints/step-00005000.pt"; SAVED="$OUT/new.jsonl"; fi
 .venv/bin/python -m evaluation.ensemble_codec_methods --config-a "$MODEL/config.yaml" --checkpoint-a "$CKPT" --saved-a "$SAVED" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$OUT/b.jsonl" \
  --input "$OUT/methods.jsonl" --output "$OUT/$kind-ensemble.jsonl" --batch-size 8 --device cuda
done
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/a.jsonl" --candidate "$OUT/new.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
def load(name):
    rows = [json.loads(s) for s in (root/name).read_text().splitlines() if s.strip()]
    result = {r['id']:r for r in rows}
    if not rows or len(result) != len(rows): raise SystemExit(f'Empty or duplicate IDs: {name}')
    return result
source,a,new,old_ens,new_ens = (load(x) for x in ('methods.jsonl','a.jsonl','new.jsonl','old-ensemble.jsonl','new-ensemble.jsonl'))
if any(source.keys() != rows.keys() for rows in (a,new,old_ens,new_ens)): raise SystemExit('IDs differ')
groups = defaultdict(lambda:[0,0,0,0,0])
for k,r in source.items():
    if a[k]['target'] != r['target'] or new[k]['target'] != r['target']: raise SystemExit(f'Target differs: {k}')
    g = groups[r['source'].split('/')[0]]; g[0] += 1
    g[1] += a[k]['reconstructed']==r['target']; g[2] += new[k]['reconstructed']==r['target']
    g[3] += old_ens[k]['ensemble_exact']; g[4] += new_ens[k]['ensemble_exact']
print('\n=== Second fresh project holdout ===')
for name,(n,a1,b1,c,d) in sorted(groups.items()):
    print(f'{name}: examples={n}; old single={a1}; broader-data single={b1}; old ensemble={c}; broader-data ensemble={d}')
print('Fixed 5k candidates, known-target codec only. Prior datasets and supplemental training methods were screened lexically.')
print('Semantic duplication and pretraining exposure remain possible. No compile, behavior, generation or end-to-end speed test.')
PY
