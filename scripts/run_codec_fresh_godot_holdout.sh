#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
ADAPTER="$BASE/codec-latent-refiner-1k/adapter.pt"
METHODS=data/mycode/godot-csharp-method-pairs-ctx256-prompt64
STRESS=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
PREVIOUS="$BASE/codec-external-methods/methods.jsonl"
ROOT=data/external-codec-holdout-2026-11-26
OUT="$BASE/codec-fresh-godot-holdout"
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$ADAPTER" "$METHODS/train.jsonl" "$METHODS/validation.jsonl" "$STRESS" "$PREVIOUS"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$ROOT" "$OUT/methods.jsonl" "$OUT/methods.manifest.json" "$OUT/a.jsonl" "$OUT/b.jsonl" "$OUT/adapter.jsonl" "$OUT/ensemble.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_build_fresh_codec_methods.py
.venv/bin/python evaluation/test_build_fresh_codec_methods_integration.py
.venv/bin/python evaluation/test_codec_latent_refiner_integration.py
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
fetch_repo sample https://github.com/godotmancer/SampleGodotCSharpProject.git 84452a0bb15818e88e5f67a740bf00f05e61b6f5
fetch_repo template https://github.com/chickensoft-games/GodotGame.git 93392ee8919568812e80127aef8ebf37163f2818
fetch_repo roguelike https://github.com/nick-shimpo/null-and-void.git 7838410cdc84d99fa76f73f3613b8e57edba9fca
.venv/bin/python -m evaluation.build_fresh_codec_methods --root "$ROOT" --reference-config "$A/config.yaml" \
 --comparison-config "$B/config.yaml" --method-train "$METHODS/train.jsonl" --method-validation "$METHODS/validation.jsonl" \
 --stress "$STRESS" --extra-protected "$PREVIOUS" --max-per-repository 40 --max-per-file 3 --output "$OUT/methods.jsonl"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in a b; do
 if [ "$name" = a ]; then MODEL="$A"; else MODEL="$B"; fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
  --input "$OUT/methods.jsonl" --text-key target --code-only --batch-size 8 --output "$OUT/$name.jsonl"
done
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$A/config.yaml" --checkpoint "$A/artifact/tokenizer.pt" \
 --adapter "$ADAPTER" --input "$OUT/methods.jsonl" --baseline "$OUT/a.jsonl" --batch-size 8 --device cuda --output "$OUT/adapter.jsonl"
.venv/bin/python -m evaluation.ensemble_codec_methods --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" --saved-a "$OUT/a.jsonl" \
 --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$OUT/b.jsonl" \
 --input "$OUT/methods.jsonl" --output "$OUT/ensemble.jsonl" --batch-size 8 --device cuda
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1])
def load(name):
    rows = [json.loads(s) for s in (root/name).read_text().splitlines() if s.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(rows) != len(result): raise SystemExit(f'Empty or duplicate IDs: {name}')
    return result
source, a, b, adapter, ensemble = (load(name) for name in ('methods.jsonl','a.jsonl','b.jsonl','adapter.jsonl','ensemble.jsonl'))
if any(source.keys() != rows.keys() for rows in (a,b,adapter,ensemble)): raise SystemExit('IDs differ')
groups = defaultdict(lambda: [0,0,0,0,0])
for key, row in source.items():
    if a[key]['target'] != row['target'] or b[key]['target'] != row['target']: raise SystemExit(f'Target differs: {key}')
    g = groups[row['source'].split('/')[0]]; g[0] += 1
    g[1] += a[key]['reconstructed'] == row['target']; g[2] += b[key]['reconstructed'] == row['target']
    g[3] += adapter[key]['adapter_exact']; g[4] += ensemble[key]['ensemble_exact']
print('\n=== Fresh project holdout ===')
for name, (n, x, y, z, e) in sorted(groups.items()):
    print(f'{name}: examples={n}; width192={x}; width256={y}; 192+adapter={z}; two-codec ensemble={e}')
print('Exact text; known-target codec only, no signatures or source-derived sidecars.')
print('Public repositories were lexically screened against prior data, but renamed/semantic duplicates and pretraining exposure remain possible.')
print('No compilation, behavior, generation or end-to-end timing. Do not tune against this holdout.')
PY
