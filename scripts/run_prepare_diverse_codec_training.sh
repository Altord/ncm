#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
METHODS=data/mycode/godot-csharp-method-pairs-ctx256-prompt64
STRESS=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
PRIOR="$BASE/codec-external-methods/methods.jsonl"
HOLDOUT="$BASE/codec-fresh-godot-holdout/methods.jsonl"
ROOT=data/external-codec-training-2026-11-26
OUT="$BASE/codec-diverse-training-data"
for f in "$A/config.yaml" "$B/config.yaml" "$METHODS/train.jsonl" "$METHODS/validation.jsonl" "$STRESS" "$PRIOR" "$HOLDOUT"; do
 [ -f "$f" ] || { echo "Missing protected or configuration file: $f" >&2; exit 1; }
done
for f in "$ROOT" "$OUT/methods.jsonl" "$OUT/methods.manifest.json" "$OUT/train.jsonl" "$OUT/train.manifest.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_build_fresh_codec_methods.py
.venv/bin/python evaluation/test_build_fresh_codec_methods_integration.py
.venv/bin/python evaluation/test_merge_codec_training.py
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
fetch_repo sakuga https://github.com/NoisyChain/Sakuga-Engine.git 866ce24da7c212c89232894cf6144981df2d4dbe
fetch_repo cycling https://github.com/Bills-Institute-of-Technology/GODOT4_CycleTour.git 40f48f5318658bef9623735026e715c6e2e2362a
fetch_repo book https://github.com/PacktPublishing/Game-Development-with-Godot-4-and-C-Sharp.git d38fa94132ae5ea5a71135161d1c5cae837c2c5b
fetch_repo raccoons https://github.com/Raccoons-Rise-Up/RaccoonsRiseUp.git 65e987b16b671454d9b4a88bf94447d89aabf39d
.venv/bin/python -m evaluation.build_fresh_codec_methods --root "$ROOT" --reference-config "$A/config.yaml" \
 --comparison-config "$B/config.yaml" --method-train "$METHODS/train.jsonl" --method-validation "$METHODS/validation.jsonl" \
 --stress "$STRESS" --extra-protected "$PRIOR" --extra-protected "$HOLDOUT" --max-per-repository 400 --max-per-file 8 --output "$OUT/methods.jsonl"
.venv/bin/python - "$A/config.yaml" "$OUT" <<'PY'
import subprocess, sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1]); OmegaConf.resolve(cfg)
out = sys.argv[2]
subprocess.run([sys.executable, '-m', 'training.merge_codec_training', '--original', str(cfg.data.input),
                '--supplemental', out+'/methods.jsonl', '--manifest', out+'/methods.manifest.json',
                '--text-key', str(cfg.data.get('text_key', 'text')), '--min-supplemental', '200',
                '--output', out+'/train.jsonl'], check=True)
PY
echo 'Prepared training data only; no model training performed. The fresh holdout remains excluded from training.'
