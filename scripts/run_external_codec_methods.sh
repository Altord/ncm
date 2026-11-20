#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
ROOT=data/external-codec-eval-2026-11
OUT="$BASE/codec-external-methods"
METHODS=data/mycode/godot-csharp-method-pairs-ctx256-prompt64
STRESS=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$METHODS/train.jsonl" "$METHODS/validation.jsonl" "$STRESS"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
if [ -e "$OUT/methods.jsonl" ] || [ -e "$OUT/methods.manifest.json" ]; then echo 'Existing dataset; stopping to preserve it.' >&2; exit 1; fi
.venv/bin/python evaluation/test_build_fresh_codec_methods.py
.venv/bin/python evaluation/test_build_fresh_codec_methods_integration.py
.venv/bin/python evaluation/test_ensemble_codec_methods.py
.venv/bin/python evaluation/test_ensemble_codec_integration.py
mkdir -p "$ROOT" "$OUT"
fetch_repo() {
  local name="$1" url="$2" commit="$3" path="$ROOT/$1"
  if [ -e "$path" ]; then
    echo "Existing repository path: $path; stopping to preserve it." >&2; return 1
  fi
  mkdir -p "$path"
  git -C "$path" init -q
  git -C "$path" remote add origin "$url"
  git -C "$path" -c protocol.version=2 fetch --depth 1 --filter=blob:none origin "$commit"
  git -C "$path" sparse-checkout init --no-cone
  git -C "$path" sparse-checkout set --no-cone '/*.cs' '**/*.cs' '/LICENSE*' '/README*'
  git -C "$path" checkout --detach FETCH_HEAD
  if [ "$(git -C "$path" rev-parse HEAD)" != "$commit" ]; then echo "Unexpected revision in $path" >&2; return 1; fi
}
echo '===== Fetch pinned public source snapshots ====='
fetch_repo snake https://github.com/binamonk/Godot-SnakeCS.git 7d13b69dfd6bbbb4d0936227e3bdaee849eb4939
fetch_repo flock https://github.com/SadColourfulHues/Godot_FlockTest.git 37c4d43688fb5499c168c7d279027e3faf0623f7
echo '===== Build and screen external methods ====='
.venv/bin/python -m evaluation.build_fresh_codec_methods --root "$ROOT" --reference-config "$A/config.yaml" \
  --comparison-config "$B/config.yaml" --method-train "$METHODS/train.jsonl" --method-validation "$METHODS/validation.jsonl" \
  --stress "$STRESS" --output "$OUT/methods.jsonl"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in a b; do
  if [ "$name" = a ]; then MODEL="$A"; else MODEL="$B"; fi
  .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
    --input "$OUT/methods.jsonl" --text-key target --code-only --batch-size 8 --output "$OUT/$name.jsonl"
done
.venv/bin/python -m evaluation.ensemble_codec_methods \
  --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" --saved-a "$OUT/a.jsonl" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$OUT/b.jsonl" \
  --input "$OUT/methods.jsonl" --output "$OUT/ensemble.jsonl" --batch-size 8 --device cuda
echo '===== Pinned dataset provenance ====='
cat "$OUT/methods.manifest.json"
