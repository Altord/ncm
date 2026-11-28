#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
NEW="$BASE/codec-stochastic-decoder-finetune"
SAVED="$BASE/codec-diverse-data-5k-stress"
HOLD="$BASE/codec-diverse-5k-fresh-holdout"
INPUT=examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl
OUT="$BASE/codec-stochastic-decoder-followup"
for f in "$MODEL/config.yaml" "$NEW/decoder.pt" "$SAVED/candidate.jsonl" "$SAVED/candidate-compiled.jsonl" "$HOLD/methods.jsonl" "$HOLD/new.jsonl" "$INPUT"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/holdout.jsonl" "$OUT/stress.jsonl" "$OUT/stress-compiled.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if ! command -v dotnet >/dev/null && [ -x "$HOME/.dotnet/dotnet" ]; then export DOTNET_ROOT="$HOME/.dotnet"; export PATH="$DOTNET_ROOT:$PATH"; fi
command -v dotnet >/dev/null || { echo 'dotnet SDK not found.' >&2; exit 1; }
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in holdout stress; do
 if [ "$name" = holdout ]; then SOURCE="$HOLD/methods.jsonl"; REF="$HOLD/new.jsonl"; CODE=(--code-only); else SOURCE="$INPUT"; REF="$SAVED/candidate.jsonl"; CODE=(); fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$NEW/decoder.pt" --input "$SOURCE" --text-key target "${CODE[@]}" --batch-size 8 --output "$OUT/$name.jsonl"
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$REF" --candidate "$OUT/$name.jsonl"
done
# Reuse the original compiled stress baseline only when text matches; other candidates are compiled afresh.
.venv/bin/python -m evaluation.compile_saved_codec_candidate --input "$INPUT" --candidate "$OUT/stress.jsonl" --baseline "$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k/stress.jsonl" --output "$OUT/stress-compiled.jsonl"
