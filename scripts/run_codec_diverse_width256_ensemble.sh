#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/codec-diverse-data-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
OUT="$BASE/codec-diverse-width256-ensemble"
for f in "$A/config.yaml" "$A/checkpoints/step-00005000.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" \
 "$A/methods-candidate.jsonl" "$A/external-candidate.jsonl" "$B/methods.jsonl" "$BASE/codec-external-methods/b.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 for f in "$OUT/$name.jsonl" "$OUT/$name.report.json"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
 done
done
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"; SAVED="$B/methods.jsonl"; else INPUT="$BASE/codec-external-methods/methods.jsonl"; SAVED="$BASE/codec-external-methods/b.jsonl"; fi
 .venv/bin/python -m evaluation.ensemble_codec_methods \
  --config-a "$A/config.yaml" --checkpoint-a "$A/checkpoints/step-00005000.pt" --saved-a "$A/$name-candidate.jsonl" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$SAVED" \
  --input "$INPUT" --output "$OUT/$name.jsonl" --batch-size 8 --device cuda
done
