#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-diverse-data-5k"
OUT="$BASE/codec-original-diverse-ensemble"
for f in "$A/config.yaml" "$A/checkpoints/step-00005000.pt" "$B/config.yaml" "$B/checkpoints/step-00005000.pt" \
 "$B/methods-reference.jsonl" "$B/methods-candidate.jsonl" "$B/external-reference.jsonl" "$B/external-candidate.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for name in methods external; do
 for f in "$OUT/$name.jsonl" "$OUT/$name.report.json"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
 done
done
.venv/bin/python evaluation/test_ensemble_codec_methods.py
.venv/bin/python evaluation/test_ensemble_codec_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"; else INPUT="$BASE/codec-external-methods/methods.jsonl"; fi
 .venv/bin/python -m evaluation.ensemble_codec_methods \
  --config-a "$A/config.yaml" --checkpoint-a "$A/checkpoints/step-00005000.pt" --saved-a "$B/$name-reference.jsonl" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/checkpoints/step-00005000.pt" --saved-b "$B/$name-candidate.jsonl" \
  --input "$INPUT" --output "$OUT/$name.jsonl" --batch-size 8 --device cuda
done
