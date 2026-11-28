#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
C="$BASE/codec-diverse-data-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXT="$BASE/codec-external-methods"
OUT="$BASE/codec-diverse-three-ensemble"
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$C/config.yaml" "$C/checkpoints/step-00005000.pt" \
 "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$B/methods.jsonl" "$C/methods-candidate.jsonl" \
 "$EXT/methods.jsonl" "$EXT/a.jsonl" "$EXT/b.jsonl" "$C/external-candidate.jsonl"; do
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
 if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; SA="$METHODS/reference.jsonl"; SB="$B/methods.jsonl"; else INPUT="$EXT/methods.jsonl"; SA="$EXT/a.jsonl"; SB="$EXT/b.jsonl"; fi
 .venv/bin/python -m evaluation.ensemble_codec_methods \
  --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" --saved-a "$SA" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" --saved-b "$SB" \
  --config-c "$C/config.yaml" --checkpoint-c "$C/checkpoints/step-00005000.pt" --saved-c "$C/$name-candidate.jsonl" \
  --input "$INPUT" --output "$OUT/$name.jsonl" --batch-size 4 --device cuda
done
.venv/bin/python -m evaluation.compare_saved_ensemble_exact --input "$METHODS/heldout-methods.jsonl" --reference "$BASE/codec-two-model-ensemble-methods/results.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python -m evaluation.compare_saved_ensemble_exact --input "$EXT/methods.jsonl" --reference "$EXT/ensemble.jsonl" --candidate "$OUT/external.jsonl"
