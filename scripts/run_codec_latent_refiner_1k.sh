#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXTERNAL="$BASE/codec-external-methods"
OUT="$BASE/codec-latent-refiner-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
 echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$METHODS/reference.report.json" "$EXTERNAL/methods.jsonl" "$EXTERNAL/a.jsonl" "$EXTERNAL/a.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/adapter.pt" "$OUT/methods.jsonl" "$OUT/methods.report.json" "$OUT/external.jsonl" "$OUT/external.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_latent_refiner.py
.venv/bin/python evaluation/test_codec_latent_refiner_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode train --config "$MODEL/config.yaml" \
 --checkpoint "$MODEL/artifact/tokenizer.pt" --output "$OUT/adapter.pt" --steps 1000 --validate-every 500 --batch-size 8 --device cuda
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; BASELINE="$METHODS/reference.jsonl"; else INPUT="$EXTERNAL/methods.jsonl"; BASELINE="$EXTERNAL/a.jsonl"; fi
 .venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$MODEL/config.yaml" \
  --checkpoint "$MODEL/artifact/tokenizer.pt" --adapter "$OUT/adapter.pt" --input "$INPUT" --baseline "$BASELINE" \
  --batch-size 8 --device cuda --output "$OUT/$name.jsonl"
done
