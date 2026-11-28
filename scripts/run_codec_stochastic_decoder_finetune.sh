#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
OUT="$BASE/codec-stochastic-decoder-finetune"
for f in "$MODEL/config.yaml" "$MODEL/checkpoints/step-00005000.pt" "$MODEL/methods-candidate.jsonl" "$MODEL/external-candidate.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/decoder.pt" "$OUT/methods.jsonl" "$OUT/external.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|finetune_stochastic_codec_decoder)' >/dev/null; then
 echo 'A trainer is already running.' >&2; exit 1
fi
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.finetune_stochastic_codec_decoder --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00005000.pt" \
 --output "$OUT/decoder.pt" --steps 1000 --validate-every 250 --batch-size 8 --lr 0.00001 --device cuda
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"; else INPUT="$BASE/codec-external-methods/methods.jsonl"; fi
 .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$OUT/decoder.pt" --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name.jsonl"
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$MODEL/$name-candidate.jsonl" --candidate "$OUT/$name.jsonl"
done
echo 'Decoder-only fine-tuning screen; reused development data. No compile, behavior or generation test.'
