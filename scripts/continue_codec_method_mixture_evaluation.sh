#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
OUT="$BASE/codec-method-mixture-5k"
HELDOUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
for f in "$REF/config.yaml" "$REF/validation.report.json" "$OUT/config.yaml" "$OUT/artifact/tokenizer.pt" "$OUT/validation.report.json" "$HELDOUT" "$BASE/codec-method-boundary-rare-repeat/reference.jsonl"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
.venv/bin/python evaluation/test_compare_codec_aggregate.py
.venv/bin/python -m evaluation.compare_codec_aggregate \
  --reference-report "$REF/validation.report.json" --candidate-report "$OUT/validation.report.json" \
  --reference-config "$REF/config.yaml" --candidate-config "$OUT/config.yaml"
if [ -f "$OUT/methods.report.json" ] && [ -f "$OUT/methods.jsonl" ]; then
  echo 'Using saved method evaluation.'
elif [ -e "$OUT/methods.jsonl" ] || [ -e "$OUT/methods.report.json" ]; then
  echo 'Incomplete method output exists; stopping to preserve it.' >&2; exit 1
else
  echo 'Running missing held-out method reconstruction evaluation.'
  TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 .venv/bin/python evaluation/godot_csharp_codec_eval.py \
    --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
    --input "$HELDOUT" --text-key target --code-only --batch-size 16 --output "$OUT/methods.jsonl"
fi
.venv/bin/python evaluation/compare_codec_pairs.py \
  --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: r = json.load(f)
print('\n=== Method-mixture codec screen results to share ===')
print(f"Methods: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; similarity={r['mean_character_similarity']:.2%}")
print('Rare-only reference: 218/348 exact methods. See paired comparison above.')
print('Known-target codec reconstruction only; no compilation or generation evaluation.')
PY
