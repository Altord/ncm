#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
PRIOR="$BASE/codec-cached-ensemble-student-5k"
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
HELDOUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
VALID=data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl
OUT="$BASE/codec-targeted-ensemble-student-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$PRIOR/config.yaml" "$PRIOR/cache/manifest.json" "$PRIOR/methods.jsonl" "$PRIOR/methods.report.json" "$REF/config.yaml" "$REF/validation.report.json" "$VALID" "$HELDOUT"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/checkpoints/latest.pt" "$OUT/artifact/tokenizer.pt" "$OUT/methods.jsonl"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_distillation.py
.venv/bin/python evaluation/test_cached_ensemble_student_integration.py
.venv/bin/python evaluation/test_compare_codec_aggregate.py
mkdir -p "$OUT"
.venv/bin/python - "$PRIOR" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
prior, out = sys.argv[1:]
cfg = OmegaConf.load(prior + '/config.yaml')
d = cfg.train.get('distillation')
if not d or not d.get('cached_ensemble', False) or d.get('student_errors_only', False):
    raise SystemExit('Prior config is not the expected cached ensemble student')
if int(cfg.vqvae.embed_dim) != 192 or len(cfg.vqvae.vqvae_vocab_size) != 64:
    raise SystemExit('Unexpected codec architecture')
d.student_errors_only = True
cfg.task = 'codec-targeted-ensemble-student-5k'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
OmegaConf.save(cfg, out + '/config.yaml')
print('Fresh student; cached teacher reused. Distillation is restricted to tokens the student misses and teacher gets right.', flush=True)
print('Loss is normalized over eligible tokens, so its effective emphasis changes even though its configured weight is unchanged.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$VALID" --text-key text --code-only --batch-size 16 --output "$OUT/validation.jsonl"
.venv/bin/python -m evaluation.compare_codec_aggregate --reference-report "$REF/validation.report.json" \
  --candidate-report "$OUT/validation.report.json" --reference-config "$REF/config.yaml" --candidate-config "$OUT/config.yaml"
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$HELDOUT" --text-key target --code-only --batch-size 16 --output "$OUT/methods.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$PRIOR/methods.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: r = json.load(f)
print('\n=== Targeted ensemble student results to share ===')
print(f"Methods: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; similarity={r['mean_character_similarity']:.2%}")
print('Prior student: 209/348 exact; rare-only reference: 218/348; two-codec ensemble: 261/348.')
print('Known-target codec reconstruction only; no compile, behavior, or generator test.')
PY
