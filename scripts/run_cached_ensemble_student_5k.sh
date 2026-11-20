#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
A="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
B="$BASE/codec-width256-encoder8-rare2-5k"
HELDOUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
OUT="$BASE/codec-cached-ensemble-student-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$A/config.yaml" "$A/artifact/tokenizer.pt" "$A/validation.report.json" "$B/config.yaml" "$B/artifact/tokenizer.pt" "$HELDOUT" "$BASE/codec-method-boundary-rare-repeat/reference.jsonl"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/cache" "$OUT/checkpoints/latest.pt" "$OUT/artifact/tokenizer.pt" "$OUT/methods.jsonl"; do
  [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_ensemble_teacher_cache.py
.venv/bin/python evaluation/test_ensemble_teacher_cache_integration.py
.venv/bin/python evaluation/test_cached_ensemble_student_integration.py
.venv/bin/python evaluation/test_codec_distillation.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
echo '===== Cache ensemble predictions on training data only ====='
.venv/bin/python -m training.prepare_ensemble_teacher_cache \
  --config-a "$A/config.yaml" --checkpoint-a "$A/artifact/tokenizer.pt" \
  --config-b "$B/config.yaml" --checkpoint-b "$B/artifact/tokenizer.pt" \
  --output "$OUT/cache" --topk 4 --batch-size 8 --device cuda
.venv/bin/python - "$A" "$B" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
a, b, out = sys.argv[1:]
cfg = OmegaConf.load(a + '/config.yaml')
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    int(cfg.vqvae.dec_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 64):
    raise SystemExit('Unexpected student reference configuration')
if cfg.train.get('error_teacher', {}).get('enabled', False) or float(cfg.train.get('distillation', {}).get('weight', 0.0)):
    raise SystemExit('Unexpected teacher settings in reference')
cfg.data.cache_dir = out + '/cache'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
cfg.task = 'codec-cached-ensemble-student-5k'
cfg.train.distillation = {'weight': 0.1, 'topk': 4, 'cached_ensemble': True,
    'teachers': [{'config': a + '/config.yaml', 'checkpoint': a + '/artifact/tokenizer.pt'},
                 {'config': b + '/config.yaml', 'checkpoint': b + '/artifact/tokenizer.pt'}]}
OmegaConf.save(cfg, out + '/config.yaml')
print('Fresh width-192 student, 5,000 steps. Cached top-4 ensemble supervision applies only when the teacher is correct.', flush=True)
print('Teacher predictions are computed once; only the student runs during training and inference.', flush=True)
PY
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
VALID=data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl
[ -f "$VALID" ] || { echo "Missing: $VALID" >&2; exit 1; }
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$VALID" --text-key text --code-only --batch-size 16 --output "$OUT/validation.jsonl"
.venv/bin/python -m evaluation.compare_codec_aggregate --reference-report "$A/validation.report.json" \
  --candidate-report "$OUT/validation.report.json" --reference-config "$A/config.yaml" --candidate-config "$OUT/config.yaml"
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$HELDOUT" --text-key target --code-only --batch-size 16 --output "$OUT/methods.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: r = json.load(f)
print('\n=== Single-codec ensemble student results to share ===')
print(f"Methods: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; similarity={r['mean_character_similarity']:.2%}")
print('References: width-192 single codec 218/348 exact; two-codec ensemble 261/348 exact.')
print('Known-target codec reconstruction only; no compiler or generator evaluation.')
PY
