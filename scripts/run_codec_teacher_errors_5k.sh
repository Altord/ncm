#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
OUT="$BASE/codec-teacher-error-weight-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for file in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$METHODS" "$BASE/codec-method-boundary-rare-repeat/reference.jsonl"; do
  [ -f "$file" ] || { echo "Missing: $file" >&2; exit 1; }
done
for file in "$OUT/config.yaml" "$OUT/checkpoints/latest.pt" "$OUT/artifact/tokenizer.pt" "$OUT/methods.jsonl"; do
  [ ! -e "$file" ] || { echo "Existing output: $file; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_teacher_errors.py
mkdir -p "$OUT"
.venv/bin/python - "$REF" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
ref, out = sys.argv[1:]
cfg = OmegaConf.load(ref + '/config.yaml')
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    int(cfg.vqvae.dec_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 64 or
    float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0):
    raise SystemExit('Unexpected reference codec configuration')
if float(cfg.train.get('distillation', {}).get('weight', 0.0)):
    raise SystemExit('Reference already has distillation enabled')
cfg.task = 'codec-teacher-error-weight-5k'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
cfg.train.error_teacher = {'enabled': True, 'multiplier': 2.0,
                           'config': ref + '/config.yaml', 'checkpoint': ref + '/artifact/tokenizer.pt'}
OmegaConf.save(cfg, out + '/config.yaml')
print('Fresh 64-level codec, 5,000 updates; frozen teacher flags missed training tokens for 2x weighting.', flush=True)
print('Training may be slower and use more GPU memory; validation stays unweighted.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$METHODS" --text-key target --code-only --batch-size 16 --output "$OUT/methods.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py \
  --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python - "$OUT/methods.report.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: report = json.load(f)
print('\n=== Teacher-error codec screen results to share ===')
print(f"Methods: accuracy={report['valid_token_accuracy']:.2%}; exact={report['exact_sequence_matches']}/{report['examples']}; similarity={report['mean_character_similarity']:.2%}")
print('Reference rare-only codec: 218/348 exact methods. See paired comparison above.')
print('Frozen-teacher training uses original targets only from training data; this is a known-target codec evaluation, not generation or a compile test.')
PY
