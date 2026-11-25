#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXT="$BASE/codec-external-methods"
OUT="$BASE/codec-continuous-teacher-validnorm-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$EXT/methods.jsonl" "$EXT/a.jsonl"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
if [ -e "$OUT/config.yaml" ] || [ -e "$OUT/cache" ] || [ -e "$OUT/checkpoints" ] || [ -e "$OUT/artifact" ]; then
  echo "Existing experiment output in $OUT; stopping to preserve it." >&2; exit 1
fi
.venv/bin/python evaluation/test_continuous_codec_teacher.py
.venv/bin/python evaluation/test_comment_codec_weights.py
.venv/bin/python evaluation/test_method_name_weighting.py
.venv/bin/python evaluation/test_repeated_identifier_weights.py
.venv/bin/python - "$REF/config.yaml" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1]); OmegaConf.resolve(cfg)
cache = Path(str(cfg.data.cache_dir))
if not (cache / 'manifest.json').is_file():
    raise SystemExit(f'Reference training cache missing: {cache}; prepare it with the reference configuration first.')
print(f'Using reference training cache: {cache}', flush=True)
PY
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
weights = cfg.train.get('reconstruction_token_weights')
if not weights or str(weights.get('mode', '')) != 'per_occurrence' or weights.get('literal') is not None:
    raise SystemExit('Expected reference per-occurrence weighting without a literal override')
if cfg.train.get('error_teacher', {}).get('enabled', False) or float(cfg.train.get('distillation', {}).get('weight', 0.0)):
    raise SystemExit('Unexpected teacher settings in reference')
cfg.task = 'codec-continuous-teacher-validnorm-1k'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
if cfg.vqvae.vector_quantizer_config.get('final_level_loss_weight', 1.0) != 1.0:
    raise SystemExit('Reference already changes final-level VQ weight')
if cfg.train.get('continuous_distillation', {}).get('weight', 0.0):
    raise SystemExit('Reference already has a continuous teacher')
cfg.train.continuous_distillation = {'weight': 0.1, 'topk': 4, 'normalization': 'valid'}
OmegaConf.save(cfg, out + '/config.yaml')
print('Online continuous teacher: weight 0.1, top-4, normalized over valid tokens.', flush=True)
print('Fresh 64-level codec, 1,000 updates; validation is unweighted.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 1000
.venv/bin/python - "$REF/logs/validation.jsonl" "$BASE/codec-continuous-teacher-5k/logs/validation.jsonl" "$OUT/logs/validation.jsonl" "$OUT/logs/train.jsonl" <<'PY'
import json, statistics, sys
from pathlib import Path
for label, name in zip(('reference', 'previous teacher', 'valid-normalized teacher'), sys.argv[1:4]):
    path = Path(name)
    if not path.is_file():
        print(f'{label}: missing validation log {path}'); continue
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    match = [r for r in rows if r.get('step') == 1000]
    if not match:
        print(f'{label}: no step-1000 validation entry'); continue
    r = match[-1]
    print(f"{label}: step 1000 accuracy={r['val_token_accuracy']:.2%}; exact={r['val_exact_sequence_match_rate']:.2%}; CE={r['val_recon_ce']:.4f}")
path = Path(sys.argv[4])
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
for key in ('continuous_eligible_fraction', 'continuous_distill_ce', 'continuous_applied_loss', 'examples_per_second'):
    values = [r[key] for r in rows[-5:] if key in r]
    if values: print(f'{key}: recent median={statistics.median(values):.6f}')
print('Early comparison only; no compile, generation, or claim about 5k performance.')
PY
