#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
REP="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-repeat15-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
OUT="$BASE/bf16-aot-64level-encoder8-width192-two-teacher-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for file in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$REP/config.yaml" "$REP/artifact/tokenizer.pt" "$METHODS"; do
  if [ ! -f "$file" ]; then echo "Missing required file: $file" >&2; exit 1; fi
done
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/tokenizer.pt" ]; then
  echo "Existing results in $OUT; stopping to preserve them." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_codec_distillation.py
.venv/bin/python - "$REF" "$REP" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
ref, rep, out = sys.argv[1:]
cfg = OmegaConf.load(ref + '/config.yaml')
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    int(cfg.vqvae.dec_layers) != 4 or len(cfg.vqvae.vqvae_vocab_size) != 64 or
    float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0):
    raise SystemExit('Unexpected reference codec configuration; stopping')
cfg.task = 'codec-two-teacher-5k'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
cfg.train.distillation = {'weight': 0.1, 'topk': 4, 'teachers': [
    {'config': ref + '/config.yaml', 'checkpoint': ref + '/artifact/tokenizer.pt'},
    {'config': rep + '/config.yaml', 'checkpoint': rep + '/artifact/tokenizer.pt'},
]}
OmegaConf.save(cfg, out + '/config.yaml')
print('Training a new student; both frozen teachers are used only during training.', flush=True)
print('5,000 updates, validation/checkpoint every 1,000; expect slower steps and higher VRAM.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
echo '===== Held-out-file full method reconstruction ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input "$METHODS" --text-key target --code-only --batch-size 16 \
  --output "$OUT/methods.jsonl"
echo '===== Compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --batch-size 16 --compile --progress --output "$OUT/stress.jsonl"
echo '===== Paired method comparison ====='
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
echo '===== Paired stress comparison ====='
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$REF/stress.jsonl" --candidate "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print('\n=== Two-teacher codec screen to share ===')
for name in ('methods', 'stress'):
    r = json.loads((out / f'{name}.report.json').read_text())
    print(f"{name}: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; similarity={r['mean_character_similarity']:.2%}")
    if r['compile_requested']:
        print(f"  raw compiles={r['reconstruction_compile_passes']}/{r['examples']}; originals={r['original_compile_passes']}/{r['examples']}")
print('Rare-only reference: methods exact 218/348; stress 98.92%, 39/48 exact, 39/48 compile.')
print('Repeat reference: methods exact 224/348; stress 98.80%, 37/48 exact, 39/48 compile.')
print('Known-input codec reconstruction only; no generation or semantic correctness measured.')
PY
