#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-5k"
OUT="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
CFG="$OUT/config.yaml"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator)' >/dev/null; then
  echo 'A tokenizer trainer is already running; finish or stop it first.' >&2
  exit 1
fi
if [ ! -f "$REF/config.yaml" ]; then
  echo "Missing reference configuration: $REF/config.yaml" >&2
  exit 1
fi
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/tokenizer.pt" ]; then
  echo "Existing checkpoint in $OUT; stopping to protect it." >&2
  exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_rare_codec_weights.py
.venv/bin/python - "$REF/config.yaml" "$CFG" "$OUT" <<'PY'
import sys
import json
from pathlib import Path
from omegaconf import OmegaConf
from training.data import MemmapCache
from training.reconstruction_loss import training_token_frequency_weights
source, dest, out = sys.argv[1:]
cfg = OmegaConf.load(source)
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    int(cfg.vqvae.dec_layers) != 4 or
    len(cfg.vqvae.vqvae_vocab_size) != 64 or
    float(cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob) != 0.0):
    raise ValueError('Reference config does not match expected 64-level no-drop model')
cfg.task = 'ctx256-64level-encoder8-nodrop-width192-rare2-5k'
cfg.train.rare_token_weights = {'enabled': True, 'threshold': 32, 'max_weight': 2.0}
diagnosis = Path('runs/context256-codec-comparison/codec-remaining-token-diagnosis/report.json')
if not diagnosis.exists():
    raise SystemExit(f'Missing prior diagnosis needed for the rare-token preflight: {diagnosis}')
weights, counts = training_token_frequency_weights(MemmapCache(cfg.data.cache_dir),
                                                     int(cfg.vqvae.in_vocab_size), 32, 2.0)
data = json.loads(diagnosis.read_text(encoding='utf-8'))
print('Offline diagnosis: training counts for previously missed target pieces (not used to fit weights):')
seen = set()
for row in data['examples']:
    for error in row['errors']:
        token_id = int(error['target_token_id'])
        if token_id not in seen and 0 <= token_id < len(counts):
            seen.add(token_id)
            print(f"  {error['target_piece']!r}: train count={counts[token_id]}; multiplier={weights[token_id]:.2f}x")
if not any(weights[token_id] > 1 for token_id in seen):
    raise SystemExit('None of these missed tokens qualify for upweighting; stopping before training')
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
OmegaConf.save(cfg, dest)
print('Upweight rare training token IDs up to 2x, multiplied by existing per-occurrence weights.')
print('Validation every 1000 updates; selected checkpoint uses validation token accuracy.')
print('Same codec architecture, data and 64-level index layout; validation metrics remain unweighted.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$CFG" --stop-after-steps 5000
echo '===== Held-out validation reconstruction ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --batch-size 16 --exact-residual --measure-packed \
  --output "$OUT/validation.jsonl"
echo '===== Compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --batch-size 16 --compile --progress --exact-residual --measure-packed \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Width 192 encoder 8 rare-token screen results to share ===')
for name in ('validation', 'stress'):
    r = json.loads((root / f'{name}.report.json').read_text())
    p = r['packed_measurement']
    print(f"{name}: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}; total/input={p['total_to_input_ratio']:.2f}x; VQ={p['vq_code_bytes']} B; corrections={p['residual_bytes']} B")
    if r['compile_requested']:
        print(f"  raw compiles={r['reconstruction_compile_passes']}/{r['examples']}; original compiles={r['original_compile_passes']}/{r['examples']}")
print('Width 192 encoder 8 reference: stress accuracy=98.80%; exact=36/48; raw compile=38/48.')
print('Validation uses FP32 evaluation; model size, memory and runtime are not included in packed bytes.')
PY
