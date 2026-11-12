#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=runs/context256-codec-comparison/compact-ar64-5k
SOURCE=config/train/godot-csharp-deepseek-coder-1.3b-strict-8level-code512-decoder4-ctx256-ast-occurrence-weighted-30k.yaml
CFG="$OUT/config.yaml"
mkdir -p "$OUT"
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/tokenizer.pt" ]; then
  echo "Existing checkpoint found at $OUT; stopping to protect it." >&2
  exit 1
fi
echo '===== CPU correctness smoke test ====='
.venv/bin/python evaluation/test_compact_autoregressive_codec.py
.venv/bin/python - "$SOURCE" "$CFG" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
source, dest, out = sys.argv[1:]
cfg = OmegaConf.load(source)
cfg.task = 'compact-ar64-5k'
cfg.vqvae.compression_factor = 4
cfg.vqvae.enc_layers = 4
cfg.vqvae.dec_layers = 4
cfg.vqvae.dec_positional = True
cfg.vqvae.autoregressive_decoder = True
# A dedicated learned start embedding is obtained from a token already in the vocabulary.
cfg.vqvae.decoder_bos_token_id = 0
cfg.vqvae.vqvae_vocab_size = [512]
cfg.vqvae.vector_quantizer_config.levels.manual_levels = [64]
cfg.vqvae.vector_quantizer_config.aux.fine_drop.prob = 0.0
cfg.train.precision = 'bf16'
cfg.train.max_steps = 30000
cfg.train.eval_every = 5000
cfg.train.save_every = 5000
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
OmegaConf.save(cfg, dest)
print('Compact experiment: 64 indices/chunk (512 choices each), 4x sequence reduction.')
print('Validation during training is teacher-forced; the evaluations below are free-running.')
print('Config:', dest, flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$CFG" --stop-after-steps 5000
echo '===== Free-running held-out reconstruction (may take several minutes) ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input data/mycode/godot-csharp-deepseek-coder-1.3b-strict-ctx256-ast/validation.jsonl \
  --text-key text --code-only --batch-size 16 --exact-residual --measure-packed \
  --output "$OUT/validation.jsonl"
echo '===== Free-running compile stress ====='
.venv/bin/python evaluation/godot_csharp_codec_eval.py \
  --config "$CFG" --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --batch-size 16 --compile --progress --exact-residual --measure-packed \
  --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Compact autoregressive 5k results to share ===')
for name in ('validation', 'stress'):
    report = json.loads((root / f'{name}.report.json').read_text())
    packed = report['packed_measurement']
    print(f"{name}: free-running accuracy={report['valid_token_accuracy']:.2%}; exact={report['exact_sequence_matches']}/{report['examples']}; VQ={packed['vq_code_bytes']} B; residual={packed['residual_bytes']} B; total/input={packed['total_to_input_ratio']:.2f}x; decode={packed['decode_and_argmax_seconds']:.2f}s")
    if report['compile_requested']:
        print(f"  raw compile={report['reconstruction_compile_passes']}/{report['examples']}; original compile={report['original_compile_passes']}/{report['examples']}")
print('Reference: 64-level no-drop stress 28/48 raw compile, 83.30x encoded/input.')
print('Residual requires original target tokens and cannot correct generated code.')
PY
