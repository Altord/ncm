#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
OUT="$BASE/bf16-aot-64level-encoder8-width192-rare2-method4-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator)' >/dev/null; then
  echo 'A trainer is running; finish or stop it first.' >&2; exit 1
fi
for path in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$METHODS"; do
  [ -f "$path" ] || { echo "Missing: $path" >&2; exit 1; }
done
if [ -e "$OUT/checkpoints/latest.pt" ] || [ -e "$OUT/artifact/tokenizer.pt" ] || [ -e "$OUT/cache/manifest.json" ]; then
  echo "Existing checkpoint or prepared cache in $OUT; stopping to protect it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_method_name_weighting.py
.venv/bin/python - "$REF/config.yaml" "$OUT" <<'PY'
import sys
from omegaconf import OmegaConf
src, out = sys.argv[1:]
cfg = OmegaConf.load(src)
if (int(cfg.vqvae.embed_dim) != 192 or int(cfg.vqvae.enc_layers) != 8 or
    len(cfg.vqvae.vqvae_vocab_size) != 64 or
    str(cfg.train.reconstruction_token_weights.mode) != 'per_occurrence' or
    not cfg.train.rare_token_weights.enabled):
    raise SystemExit('Unexpected baseline configuration; stopping')
cfg.task = 'codec-method-name-weight-5k'
cfg.data.cache_dir = out + '/cache'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
cfg.train.reconstruction_token_weights.method_name = 4.0
OmegaConf.save(cfg, out + '/config.yaml')
PY
.venv/bin/python -m training.prepare_data --config "$OUT/config.yaml" --stage tokenizer
.venv/bin/python - "$REF/config.yaml" "$OUT/config.yaml" <<'PY'
import sys
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from training.data import MemmapCache
oldcfg, newcfg = map(OmegaConf.load, sys.argv[1:])
old, new = MemmapCache(oldcfg.data.cache_dir), MemmapCache(newcfg.data.cache_dir)
if len(old) != len(new):
    raise SystemExit('Training cache row count changed')
count = 0
for oldrow, newrow in zip(old, new):
    for field in ('tokens', 'mask'):
        if not np.array_equal(oldrow[field], newrow[field]):
            raise SystemExit(f'Training cache {field} changed; stopping')
    a, b = np.asarray(oldrow['token_weights']), np.asarray(newrow['token_weights'])
    changed = a != b
    if np.any(changed & ((a != float(oldcfg.train.reconstruction_token_weights.identifier)) | (b != 4.0))):
        raise SystemExit('Unexpected loss-weight change; stopping')
    count += int(changed.sum())
if not count:
    raise SystemExit('No method-name token labels found; stopping before training')
print(f'Preflight: same {len(new)} examples and token IDs; {count} method-name token occurrences upweighted to 4.0.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" \
  --checkpoint "$OUT/artifact/tokenizer.pt" --input "$METHODS" --text-key target \
  --code-only --batch-size 16 --output "$OUT/methods.jsonl"
.venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$OUT/config.yaml" \
  --checkpoint "$OUT/artifact/tokenizer.pt" \
  --input examples/godot_csharp_generator_pilot/codec_compile_stress.jsonl \
  --text-key target --batch-size 16 --compile --progress --output "$OUT/stress.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$BASE/codec-method-boundary-rare-repeat/reference.jsonl" --candidate "$OUT/methods.jsonl"
.venv/bin/python evaluation/compare_codec_pairs.py --reference "$REF/stress.jsonl" --candidate "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Method-name-weight codec results to share ===')
for name in ('methods', 'stress'):
    r = json.loads((root / f'{name}.report.json').read_text())
    print(f"{name}: accuracy={r['valid_token_accuracy']:.2%}; exact={r['exact_sequence_matches']}/{r['examples']}")
    if r['compile_requested']:
        print(f"  raw compile={r['reconstruction_compile_passes']}/{r['examples']}")
print('Rare-only reference: methods 218/348 exact; stress 98.92%, 39/48 exact and 39/48 raw compiles.')
print('Known-input codec reconstruction; no generator quality or behavioral tests.')
PY
