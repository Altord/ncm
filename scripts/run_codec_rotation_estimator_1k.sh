#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/codec-diverse-data-3k"
OUT="$BASE/codec-rotation-estimator-1k"
for f in "$REF/config.yaml" "$REF/checkpoints/step-00001000.pt" "$REF/cache/manifest.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/checkpoints" "$OUT/artifact" "$OUT/methods-candidate.jsonl" "$OUT/external-candidate.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator)' >/dev/null; then
 echo 'A trainer is already running.' >&2; exit 1
fi
.venv/bin/python -m evaluation.test_rotation_straight_through
.venv/bin/python - "$REF/config.yaml" "$OUT" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
from training.data import sha256_file
cfg = OmegaConf.load(sys.argv[1]); OmegaConf.resolve(cfg)
manifest = json.loads((Path(cfg.data.cache_dir)/'manifest.json').read_text())
if manifest['source_sha256'] != sha256_file(cfg.data.input): raise SystemExit('Training cache source differs')
if len(cfg.vqvae.vqvae_vocab_size) != 64: raise SystemExit('Expected 64 levels')
out = sys.argv[2]; Path(out).mkdir(parents=True, exist_ok=True)
cfg.vqvae.vector_quantizer_config.gradient_estimator = 'rotation'
cfg.train.run_dir = out; cfg.train.artifact_dir = out+'/artifact'
cfg.train.eval_every = 1000; cfg.train.save_every = 1000
OmegaConf.save(cfg, out+'/config.yaml')
print('Fresh 1k screen: same data, seed, architecture and schedule; aggregate rotation replaces STE gradient.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 1000
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"; else INPUT="$BASE/codec-external-methods/methods.jsonl"; fi
 for kind in reference candidate; do
  if [ "$kind" = reference ]; then MODEL="$REF"; else MODEL="$OUT"; fi
  .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$MODEL/checkpoints/step-00001000.pt" \
   --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name-$kind.jsonl"
 done
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/$name-reference.jsonl" --candidate "$OUT/$name-candidate.jsonl"
done
echo 'Early 1k diagnostic; reused development data. No compilation, behavior, generation, or fresh holdout test.'
