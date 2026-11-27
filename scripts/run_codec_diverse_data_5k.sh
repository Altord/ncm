#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
SOURCE="$BASE/codec-diverse-data-3k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
OUT="$BASE/codec-diverse-data-5k"
SOURCE_STEP="$SOURCE/checkpoints/step-00003000.pt"
REF_STEP="$REF/checkpoints/step-00005000.pt"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
 echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF_STEP" "$SOURCE/config.yaml" "$SOURCE_STEP" "$SOURCE/cache/manifest.json" "$METHODS" "$EXTERNAL"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/config.yaml" "$OUT/checkpoints" "$OUT/artifact" "$OUT/methods-reference.jsonl" "$OUT/methods-candidate.jsonl" "$OUT/external-reference.jsonl" "$OUT/external-candidate.jsonl"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python - "$REF/config.yaml" "$REF_STEP" "$SOURCE/config.yaml" "$SOURCE_STEP" "$OUT" <<'PY'
import sys, torch
from pathlib import Path
from omegaconf import OmegaConf
from training.data import sha256_file
ref, src = OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[3])
OmegaConf.resolve(ref); OmegaConf.resolve(src)
for key in ('tokenizer','vqvae','model'):
    if OmegaConf.to_container(ref[key], resolve=True) != OmegaConf.to_container(src[key], resolve=True):
        raise SystemExit(f'Reference and candidate {key} differ')
for path, step in ((sys.argv[2], 4999), (sys.argv[4], 2999)):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if int(ckpt.get('step', -1)) != step: raise SystemExit(f'Unexpected step in {path}')
    if path == sys.argv[4] and ('optimizer' not in ckpt or 'rng' not in ckpt): raise SystemExit('Missing optimizer or RNG')
    del ckpt
import json
cache = json.loads((Path(src.data.cache_dir)/'manifest.json').read_text())
if cache['source_sha256'] != sha256_file(src.data.input): raise SystemExit('Training cache input hash differs')
out = sys.argv[5]; Path(out).mkdir(parents=True, exist_ok=True)
src.train.run_dir = out; src.train.artifact_dir = out+'/artifact'
src.train.eval_every = 1000; src.train.save_every = 1000
OmegaConf.save(src, out+'/config.yaml')
print('Resume 3,000 -> 5,000 total updates with optimizer and RNG; data-loader order restarts.', flush=True)
print('Training cache, 64-level model, original rarity weights and schedule remain unchanged.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --resume "$SOURCE_STEP" --stop-after-steps 5000
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; else INPUT="$EXTERNAL"; fi
 for kind in reference candidate; do
  if [ "$kind" = reference ]; then MODEL="$REF"; STEP="$REF_STEP"; else MODEL="$OUT"; STEP="$OUT/checkpoints/step-00005000.pt"; fi
  .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$STEP" \
   --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name-$kind.jsonl"
 done
 echo "===== $name: matched 5,000-step comparison ====="
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/$name-reference.jsonl" --candidate "$OUT/$name-candidate.jsonl"
done
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); print('\n=== Broader data: matched 5,000-step summary ===')
for name in ('methods','external'):
    a, b = (json.loads((root/f'{name}-{kind}.report.json').read_text()) for kind in ('reference','candidate'))
    print(f"{name}: token accuracy {a['valid_token_accuracy']:.2%} -> {b['valid_token_accuracy']:.2%}; token-exact {a['exact_sequence_matches']}/{a['examples']} -> {b['exact_sequence_matches']}/{b['examples']}")
print('Candidate loader order restarted on resume; data size and effective epochs differ. Reused development sets.')
print('Known-target reconstruction; no compilation or generation. Fresh holdout is not evaluated during this tuning run.')
PY
