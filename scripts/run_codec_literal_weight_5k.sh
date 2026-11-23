#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXT="$BASE/codec-external-methods"
OUT="$BASE/codec-literal-weight-5k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
  echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF/artifact/tokenizer.pt" "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$EXT/methods.jsonl" "$EXT/a.jsonl"; do
  [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
if [ -e "$OUT/config.yaml" ] || [ -e "$OUT/cache" ] || [ -e "$OUT/checkpoints" ] || [ -e "$OUT/artifact" ]; then
  echo "Existing experiment output in $OUT; stopping to preserve it." >&2; exit 1
fi
.venv/bin/python evaluation/test_literal_codec_weights.py
.venv/bin/python evaluation/test_comment_codec_weights.py
.venv/bin/python evaluation/test_method_name_weighting.py
.venv/bin/python evaluation/test_repeated_identifier_weights.py
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
cfg.task = 'codec-literal-weight-5k'
cfg.data.cache_dir = out + '/cache'
cfg.train.run_dir = out
cfg.train.artifact_dir = out + '/artifact'
cfg.train.eval_every = 1000
cfg.train.save_every = 1000
cfg.train.reconstruction_token_weights.literal = 2.0
OmegaConf.save(cfg, out + '/config.yaml')
print('New training cache; literal occurrence weight 2.0, other reference weights unchanged.', flush=True)
print('Fresh 64-level codec, 5,000 updates; validation is unweighted.', flush=True)
PY
.venv/bin/python -m training.prepare_data --config "$OUT/config.yaml" --stage tokenizer
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --stop-after-steps 5000
for name in methods external; do
  if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; REFERENCE="$METHODS/reference.jsonl"; else INPUT="$EXT/methods.jsonl"; REFERENCE="$EXT/a.jsonl"; fi
  .venv/bin/python evaluation/godot_csharp_codec_eval.py \
    --config "$OUT/config.yaml" --checkpoint "$OUT/artifact/tokenizer.pt" \
    --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name.jsonl"
  .venv/bin/python evaluation/compare_codec_pairs.py --reference "$REFERENCE" --candidate "$OUT/$name.jsonl"
done
.venv/bin/python - "$EXT/methods.jsonl" "$OUT/external.jsonl" "$EXT/a.jsonl" <<'PY'
import json, sys
from collections import defaultdict
def load(path):
    rows = [json.loads(line) for line in open(path, encoding='utf-8') if line.strip()]
    result = {row['id']: row for row in rows}
    if len(result) != len(rows) or not rows: raise SystemExit(f'Empty or duplicate IDs: {path}')
    return result
source, candidate, reference = map(load, sys.argv[1:])
if source.keys() != candidate.keys() or source.keys() != reference.keys(): raise SystemExit('External IDs differ')
groups = defaultdict(lambda: [0, 0, 0])
for key, row in source.items():
    if candidate[key]['target'] != row['target'] or reference[key]['target'] != row['target']: raise SystemExit(f'Target mismatch: {key}')
    group = groups[row['source'].split('/')[0]]
    group[0] += 1
    group[1] += reference[key]['reconstructed'] == row['target']
    group[2] += candidate[key]['reconstructed'] == row['target']
print('\n=== External project exact text ===')
for name, (n, old, new) in sorted(groups.items()): print(f'{name}: reference {old}/{n}; literal-weight {new}/{n}')
print('Known-target single-codec round trips; these small external samples have now informed experiment design.')
print('No signature copying, compilation, behavioral or generator tests. Tokenizer file identity is not independently verified.')
PY
