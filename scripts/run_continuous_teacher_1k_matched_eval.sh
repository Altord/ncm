#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
NEW="$BASE/codec-continuous-teacher-validnorm-1k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
OUT="$BASE/codec-continuous-teacher-validnorm-1k-matched-eval"
REF_STEP="$REF/checkpoints/step-00001000.pt"
NEW_STEP="$NEW/checkpoints/step-00001000.pt"
for f in "$REF/config.yaml" "$REF_STEP" "$NEW/config.yaml" "$NEW_STEP" "$METHODS" "$EXTERNAL"; do
 [ -f "$f" ] || { echo "Missing: $f; matched 1,000-step evaluation cannot run." >&2; exit 1; }
done
for name in methods-reference methods-candidate external-reference external-candidate; do
 [ ! -e "$OUT/$name.jsonl" ] && [ ! -e "$OUT/$name.report.json" ] || { echo "Existing output: $OUT/$name; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python - "$REF/config.yaml" "$REF_STEP" "$NEW/config.yaml" "$NEW_STEP" <<'PY'
import sys, torch
from omegaconf import OmegaConf
a, b = OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[3])
OmegaConf.resolve(a); OmegaConf.resolve(b)
for key in ('tokenizer', 'vqvae', 'data'):
    if OmegaConf.to_container(a[key], resolve=True) != OmegaConf.to_container(b[key], resolve=True):
        raise SystemExit(f'Reference and candidate {key} configurations differ')
for path in (sys.argv[2], sys.argv[4]):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if int(checkpoint.get('step', -1)) != 999:
        raise SystemExit(f'Expected stored zero-based step 999 in {path}')
    del checkpoint
print('Tokenizer, codec and data configurations match; both checkpoints are step 1,000.', flush=True)
PY
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; else INPUT="$EXTERNAL"; fi
 for kind in reference candidate; do
  if [ "$kind" = reference ]; then MODEL="$REF"; CHECKPOINT="$REF_STEP"; else MODEL="$NEW"; CHECKPOINT="$NEW_STEP"; fi
  .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$CHECKPOINT" \
   --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name-$kind.jsonl"
 done
 echo "===== $name: matched 1,000-step comparison ====="
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/$name-reference.jsonl" --candidate "$OUT/$name-candidate.jsonl"
done
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Matched 1,000-step summary ===')
for name in ('methods', 'external'):
    reports = [json.loads((root / f'{name}-{kind}.report.json').read_text()) for kind in ('reference', 'candidate')]
    a, b = reports
    print(f"{name}: accuracy {a['valid_token_accuracy']:.2%} -> {b['valid_token_accuracy']:.2%}; exact {a['exact_sequence_matches']}/{a['examples']} -> {b['exact_sequence_matches']}/{b['examples']}")
print('Both are fresh-training step-1,000 checkpoints. This does not predict 5k outcomes or test compilation or generation.')
print('Reused development datasets; tokenizer configuration is not a file-identity check.')
PY
