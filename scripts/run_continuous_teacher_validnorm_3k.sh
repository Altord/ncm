#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
SOURCE="$BASE/codec-continuous-teacher-validnorm-1k"
OUT="$BASE/codec-continuous-teacher-validnorm-3k"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
REF_STEP="$REF/checkpoints/step-00003000.pt"
SOURCE_STEP="$SOURCE/checkpoints/step-00001000.pt"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
 echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$REF/config.yaml" "$REF_STEP" "$SOURCE/config.yaml" "$SOURCE_STEP" "$METHODS" "$EXTERNAL"; do
 [ -f "$f" ] || { echo "Missing: $f; cannot run the matched continuation." >&2; exit 1; }
done
if [ -e "$OUT/config.yaml" ] || [ -e "$OUT/checkpoints" ] || [ -e "$OUT/artifact" ]; then
 echo "Existing experiment output in $OUT; stopping to preserve it." >&2; exit 1
fi
for name in methods-reference methods-candidate external-reference external-candidate; do
 [ ! -e "$OUT/$name.jsonl" ] && [ ! -e "$OUT/$name.report.json" ] || { echo "Existing output: $OUT/$name; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_continuous_codec_teacher.py
.venv/bin/python - "$REF/config.yaml" "$REF_STEP" "$SOURCE/config.yaml" "$SOURCE_STEP" "$OUT/config.yaml" "$OUT" <<'PY'
import sys, torch
from pathlib import Path
from omegaconf import OmegaConf
ref, src = OmegaConf.load(sys.argv[1]), OmegaConf.load(sys.argv[3])
OmegaConf.resolve(ref); OmegaConf.resolve(src)
for key in ('tokenizer', 'vqvae', 'data'):
    if OmegaConf.to_container(ref[key], resolve=True) != OmegaConf.to_container(src[key], resolve=True):
        raise SystemExit(f'Reference and candidate {key} differ')
teacher = src.train.get('continuous_distillation', {})
if float(teacher.get('weight', 0)) != 0.1 or int(teacher.get('topk', 0)) != 4 or str(teacher.get('normalization', '')) != 'valid':
    raise SystemExit('Unexpected teacher settings')
for path, step in ((sys.argv[2], 2999), (sys.argv[4], 999)):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if int(ckpt.get('step', -1)) != step: raise SystemExit(f'Unexpected stored step in {path}')
    if path == sys.argv[4] and ('optimizer' not in ckpt or 'rng' not in ckpt):
        raise SystemExit('Source checkpoint lacks optimizer or RNG state')
    del ckpt
src.train.run_dir = sys.argv[6]
src.train.artifact_dir = sys.argv[6] + '/artifact'
src.train.eval_every = 1000
src.train.save_every = 1000
Path(sys.argv[6]).mkdir(parents=True, exist_ok=True)
OmegaConf.save(src, sys.argv[5])
print('Resume 1,000 -> 3,000 updates; checkpoint, optimizer and RNG restored. Data-loader order restarts.', flush=True)
print('Stops at 3,000; existing learning-rate schedule remains unchanged.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m training.train_tokenizer --config "$OUT/config.yaml" --resume "$SOURCE_STEP" --stop-after-steps 3000
NEW_STEP="$OUT/checkpoints/step-00003000.pt"
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; else INPUT="$EXTERNAL"; fi
 for kind in reference candidate; do
  if [ "$kind" = reference ]; then MODEL="$REF"; CHECKPOINT="$REF_STEP"; else MODEL="$OUT"; CHECKPOINT="$NEW_STEP"; fi
  .venv/bin/python evaluation/godot_csharp_codec_eval.py --config "$MODEL/config.yaml" --checkpoint "$CHECKPOINT" \
   --input "$INPUT" --text-key target --code-only --batch-size 8 --output "$OUT/$name-$kind.jsonl"
 done
 echo "===== $name: matched 3,000-step comparison ====="
 .venv/bin/python evaluation/compare_codec_pairs.py --reference "$OUT/$name-reference.jsonl" --candidate "$OUT/$name-candidate.jsonl"
done
.venv/bin/python - "$REF/logs/validation.jsonl" "$OUT/logs/validation.jsonl" "$OUT" <<'PY'
import json, sys
from pathlib import Path
print('\n=== 3,000-step validation comparison ===')
for label, path in [('reference', sys.argv[1]), ('candidate', sys.argv[2])]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    matched = [r for r in rows if r.get('step') == 3000]
    if not matched: print(f'{label}: missing step-3000 validation'); continue
    r = matched[-1]
    print(f"{label}: accuracy={r['val_token_accuracy']:.2%}; exact={r['val_exact_sequence_match_rate']:.2%}; CE={r['val_recon_ce']:.4f}")
for name in ('methods', 'external'):
    reports = [json.loads((Path(sys.argv[3]) / f'{name}-{kind}.report.json').read_text()) for kind in ('reference', 'candidate')]
    a, b = reports
    print(f"{name}: accuracy {a['valid_token_accuracy']:.2%} -> {b['valid_token_accuracy']:.2%}; exact {a['exact_sequence_matches']}/{a['examples']} -> {b['exact_sequence_matches']}/{b['examples']}")
print('Both evaluated checkpoints are step 3,000; candidate data-loader order restarted on resume.')
print('Known-target codec, reused development sets; no compile, behavior or generator evaluation.')
PY
