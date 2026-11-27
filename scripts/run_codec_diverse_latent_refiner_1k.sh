#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/codec-diverse-data-5k"
CHECKPOINT="$MODEL/checkpoints/step-00005000.pt"
METHODS="$BASE/codec-method-boundary-rare-repeat/heldout-methods.jsonl"
EXTERNAL="$BASE/codec-external-methods/methods.jsonl"
OUT="$BASE/codec-diverse-latent-refiner-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
 echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$MODEL/config.yaml" "$CHECKPOINT" "$METHODS" "$EXTERNAL" \
 "$MODEL/methods-candidate.jsonl" "$MODEL/methods-candidate.report.json" \
 "$MODEL/external-candidate.jsonl" "$MODEL/external-candidate.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/adapter.pt" "$OUT/methods.jsonl" "$OUT/methods.report.json" "$OUT/external.jsonl" "$OUT/external.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_latent_refiner.py
.venv/bin/python evaluation/test_codec_latent_refiner_integration.py
.venv/bin/python - "$MODEL/config.yaml" "$CHECKPOINT" "$MODEL/methods-candidate.report.json" "$METHODS" "$MODEL/external-candidate.report.json" "$EXTERNAL" <<'PY'
import json, sys, torch
from pathlib import Path
from omegaconf import OmegaConf
from training.data import sha256_file
cfg = OmegaConf.load(sys.argv[1]); OmegaConf.resolve(cfg)
ckpt = torch.load(sys.argv[2], map_location='cpu', weights_only=False)
if int(ckpt.get('step',-1)) != 4999: raise SystemExit('Expected step-5000 codec checkpoint')
del ckpt
if not Path(cfg.data.cache_dir,'manifest.json').is_file() or not Path(cfg.train.validation_cache_dir,'manifest.json').is_file():
    raise SystemExit('Training or validation cache is missing')
for report, source in ((sys.argv[3],sys.argv[4]),(sys.argv[5],sys.argv[6])):
    r = json.loads(Path(report).read_text())
    if r['input_sha256'] != sha256_file(source) or Path(r['checkpoint']).resolve() != Path(sys.argv[2]).resolve():
        raise SystemExit(f'Baseline report does not match input and checkpoint: {report}')
print('Broader-data codec, validation cache and saved baselines verified.', flush=True)
PY
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode train --config "$MODEL/config.yaml" \
 --checkpoint "$CHECKPOINT" --output "$OUT/adapter.pt" --steps 1000 --validate-every 500 --batch-size 8 --device cuda
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS"; else INPUT="$EXTERNAL"; fi
 .venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$MODEL/config.yaml" \
  --checkpoint "$CHECKPOINT" --adapter "$OUT/adapter.pt" --input "$INPUT" --baseline "$MODEL/$name-candidate.jsonl" \
  --batch-size 8 --device cuda --output "$OUT/$name.jsonl"
done
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); print('\n=== Broader-data codec and latent adapter ===')
for name in ('methods','external'):
    r = json.loads((root/f'{name}.report.json').read_text())
    print(f"{name}: exact {r['baseline_exact']}/{r['examples']} -> {r['adapter_exact']}/{r['examples']}; gained={r['gained_exact']}; lost={r['lost_exact']}; token accuracy {r['baseline_token_accuracy']:.4%} -> {r['adapter_token_accuracy']:.4%}")
print('Single frozen broader-data codec; adapter uses only VQ latents. Checkpoint selected by cached validation token accuracy.')
print('Reused development datasets; no compile, behavior, generator or fresh-holdout evaluation.')
PY
