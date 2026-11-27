#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat"
EXTERNAL="$BASE/codec-external-methods"
PRIOR="$BASE/codec-latent-refiner-1k"
OUT="$BASE/codec-masked-latent-refiner-1k"
if pgrep -f '[p]ython([0-9.]*)? .*training[.](train_tokenizer|train_generator|train_prompt_copy_screen|train_context_refinement_screen)' >/dev/null; then
 echo 'A trainer is already running; finish or stop it first.' >&2; exit 1
fi
for f in "$MODEL/config.yaml" "$MODEL/artifact/tokenizer.pt" "$METHODS/heldout-methods.jsonl" "$METHODS/reference.jsonl" "$METHODS/reference.report.json" "$EXTERNAL/methods.jsonl" "$EXTERNAL/a.jsonl" "$EXTERNAL/a.report.json" "$PRIOR/methods.jsonl" "$PRIOR/methods.report.json" "$PRIOR/external.jsonl" "$PRIOR/external.report.json"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/adapter.pt" "$OUT/methods.jsonl" "$OUT/methods.report.json" "$OUT/external.jsonl" "$OUT/external.report.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_codec_latent_refiner.py
.venv/bin/python evaluation/test_codec_latent_refiner_integration.py
mkdir -p "$OUT"
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
.venv/bin/python -m evaluation.codec_latent_refiner_screen --mode train --config "$MODEL/config.yaml" \
 --checkpoint "$MODEL/artifact/tokenizer.pt" --output "$OUT/adapter.pt" --steps 1000 --validate-every 500 \
 --batch-size 8 --architecture mlp --latent-mask valid --device cuda
for name in methods external; do
 if [ "$name" = methods ]; then INPUT="$METHODS/heldout-methods.jsonl"; BASELINE="$METHODS/reference.jsonl"; else INPUT="$EXTERNAL/methods.jsonl"; BASELINE="$EXTERNAL/a.jsonl"; fi
 .venv/bin/python -m evaluation.codec_latent_refiner_screen --mode evaluate --config "$MODEL/config.yaml" \
  --checkpoint "$MODEL/artifact/tokenizer.pt" --adapter "$OUT/adapter.pt" --input "$INPUT" --baseline "$BASELINE" \
  --batch-size 8 --device cuda --output "$OUT/$name.jsonl"
done
.venv/bin/python - "$PRIOR" "$OUT" <<'PY'
import json, sys
from pathlib import Path
prior, out = map(Path, sys.argv[1:])
def load(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    result = {r['id']: r for r in rows}
    if not rows or len(result) != len(rows): raise SystemExit(f'Empty or duplicate IDs: {path}')
    return result
for name in ('methods','external'):
    old_report, new_report = (json.loads((root/f'{name}.report.json').read_text()) for root in (prior, out))
    for key in ('input_sha256','codec_sha256','examples','valid_tokens'):
        if old_report[key] != new_report[key]: raise SystemExit(f'{name}: mismatched {key}')
    old, new = (load(root/f'{name}.jsonl') for root in (prior, out))
    if old.keys() != new.keys() or any(old[k]['baseline_exact'] != new[k]['baseline_exact'] for k in old):
        raise SystemExit(f'{name}: IDs or baseline scores differ')
    gains = sum(not old[k]['adapter_exact'] and new[k]['adapter_exact'] for k in old)
    losses = sum(old[k]['adapter_exact'] and not new[k]['adapter_exact'] for k in old)
    print(f"\n{name}: prior exact={old_report['adapter_exact']}/{len(old)}; masked-latent exact={new_report['adapter_exact']}/{len(new)}; gained={gains}; lost={losses}")
    print(f"Token accuracy: {old_report['adapter_token_accuracy']:.4%} -> {new_report['adapter_token_accuracy']:.4%}")
print('Known-target codec reconstruction; no compile, behavioral or generator test. Reused development sets.')
PY
