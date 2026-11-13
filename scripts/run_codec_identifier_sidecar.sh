#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
MODEL="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHODS="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
OUT="$BASE/codec-identifier-sidecar-rare2"
for path in "$MODEL/stress.jsonl" "$MODEL/stress.report.json" "$METHODS" "${METHODS%.jsonl}.report.json" "$MODEL/config.yaml"; do
  [ -f "$path" ] || { echo "Missing: $path" >&2; exit 1; }
done
if [ -e "$OUT/methods.jsonl" ] || [ -e "$OUT/stress.jsonl" ]; then
  echo "Existing experiment output in $OUT; stopping to preserve it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python evaluation/test_codec_identifier_sidecar.py
.venv/bin/python - "$MODEL" "$METHODS" <<'PY'
import json, math, sys
from pathlib import Path
from omegaconf import OmegaConf
model, methods = map(Path, sys.argv[1:])
cfg = OmegaConf.load(model / 'config.yaml')
levels = list(cfg.vqvae.vector_quantizer_config.levels.manual_levels)
sizes = list(cfg.vqvae.vqvae_vocab_size)
if len(levels) != 64 or len(levels) != len(sizes):
    raise SystemExit('Unexpected VQ configuration')
packed = math.ceil(sum(int(n) * (int(size)-1).bit_length() for n, size in zip(levels, sizes)) / 8)
if packed != 9219:
    raise SystemExit(f'Unexpected VQ bytes per record: {packed}')
checkpoint = (model / 'artifact/tokenizer.pt').resolve()
for path in (model / 'stress.report.json', methods.with_suffix('.report.json')):
    report = json.loads(path.read_text())
    if Path(report['checkpoint']).resolve() != checkpoint:
        raise SystemExit(f'Checkpoint mismatch in {path}')
print(f'Preflight: same frozen checkpoint and {packed} fixed VQ bytes per record.', flush=True)
PY
echo '===== Known-source identifier sidecar: full methods ====='
.venv/bin/python evaluation/codec_identifier_sidecar.py --input "$METHODS" \
  --vq-bytes-per-record 9219 --output "$OUT/methods.jsonl"
echo '===== Known-source identifier sidecar: stress + compile ====='
.venv/bin/python evaluation/codec_identifier_sidecar.py --input "$MODEL/stress.jsonl" \
  --vq-bytes-per-record 9219 --compile-changed --output "$OUT/stress.jsonl"
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
print('\n=== Identifier-sidecar results to share ===')
for name in ('methods', 'stress'):
    r = json.loads((root / f'{name}.report.json').read_text())
    print(f"{name}: exact text {r['before_exact']} -> {r['after_exact']}/{r['examples']}; "
          f"character similarity {r['before_mean_character_similarity']:.2%} -> {r['after_mean_character_similarity']:.2%}; "
          f"identifier sidecar={r['sidecar_bytes']} B; VQ+sidecar/input={r['vq_plus_sidecar_to_input']:.2f}x")
    if r['compiles_before'] is not None:
        print(f"  compile: {r['compiles_before']} -> {r['compiles_after']}/{r['examples']}; new compiler calls={r['new_compiler_calls']}")
print('Original source is needed to build the correction sidecar; no raw codec or generator improvement is claimed.')
print('The VQ byte figure uses prior fixed 9,219 B per 64-level record and excludes other framing and model costs.')
PY
