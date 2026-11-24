#!/usr/bin/env bash
set -euo pipefail
trap 'ncm_exit_status=$?; printf "\a" || true; exit "$ncm_exit_status"' EXIT
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
RUN="$BASE/codec-parameter-augmentation-5k"
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width192-rare2-5k"
METHOD_REF="$BASE/codec-method-boundary-rare-repeat/reference.jsonl"
EXTERNAL_REF="$BASE/codec-external-methods/a.jsonl"
OUT="$BASE/codec-parameter-augmentation-diagnosis"
for f in "$RUN/config.yaml" "$RUN/cache/manifest.json" "$RUN/augmented-train.jsonl" "$REF/config.yaml" "$METHOD_REF" "$EXTERNAL_REF" "$RUN/methods.jsonl" "$RUN/external.jsonl"; do
 [ -f "$f" ] || { echo "Missing: $f" >&2; exit 1; }
done
for f in "$OUT/methods.json" "$OUT/external.json"; do
 [ ! -e "$f" ] || { echo "Existing output: $f; stopping to preserve it." >&2; exit 1; }
done
.venv/bin/python evaluation/test_compare_codec_error_regions.py
mkdir -p "$OUT"
.venv/bin/python - "$REF/config.yaml" "$RUN/augmented-train.jsonl" "$RUN/cache/manifest.json" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
from training.data import sha256_file
cfg = OmegaConf.load(sys.argv[1]); OmegaConf.resolve(cfg)
original = Path(str(cfg.data.input)); augmented = Path(sys.argv[2])
manifest = json.loads(Path(sys.argv[3]).read_text(encoding='utf-8'))
if manifest['source_sha256'] != sha256_file(augmented):
    raise SystemExit('Training cache and augmented corpus differ')
def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)
original_iter = rows(original)
mixed_iter = rows(augmented)
count = 0
for item in original_iter:
    if next(mixed_iter, None) != item: raise SystemExit(f'Original corpus prefix differs at row {count}')
    count += 1
extra = sum(1 for _ in mixed_iter)
print(f'\n=== Training corpus coverage ===\nOriginal records: {count}; appended renamed records: {extra}; total: {count + extra}')
print('The count is corpus augmentation, not a measure of semantically valid renames or identifier accuracy.')
PY
for name in methods external; do
 if [ "$name" = methods ]; then REFERENCE="$METHOD_REF"; else REFERENCE="$EXTERNAL_REF"; fi
 echo "===== $name region comparison ====="
 .venv/bin/python -m evaluation.compare_codec_error_regions --reference "$REFERENCE" --candidate "$RUN/$name.jsonl" --output "$OUT/$name.json"
 echo "===== $name method-name comparison ====="
 .venv/bin/python evaluation/compare_codec_method_names.py --reference "$REFERENCE" --candidate "$RUN/$name.jsonl"
done
echo 'Offline known-target diagnostics; categories overlap, AST/diff matching is heuristic, and method names are not all identifiers.'
