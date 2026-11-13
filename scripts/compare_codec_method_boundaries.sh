#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=runs/context256-codec-comparison
REF="$BASE/bf16-aot-64level-encoder8-nodrop-width128-5k"
FOCAL="$BASE/bf16-aot-64level-encoder8-nodrop-width128-focal1-5k"
DATA=data/mycode/godot-csharp-method-pairs-ctx256-prompt64/validation.jsonl
OUT="$BASE/codec-method-boundary-comparison"
if [ ! -f "$DATA" ] || [ ! -f "$REF/config.yaml" ] || [ ! -f "$FOCAL/config.yaml" ]; then
  echo 'Missing held-out method data or codec configuration.' >&2; exit 1
fi
if [ -e "$OUT/reference.report.json" ] || [ -e "$OUT/focal.report.json" ]; then
  echo "Existing comparison report in $OUT; stopping to protect it." >&2; exit 1
fi
mkdir -p "$OUT"
.venv/bin/python - "$REF/config.yaml" "$FOCAL/config.yaml" "$DATA" "$OUT" <<'PY'
import json, sys
from pathlib import Path
from omegaconf import OmegaConf
from training.data import load_base_tokenizer, encode_text, sha256_file
config, focal_config, data, out = sys.argv[1:]
cfg = OmegaConf.load(config)
other = OmegaConf.load(focal_config)
tok = load_base_tokenizer(cfg.tokenizer)
train = Path(cfg.data.input)
if not train.is_file():
    raise SystemExit(f'Missing codec training corpus for source separation check: {train}')
other_train = Path(other.data.input)
if (not other_train.is_file() or sha256_file(train) != sha256_file(other_train) or
    OmegaConf.to_container(cfg.tokenizer, resolve=True) != OmegaConf.to_container(other.tokenizer, resolve=True) or
    int(cfg.model.context_length) != int(other.model.context_length)):
    raise SystemExit('Codec training corpus, tokenizer or context differs; stopping comparison')
train_rows = [json.loads(line) for line in train.read_text(encoding='utf-8').splitlines() if line.strip()]
if any(not row.get('source') for row in train_rows):
    raise SystemExit('Codec training records lack source paths; cannot establish file separation')
train_sources = {row['source'] for row in train_rows}
rows = [json.loads(line) for line in Path(data).read_text(encoding='utf-8').splitlines() if line.strip()]
if not rows or any(not row.get('source') or not isinstance(row.get('target'), str) for row in rows):
    raise SystemExit('Method input is empty or lacks source/target fields')
overlap = sorted({row['source'] for row in rows} & train_sources)
if overlap:
    raise SystemExit(f'{len(overlap)} method source files overlap codec train files; stopping: {overlap[:5]}')
kept = [r for r in rows if len(encode_text(tok, r['target'], True)) <= int(cfg.model.context_length)]
if not kept:
    raise SystemExit('No methods fit the actual codec tokenizer context')
dest = Path(out) / 'heldout-methods.jsonl'
text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in kept)
if dest.exists() and dest.read_text(encoding='utf-8') != text:
    raise SystemExit('Existing filtered input differs; stopping')
if not dest.exists():
    dest.write_text(text, encoding='utf-8')
print(f'File-path overlap with codec training: 0; kept {len(kept)}/{len(rows)} full methods; '
      f'excluded {len(rows)-len(kept)} that exceed the actual context.', flush=True)
print('Source-path check does not rule out copied or near-duplicate code across files.', flush=True)
PY
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
for name in reference focal; do
  if [ "$name" = reference ]; then MODEL="$REF"; else MODEL="$FOCAL"; fi
  echo "===== $name: held-out-file method reconstruction ====="
  .venv/bin/python evaluation/godot_csharp_codec_eval.py \
    --config "$MODEL/config.yaml" --checkpoint "$MODEL/artifact/tokenizer.pt" \
    --input "$OUT/heldout-methods.jsonl" --text-key target --code-only \
    --batch-size 16 --exact-residual --measure-packed --output "$OUT/$name.jsonl"
done
.venv/bin/python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
print('\n=== Full-method codec comparison to share ===')
for name in ('reference', 'focal'):
    r = json.loads((out / f'{name}.report.json').read_text())
    print(f"{name}: examples={r['examples']}; token accuracy={r['valid_token_accuracy']:.2%}; "
          f"exact={r['exact_sequence_matches']}/{r['examples']}; "
          f"similarity={r['mean_character_similarity']:.2%}; truncated={r['exact_residual']['truncated_inputs']}")
print('No compilation: extracted methods can depend on their original class/project.')
print('Previously held-out source files and new boundaries, not an untouched final test.')
print('Reports:', out.resolve())
PY
