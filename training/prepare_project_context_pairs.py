"""Prepare target-removed nearby project context for pointer training."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from evaluation.measure_project_context_copyability import augment_record
from training.data import load_base_tokenizer, sha256_file


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Empty input or duplicate IDs: {path}')
    return rows


def prepare_rows(rows, source_root, tokenizer, budget, *, progress=False):
    output, counts = [], Counter()
    for index, row in enumerate(rows, 1):
        augmented, reason = augment_record(row, source_root, tokenizer, budget)
        if augmented.get('target') != row.get('target') or augmented.get('source') != row.get('source'):
            raise ValueError(f'Target or source changed: {row["id"]}')
        if not augmented['prompt'].startswith(row['prompt']):
            raise ValueError(f'Original prompt not preserved: {row["id"]}')
        counts[reason] += 1
        output.append(augmented)
        if progress and (index % 256 == 0 or index == len(rows)):
            print(f'Prepared {index}/{len(rows)}; context added={counts["context_added"]}', flush=True)
    return output, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--train', required=True)
    p.add_argument('--validation-original', required=True)
    p.add_argument('--validation-augmented', required=True)
    p.add_argument('--coverage-report', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--context-tokens', type=int, default=256)
    args = p.parse_args()
    if args.context_tokens < 1:
        p.error('context-tokens must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.manifest.json').exists():
        raise SystemExit('Output exists; refusing to overwrite')
    if not Path(args.source_root).is_dir():
        raise SystemExit('Source root is unavailable')
    report = json.loads(Path(args.coverage_report).read_text(encoding='utf-8'))
    if (report['input_sha256'] != sha256_file(args.validation_original) or
        report['codec_sha256'] != sha256_file(args.checkpoint) or
        report['context_token_budget'] != args.context_tokens):
        raise SystemExit('Coverage report does not match input, codec or context budget')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    original_val, saved_val = load_rows(args.validation_original), load_rows(args.validation_augmented)
    reproduced_val, val_counts = prepare_rows(original_val, args.source_root, tok, args.context_tokens)
    if reproduced_val != saved_val or dict(val_counts) != report['context_status']:
        raise SystemExit('Validation context differs from coverage run; source data may have changed')
    train = load_rows(args.train)
    if any(not r.get('source') for r in train + original_val):
        raise SystemExit('Training or validation rows lack source paths')
    overlap = {r.get('source') for r in train} & {r.get('source') for r in original_val}
    if overlap:
        raise SystemExit(f'Training and validation source paths overlap: {sorted(overlap)[:5]}')
    normalize = lambda text: re.sub(r'\s+', '', text)
    duplicates = {normalize(r['target']) for r in train} & {normalize(r['target']) for r in original_val}
    if duplicates:
        raise SystemExit(f'Training and validation contain {len(duplicates)} whitespace-normalized duplicate methods')
    augmented_train, counts = prepare_rows(train, args.source_root, tok, args.context_tokens, progress=True)
    if not counts['context_added']:
        raise SystemExit('No training context was added; stopping')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    if tmp.exists():
        raise SystemExit(f'Temporary output already exists: {tmp}')
    with tmp.open('x', encoding='utf-8') as f:
        for row in augmented_train:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    tmp.replace(out)
    manifest = {'examples': len(train), 'train_sha256': sha256_file(args.train),
                'augmented_train_sha256': sha256_file(out), 'context_status': dict(counts),
                'validation_examples': len(original_val), 'validation_context_status': dict(val_counts),
                'validation_augmented_sha256': sha256_file(args.validation_augmented),
                'context_tokens': args.context_tokens,
                'limitations': 'Target method removed, but other nearby project text may include partial or near-duplicate information. Known-target VQ codec evaluation, not a generator test.'}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('\n=== Project-context training dataset ===\n' + json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
