"""Build signature-to-method practice pairs; not natural-language instructions."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from transformers import AutoTokenizer
from training.build_reference_corpus import is_validation, source_license, SKIP_PARTS
from training.csharp_structure import csharp_parser, _walk


def normalize(text):
    return re.sub(r'\s+', '', text)


def method_pairs(path, root, parser, tokenizer, prefix_limit, target_limit, counts):
    raw = path.read_bytes()
    raw.decode('utf-8')  # Preserve exact source bytes; do not silently replace invalid text.
    tree = parser.parse(raw)
    for node in _walk(tree.root_node):
        if node.type != 'method_declaration' or node.has_error:
            continue
        body = node.child_by_field_name('body')
        name = node.child_by_field_name('name')
        if body is None or name is None:
            counts['missing_body_or_name'] += 1
            continue
        signature = raw[node.start_byte:body.start_byte].decode('utf-8').strip()
        target = raw[node.start_byte:node.end_byte].decode('utf-8').strip()
        if not signature or not target:
            continue
        prompt = 'Complete this C# method: ' + signature
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if len(prompt_ids) > prefix_limit:
            counts['long_prompt'] += 1
            continue
        # Leave room for EOS: no truncation is allowed for these examples.
        if len(target_ids) + 1 > target_limit:
            counts['long_target'] += 1
            continue
        relative = path.relative_to(root).as_posix()
        yield {'id': hashlib.sha256((relative + ':' + str(node.start_byte)).encode()).hexdigest()[:16],
               'prompt': prompt, 'target': target, 'source': relative,
               'source_line': node.start_point.row + 1,
               'license': source_license(path, root), 'task_type': 'signature_to_complete_method',
               'prompt_tokens': len(prompt_ids), 'target_tokens_without_eos': len(target_ids)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('data/reference'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--tokenizer', default='deepseek-ai/deepseek-coder-1.3b-base')
    ap.add_argument('--prefix-length', type=int, default=64)
    ap.add_argument('--target-length', type=int, default=256)
    ap.add_argument('--validation-fraction', type=float, default=0.1)
    args = ap.parse_args()
    root = args.root.resolve()
    if not root.is_dir() or not 0 < args.validation_fraction < 0.5:
        ap.error('root must exist and validation fraction must be between 0 and 0.5')
    if args.prefix_length < 1 or args.target_length < 2:
        ap.error('invalid token limits')
    if any((args.output / name).exists() for name in ('train.jsonl', 'validation.jsonl', 'manifest.json')):
        raise SystemExit(f'Output already exists under {args.output}; choose a new output path')
    parser = csharp_parser()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    protected = set()
    for name in ('test.jsonl', 'validation.jsonl', 'codec_compile_stress.jsonl'):
        path = Path('examples/godot_csharp_generator_pilot') / name
        if not path.is_file():
            raise SystemExit(f'Missing protected pilot dataset: {path}')
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if isinstance(row.get('target'), str):
                protected.add(normalize(row['target']))
    candidates = {'train': [], 'validation': []}
    counts = Counter()
    files_scanned = 0
    for path in sorted(root.rglob('*.cs')):
        if any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        try:
            files_scanned += 1
            if files_scanned % 50 == 0:
                print(f'Scanned {files_scanned} C# files; candidate methods: '
                      f"{len(candidates['train']) + len(candidates['validation'])}", flush=True)
            if path.stat().st_size > 2_000_000:
                counts['oversized_file'] += 1
                continue
            split = 'validation' if is_validation(path, root, args.validation_fraction) else 'train'
            candidates[split].extend(method_pairs(path, root, parser, tokenizer,
                                                  args.prefix_length, args.target_length, counts))
        except (OSError, UnicodeDecodeError):
            counts['unreadable_files'] += 1
    # Deduplicate across splits; prefer validation for any repeated method body.
    seen = set(protected)
    rows = {}
    for split in ('validation', 'train'):
        rows[split] = []
        for row in candidates[split]:
            key = normalize(row['target'])
            if key in seen:
                counts['duplicate_or_protected'] += 1
                continue
            seen.add(key)
            rows[split].append(row)
    if not rows['train'] or not rows['validation']:
        raise SystemExit('No train or validation examples survived; nothing written')
    args.output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation'):
        with (args.output / f'{split}.jsonl').open('x', encoding='utf-8') as f:
            for row in rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'task': 'signature_to_complete_method', 'root': str(root),
                'tokenizer': args.tokenizer, 'prefix_length': args.prefix_length,
                'target_length': args.target_length,
                'split': 'stable SHA-256 of relative file path; deduplicate validation before train',
                'source_manifest': str((root / 'SOURCES.tsv').resolve()),
                'files_scanned': files_scanned,
                'counts': {k: len(v) for k, v in rows.items()}, 'excluded': dict(counts),
                'limitations': 'Synthetic signature completion, not natural-language instruction training; '
                               'some methods depend on surrounding class/project. '
                               'Exact normalized duplicate protection is not semantic or near-duplicate detection.'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
