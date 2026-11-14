"""Measure pointer-copy opportunities from nearby project code with target method removed.

Uses known targets only for locating/removing the method and offline scoring.
This is a token-ID upper bound for the current pointer, not a generator test.
"""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vqvae import VQVAE
from training.csharp_structure import csharp_parser, _walk
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file


@lru_cache(maxsize=64)
def parse_source(path):
    raw = Path(path).read_bytes()
    if len(raw) > 2_000_000:
        raise ValueError('source_too_large')
    raw.decode('utf-8')
    tree = csharp_parser().parse(raw)
    return raw, tree


def augment_record(row, root, tokenizer, max_tokens=256, window=600):
    """Supply nearby file text after removing the entire target method node."""
    original_prompt = row['prompt']
    original_ids = encode_text(tokenizer, original_prompt, False)
    if not original_ids or len(original_ids) > max_tokens:
        raise ValueError('Original prompt is empty or exceeds context budget')
    source = row.get('source')
    line = row.get('source_line')
    if not source or not isinstance(line, int):
        return row.copy(), 'missing_source_or_line'
    root = Path(root).resolve()
    path = (root / source).resolve()
    if not path.is_relative_to(root):
        return row.copy(), 'source_outside_root'
    try:
        raw, tree = parse_source(str(path))
    except (OSError, UnicodeError, ValueError):
        return row.copy(), 'source_unavailable'
    target = row['target'].strip()
    matches = [n for n in _walk(tree.root_node)
               if n.type == 'method_declaration' and n.start_point.row + 1 == line
               and raw[n.start_byte:n.end_byte].decode('utf-8').strip() == target]
    if len(matches) != 1:
        return row.copy(), 'method_not_uniquely_located'
    node = matches[0]
    left = raw[:node.start_byte].decode('utf-8')
    right = raw[node.end_byte:].decode('utf-8')
    # Avoid including a verbatim duplicate of the target elsewhere in the file.
    if target and target in left + right:
        return row.copy(), 'verbatim_target_elsewhere'
    size = int(window)
    while size > 0:
        context = left[-size:] + '\n/* target method omitted */\n' + right[:size]
        if target and target in context:
            return row.copy(), 'verbatim_target_in_context'
        prompt = original_prompt + '\nNearby project code (target method omitted):\n' + context
        if len(encode_text(tokenizer, prompt, False)) <= max_tokens:
            augmented = row.copy()
            augmented['prompt'] = prompt
            return augmented, 'context_added'
        size //= 2
    return row.copy(), 'no_context_budget'


def token_coverage(ids, mask, guessed, original_prompt, context_prompt):
    original_prompt, context_prompt = set(original_prompt), set(context_prompt)
    wrong = [truth for truth, guess, valid in zip(ids, guessed, mask) if valid and truth != guess]
    return {'valid_tokens': sum(mask), 'baseline_errors': len(wrong),
            'prompt_copyable': sum(token in original_prompt for token in wrong),
            'context_copyable': sum(token in context_prompt for token in wrong),
            'newly_copyable': sum(token not in original_prompt and token in context_prompt for token in wrong),
            'no_longer_copyable': sum(token in original_prompt and token not in context_prompt for token in wrong)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--source-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--context-tokens', type=int, default=256)
    args = p.parse_args()
    if min(args.batch_size, args.context_tokens) < 1:
        p.error('batch size and context-token budget must be positive')
    if not Path(args.source_root).is_dir():
        raise SystemExit(f'Source root is unavailable: {args.source_root}')
    out = Path(args.output)
    if any((out / name).exists() for name in ('report.json', 'augmented.jsonl', 'results.jsonl')):
        raise SystemExit('Output exists; choose a new directory')
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    records = [json.loads(line) for line in Path(args.input).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records or len({r['id'] for r in records}) != len(records):
        raise SystemExit('Empty input or duplicate IDs')
    start = time.perf_counter()
    augmented, reasons = [], Counter()
    for i, row in enumerate(records, 1):
        item, reason = augment_record(row, args.source_root, tok, args.context_tokens)
        augmented.append(item)
        reasons[reason] += 1
        if i % 64 == 0 or i == len(records):
            print(f'Prepared context {i}/{len(records)}; added={reasons["context_added"]}', flush=True)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    if vqcfg.get('autoregressive_decoder', False):
        raise SystemExit('Expected parallel codec')
    codec = VQVAE(**OmegaConf.create(vqcfg))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    codec.load_state_dict(checkpoint['model'], strict=True)
    codec.to(device).eval().requires_grad_(False)
    context = int(cfg.model.context_length)
    pad, eos = int(tok.pad_token_id), int(tok.eos_token_id)
    totals = Counter()
    details = []
    with torch.inference_mode():
        for start_idx in range(0, len(records), args.batch_size):
            originals = records[start_idx:start_idx + args.batch_size]
            new = augmented[start_idx:start_idx + args.batch_size]
            xs, masks = [], []
            for row in originals:
                ids = encode_text(tok, row['target'], True)
                if len(ids) > context:
                    raise SystemExit(f'Target exceeds codec context: {row["id"]}')
                padded, mask = fixed_sequence(ids, context, pad, eos)
                xs.append(padded)
                masks.append(mask)
            x = torch.tensor(xs, device=device, dtype=torch.long)
            pred = codec.decode(codec.encode(x)).argmax(dim=-1).cpu().tolist()
            for original, item, ids, mask, guessed in zip(originals, new, xs, masks, pred):
                stats = token_coverage(ids, mask, guessed,
                    encode_text(tok, original['prompt'], False), encode_text(tok, item['prompt'], False))
                totals.update(stats)
                details.append({'id': original['id'], **stats})
            print(f'Measured {min(start_idx + len(originals), len(records))}/{len(records)}', flush=True)
    valid, errors = totals['valid_tokens'], totals['baseline_errors']
    report = {'examples': len(records), 'context_status': dict(reasons),
              'input_sha256': sha256_file(args.input), 'codec_sha256': sha256_file(args.checkpoint),
              'context_token_budget': args.context_tokens, **totals,
              'baseline_token_accuracy': (valid-errors) / valid,
              'ideal_prompt_pointer_token_accuracy': (valid-errors+totals['prompt_copyable']) / valid,
              'ideal_context_pointer_token_accuracy': (valid-errors+totals['context_copyable']) / valid,
              'elapsed_seconds': time.perf_counter()-start,
              'interpretation': 'Optimistic token-ID availability ceiling for the existing pointer mixture only: assumes every copyable error is corrected and no regressions. Availability is not proof a learned pointer will select the correct occurrence or reconstruct a full identifier.',
              'limitations': 'Known-target codec latents. Context is nearby text from the same source file after target method removal, not an independent generator test. Exact target duplicates elsewhere in the file are excluded, but near duplicates and other forms of leakage are not ruled out. Source path names are locally available editing-location assumptions.'}
    with (out / 'augmented.jsonl').open('x', encoding='utf-8') as f:
        for row in augmented:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (out / 'results.jsonl').open('x', encoding='utf-8') as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Project context copyability ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
