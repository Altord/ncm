"""Measure frozen codec quality and packed size when stopping at earlier VQ levels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, iter_records, load_base_tokenizer
from evaluation.godot_csharp_codec_eval import exact_residual, _varint, visible_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--text-key', default='text')
    p.add_argument('--code-only', action='store_true')
    p.add_argument('--levels', default='4,8,16,32,64')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tokenizer = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    model = VQVAE(**OmegaConf.create(vqcfg))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    quant = model.vq
    requested = sorted(set(int(x) for x in args.levels.split(',')))
    if not requested or requested[0] < 1 or requested[-1] > len(quant.levels):
        raise ValueError('Requested level count is outside the trained codec')
    if args.batch_size < 1:
        raise ValueError('batch size must be positive')
    records = []
    for row in iter_records(Path(args.input), 'jsonl'):
        if args.code_only and Path(str(row.get('source', ''))).suffix.lower() not in ('.cs', '.csproj'):
            continue
        text = row.get(args.text_key)
        if not isinstance(text, str):
            raise ValueError('Missing text field')
        records.append(text)
    if not records:
        raise ValueError('No matching input records')
    counts = {n: {'correct': 0, 'tokens': 0, 'exact': 0, 'vq_bytes': 0,
                  'residual_bytes': 0, 'framing_bytes': 0} for n in requested}
    source_bytes = sum(len(text.encode('utf-8')) for text in records)
    context = int(cfg.model.context_length)
    pad, eos = int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)
    truncated = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(records), args.batch_size):
            chunk = records[begin:begin + args.batch_size]
            encoded = [fixed_sequence(encode_text(tokenizer, text, append_eos=True), context, pad, eos) for text in chunk]
            padded, masks = zip(*encoded)
            lengths = [sum(mask) for mask in masks]
            truths = [list(row[:length]) for row, length in zip(padded, lengths)]
            truncated += sum(visible_text(tokenizer, truth.copy(), eos, pad) != text for truth, text in zip(truths, chunk))
            codes = model.encode(torch.tensor(padded, dtype=torch.long, device=device))
            z = torch.zeros(len(chunk), quant.levels[-1], quant.embed_dim, device=device)
            offset = bits = 0
            for i, level in enumerate(quant.levels):
                q = quant.embeddings[i](codes[:, offset:offset + level])
                if i != len(quant.levels) - 1:
                    q = quant._up_to_full(q, i, quant.levels[-1])
                z = z + quant._apply_quant_resi(q, i)
                offset += level
                bits += level * max(1, math.ceil(math.log2(int(quant.vqvae_vocab_sizes[i]))))
                n = i + 1
                if n not in counts:
                    continue
                predictions = model.decoder(z).argmax(dim=-1).tolist()
                s = counts[n]
                for truth, pred, length in zip(truths, predictions, lengths):
                    pred = pred[:length]
                    payload = exact_residual(truth, pred)
                    s['tokens'] += length
                    s['correct'] += sum(a == b for a, b in zip(truth, pred))
                    s['exact'] += truth == pred
                    s['vq_bytes'] += (bits + 7) // 8
                    s['residual_bytes'] += len(payload)
                    s['framing_bytes'] += len(_varint(length)) + len(_varint(len(payload)))
            print(f'Processed {min(begin + len(chunk), len(records))}/{len(records)} rows', flush=True)
    print('\n=== Level prefix scan ===')
    for n in requested:
        s = counts[n]
        s['total_bytes'] = s['vq_bytes'] + s['residual_bytes'] + s['framing_bytes']
        s['total_to_input_ratio'] = s['total_bytes'] / max(1, source_bytes)
        print(f"{n:>2} levels: accuracy={s['correct']/max(1,s['tokens']):.2%}, exact={s['exact']}/{len(records)}, VQ={s['vq_bytes']} B, residual={s['residual_bytes']} B, total/input={s['total_to_input_ratio']:.2f}x")
    report = {'input': args.input, 'checkpoint': args.checkpoint, 'examples': len(records),
              'input_utf8_bytes': source_bytes, 'inputs_not_exactly_represented': truncated,
              'levels': counts, 'elapsed_seconds': time.perf_counter() - start,
              'note': 'Frozen checkpoint prefixes; exact residual requires the original tokens. Ratios exclude model/config and container metadata; no generator was trained.'}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {output.resolve()}')
    print(f'Inputs not exactly represented by tokenization/chunking: {truncated}')


if __name__ == '__main__':
    main()
