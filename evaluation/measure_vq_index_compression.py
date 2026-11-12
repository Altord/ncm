"""Measure actual per-record lossless compression of a frozen codec's indices."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import encode_text, fixed_sequence, iter_records, load_base_tokenizer
from evaluation.godot_csharp_codec_eval import pack_codes, unpack_codes, _varint, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--text-key', default='text')
    p.add_argument('--code-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tokenizer = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    model = VQVAE(**OmegaConf.create(vqcfg))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    widths = [max(1, math.ceil(math.log2(int(vocab))))
              for length, vocab in zip(model.vq.levels, model.vq.vqvae_vocab_sizes)
              for _ in range(int(length))]
    records = []
    for row in iter_records(Path(args.input), 'jsonl'):
        if args.code_only and Path(str(row.get('source', ''))).suffix.lower() not in ('.cs', '.csproj'):
            continue
        value = row.get(args.text_key)
        if not isinstance(value, str):
            raise ValueError(f'Missing text field {args.text_key!r}')
        records.append(value)
    if not records:
        raise ValueError('No matching input records')
    totals = dict(input_bytes=0, raw_zlib_bytes=0, fixed_width_bytes=0,
                  compressed_index_payload_bytes=0, compressed_index_framing_bytes=0,
                  truncated_rows=0)
    compress_seconds = decompress_seconds = 0.0
    pad, eos = int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)
    context = int(cfg.model.context_length)
    with torch.inference_mode():
        for begin in range(0, len(records), args.batch_size):
            chunk = records[begin:begin + args.batch_size]
            padded = []
            for text in chunk:
                ids = encode_text(tokenizer, text, append_eos=True)
                totals['truncated_rows'] += int(len(ids) > context)
                padded.append(fixed_sequence(ids, context, pad, eos)[0])
                data = text.encode('utf-8')
                totals['input_bytes'] += len(data)
                totals['raw_zlib_bytes'] += len(zlib.compress(data))
            rows = model.encode(torch.tensor(padded, device=device, dtype=torch.long)).cpu().tolist()
            for indices in rows:
                fixed = pack_codes(indices, widths)
                start = time.perf_counter()
                compressed = zlib.compress(fixed)
                compress_seconds += time.perf_counter() - start
                start = time.perf_counter()
                recovered = zlib.decompress(compressed)
                decompress_seconds += time.perf_counter() - start
                if recovered != fixed or unpack_codes(recovered, widths) != indices:
                    raise AssertionError('Compressed index round trip failed')
                totals['fixed_width_bytes'] += len(fixed)
                totals['compressed_index_payload_bytes'] += len(compressed)
                totals['compressed_index_framing_bytes'] += len(_varint(len(compressed)))
            print(f'Processed {min(begin + len(chunk), len(records))}/{len(records)} records', flush=True)
    total = totals['compressed_index_payload_bytes'] + totals['compressed_index_framing_bytes']
    report = dict(input=args.input, input_sha256=file_sha256(Path(args.input)),
                  checkpoint=args.checkpoint, examples=len(records), indices_per_chunk=len(widths),
                  device=str(device), **totals, compressed_index_total_bytes=total,
                  compressed_to_fixed_ratio=total / totals['fixed_width_bytes'],
                  compressed_index_to_input_ratio=total / max(1, totals['input_bytes']),
                  compress_seconds=compress_seconds, decompress_seconds=decompress_seconds,
                  note='Zlib per packed chunk, including compressed length framing; verified indices. '
                       'Excludes original-text corrections, shared model/config, and other container metadata. '
                       'Does not reduce generator sequence length. Source zlib has no extra record framing.')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Frozen VQ index compression ===')
    print(f"Records: {len(records)}, indices/chunk: {len(widths)}, truncated: {totals['truncated_rows']}")
    print(f"Fixed VQ: {totals['fixed_width_bytes']} B; zlib VQ including framing: {total} B "
          f"({report['compressed_to_fixed_ratio']:.2f}x fixed VQ)")
    print(f"Input: {totals['input_bytes']} B; independently zlib-compressed input: {totals['raw_zlib_bytes']} B; "
          f"compressed VQ/input: {report['compressed_index_to_input_ratio']:.2f}x")
    print(f"CPU index compression: {compress_seconds:.3f}s; decompression: {decompress_seconds:.3f}s")
    print(report['note'])
    print(f'Report: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
