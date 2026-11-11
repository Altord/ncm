"""Compare saved exact codec measurements with raw and zlib storage per chunk."""
import argparse
import json
from pathlib import Path
import zlib


def varint_size(value):
    count = 1
    while value >= 128:
        value >>= 7
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, metavar='NAME=JSONL')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reports = {}
    baseline = None
    for spec in args.input:
        name, sep, path = spec.partition('=')
        if not sep or not name or name in reports:
            parser.error('Each --input needs a unique NAME=JSONL')
        with Path(path).open(encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        texts = [row['target'] for row in rows]
        if not rows or (baseline is not None and texts != baseline):
            raise ValueError(f'{name}: inputs are empty or differ from the first dataset')
        baseline = texts
        raw = compressed = codec = codec_bytes = corrections = 0
        fallback = 0
        short = 0
        for row, text in zip(rows, texts):
            data = text.encode('utf-8')
            blob = zlib.compress(data, level=6)
            if zlib.decompress(blob) != data:
                raise AssertionError('zlib round trip failed')
            raw += len(data)
            compressed += 1 + varint_size(len(blob)) + len(blob)
            measurement = row['packed_measurement']
            residual = row['exact_residual']
            if not residual['restored_text_exact']:
                raise ValueError(f'{name}: input was truncated or not exactly recovered')
            current = measurement['total_bytes_with_residual']
            codec += current
            codec_bytes += measurement['vq_code_bytes']
            corrections += residual['binary_bytes']
            fallback += 1 + min(current, len(data) + varint_size(len(data)), len(blob) + varint_size(len(blob)))
            short += row['valid_token_count'] <= 64
        reports[name] = {'examples': len(rows), 'raw_utf8_bytes': raw, 'zlib_framed_bytes': compressed,
                         'vq_bytes': codec_bytes, 'correction_bytes': corrections,
                         'codec_and_correction_bytes': codec,
                         'best_of_three_framed_bytes': fallback, 'rows_at_most_64_tokens': short}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'results': reports, 'scope': 'Known-input lossless storage diagnostic. Zlib is per chunk; VQ model and shared configuration are excluded. Best-of-three adds one tag byte per chunk. Does not evaluate generation.'}, indent=2) + '\n', encoding='utf-8')
    print('\n=== Codec storage comparison ===')
    for name, row in reports.items():
        raw = row['raw_utf8_bytes']
        print(f"{name}: rows={row['examples']}, raw={raw} B, zlib={row['zlib_framed_bytes']} B ({row['zlib_framed_bytes']/raw:.2f}x), VQ+corrections={row['codec_and_correction_bytes']} B ({row['codec_and_correction_bytes']/raw:.2f}x), best-of-three={row['best_of_three_framed_bytes']} B ({row['best_of_three_framed_bytes']/raw:.2f}x)")
        print(f"  VQ indices={row['vq_bytes']} B; corrections={row['correction_bytes']} B; rows <=64 tokens={row['rows_at_most_64_tokens']}")
    print('Known input only; codec model/configuration excluded; zlib stored independently per chunk.')
    print(f'Report: {output.resolve()}')


if __name__ == '__main__':
    main()
