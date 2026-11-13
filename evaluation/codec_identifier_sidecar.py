"""Identifier-only, known-input codec correction experiment using saved reconstructions.

The encoder sees the original text and decoder output, selects edits inside AST
identifier spans, and serializes their positions and replacement bytes. The
decoder receives only the reconstruction and sidecar. This is a source-derived
correction stream, not a learned pointer or an inference path for unknown code.
"""
from __future__ import annotations
import argparse
import base64
import difflib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.godot_csharp_codec_eval import _varint, _read_varint
from evaluation.godot_csharp_generator_pilot import compile_candidate
from training.csharp_structure import csharp_parser, _walk, byte_to_char_offsets


def identifier_mask(text):
    prefix = 'class __NcmDiagnostic {\n'
    full = prefix + text + '\n}'
    data = full.encode('utf-8')
    mapping = byte_to_char_offsets(full)
    tree = csharp_parser().parse(data)
    mask = bytearray(len(text))
    for node in _walk(tree.root_node):
        if node.type not in ('identifier', 'verbatim_identifier'):
            continue
        start = max(0, mapping[node.start_byte] - len(prefix))
        end = min(len(text), mapping[node.end_byte] - len(prefix))
        if start < end:
            mask[start:end] = b'\x01' * (end - start)
    return mask


def encode_sidecar(source, decoded):
    mask = identifier_mask(source)
    edits = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, decoded, source, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        if j1 == j2:
            # Remove identifier-like extra decoded characters only when the
            # insertion point borders a known source identifier.
            adjacent = (j1 > 0 and mask[j1 - 1]) or (j1 < len(mask) and mask[j1])
            deleted = decoded[i1:i2]
            if not adjacent or not deleted or not all(c.isalnum() or c == '_' for c in deleted):
                continue
        elif not all(mask[j1:j2]):
            continue
        edits.append((i1, i2 - i1, source[j1:j2].encode('utf-8')))
    out = bytearray(_varint(len(edits)))
    previous_end = 0
    for start, length, value in edits:
        if start < previous_end:
            raise ValueError('Overlapping edits')
        out.extend(_varint(start - previous_end))
        out.extend(_varint(length))
        out.extend(_varint(len(value)))
        out.extend(value)
        previous_end = start + length
    return bytes(out), len(edits)


def apply_sidecar(decoded, payload):
    count, offset = _read_varint(payload, 0)
    previous_end = 0
    output = []
    for _ in range(count):
        delta, offset = _read_varint(payload, offset)
        length, offset = _read_varint(payload, offset)
        byte_length, offset = _read_varint(payload, offset)
        start = previous_end + delta
        end = start + length
        if start > len(decoded) or end > len(decoded) or offset + byte_length > len(payload):
            raise ValueError('Invalid identifier sidecar')
        output.append(decoded[previous_end:start])
        output.append(payload[offset:offset + byte_length].decode('utf-8'))
        offset += byte_length
        previous_end = end
    if offset != len(payload):
        raise ValueError('Trailing sidecar bytes')
    return ''.join(output) + decoded[previous_end:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Existing codec reconstruction JSONL')
    p.add_argument('--output', required=True)
    p.add_argument('--vq-bytes-per-record', type=int, default=None,
                   help='Measured fixed VQ bytes per record, if known; excludes model and other framing')
    p.add_argument('--compile-changed', action='store_true',
                   help='Compile changed texts; reuse saved compile results for unchanged texts')
    p.add_argument('--framework', default='net8.0')
    args = p.parse_args()
    if args.vq_bytes_per_record is not None and args.vq_bytes_per_record < 0:
        p.error('VQ bytes must not be negative')
    inp, out = Path(args.input), Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    rows = [json.loads(line) for line in inp.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise SystemExit('Input must contain unique, nonempty rows')
    results = []
    source_bytes = sidecar_bytes = edits_total = 0
    compiler_calls = 0
    encode_seconds = decode_seconds = 0.0
    for index, row in enumerate(rows, 1):
        source, decoded = row['target'], row['reconstructed']
        start = time.perf_counter()
        payload, edits = encode_sidecar(source, decoded)
        encode_seconds += time.perf_counter() - start
        start = time.perf_counter()
        restored = apply_sidecar(decoded, payload)
        decode_seconds += time.perf_counter() - start
        sidecar_bytes += len(payload)
        source_bytes += len(source.encode('utf-8'))
        edits_total += edits
        before_status = after_status = None
        if args.compile_changed:
            before_status = row.get('reconstruction_compile', {}).get('status')
            if before_status not in ('passed', 'failed', 'empty'):
                raise SystemExit(f'Saved compiler result unavailable for {row["id"]}')
            after_status = before_status
            if restored != decoded:
                compilation = compile_candidate(restored, args.framework)
                compiler_calls += 1
                after_status = compilation['status']
                if after_status not in ('passed', 'failed', 'empty'):
                    raise SystemExit(f'Compiler unavailable for {row["id"]}: {compilation}')
        results.append({'id': row['id'], 'before_exact': decoded == source,
                        'after_exact': restored == source, 'sidecar_bytes': len(payload),
                        'sidecar_base64': base64.b64encode(payload).decode('ascii'),
                        'before_character_similarity': difflib.SequenceMatcher(None, source, decoded, autojunk=False).ratio(),
                        'after_character_similarity': difflib.SequenceMatcher(None, source, restored, autojunk=False).ratio(),
                        'identifier_edits': edits, 'restored': restored,
                        'before_compile': before_status, 'after_compile': after_status})
        if index % 16 == 0 or index == len(rows):
            print(f'Processed {index}/{len(rows)}; changed compilations={compiler_calls}', flush=True)
    vq_bytes = None if args.vq_bytes_per_record is None else args.vq_bytes_per_record * len(rows)
    report = {'examples': len(rows), 'before_exact': sum(r['before_exact'] for r in results),
              'after_exact': sum(r['after_exact'] for r in results),
              'before_mean_character_similarity': sum(r['before_character_similarity'] for r in results) / len(results),
              'after_mean_character_similarity': sum(r['after_character_similarity'] for r in results) / len(results),
              'identifier_edit_count': edits_total, 'sidecar_bytes': sidecar_bytes,
              'input_utf8_bytes': source_bytes, 'vq_bytes': vq_bytes,
              'vq_plus_sidecar_bytes': None if vq_bytes is None else vq_bytes + sidecar_bytes,
              'vq_plus_sidecar_to_input': None if vq_bytes is None else (vq_bytes + sidecar_bytes) / max(1, source_bytes),
              'sidecar_encode_seconds': encode_seconds, 'sidecar_apply_seconds': decode_seconds,
              'compiles_before': sum(r['before_compile'] == 'passed' for r in results) if args.compile_changed else None,
              'compiles_after': sum(r['after_compile'] == 'passed' for r in results) if args.compile_changed else None,
              'new_compiler_calls': compiler_calls,
              'scope': 'Known source round trip only; source-derived identifier edits plus character positions are sent in sidecar. Not a learned pointer or a generator. Original targets used for encoding and offline scoring, not sidecar application.',
              'size_scope': 'Actual identifier sidecar bytes; optional externally supplied measured VQ bytes. Excludes VQ metadata, model/config, and container framing. Does not include remaining nonidentifier errors.',
              'timing_scope': 'Saved reconstruction experiment; sidecar encoding and application only, excludes codec, packaging and compile.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Identifier sidecar codec experiment ===')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
