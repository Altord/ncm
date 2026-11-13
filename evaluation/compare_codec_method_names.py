"""Compare method declaration names in saved codec reconstructions, without rerunning models."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.compare_codec_pairs import load
from training.csharp_structure import csharp_parser, _walk


def method_name(text):
    # Saved rows are method snippets, not full class files.
    source = ('class __NcmDiagnostic {\n' + text + '\n}').encode('utf-8')
    tree = csharp_parser().parse(source)
    names = []
    for node in _walk(tree.root_node):
        if node.type == 'method_declaration':
            name = node.child_by_field_name('name')
            if name is not None:
                names.append(source[name.start_byte:name.end_byte].decode('utf-8'))
    return names[0] if len(names) == 1 else None


def compare(reference, candidate):
    ref, rr = load(reference)
    new, cr = load(candidate)
    for field in ('input_sha256', 'text_key', 'sequence_length', 'code_only', 'code_extensions'):
        if field not in rr or rr[field] != cr.get(field):
            raise ValueError(f'Reports do not match: {field}')
    if ref.keys() != new.keys():
        raise ValueError('Example IDs differ')
    rows = []
    for key, a in ref.items():
        b = new[key]
        if a['target'] != b['target']:
            raise ValueError(f'Target differs: {key}')
        expected = method_name(a['target'])
        if expected is None:
            continue
        first, second = method_name(a['reconstructed']), method_name(b['reconstructed'])
        rows.append((key, expected, first, second))
    return rows, len(ref)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    args = p.parse_args()
    rows, total = compare(args.reference, args.candidate)
    if not rows:
        raise SystemExit('No targets with exactly one recognized method declaration')
    first = sum(a == target for _, target, a, b in rows)
    second = sum(b == target for _, target, a, b in rows)
    print(f'Targets with exactly one recognized method: {len(rows)}/{total}; others excluded')
    print(f'Method-name exact: reference={first}/{len(rows)}; candidate={second}/{len(rows)}')
    print(f'Method name unrecognized/ambiguous: reference={sum(a is None for _,_,a,_ in rows)}; '
          f'candidate={sum(b is None for _,_,_,b in rows)} (counted as not exact)')
    print(f'Correct only reference={sum(a == t and b != t for _,t,a,b in rows)}; '
          f'only candidate={sum(a != t and b == t for _,t,a,b in rows)}')
    print('Changed method-name outcomes:')
    for key, target, a, b in rows:
        if (a == target) != (b == target):
            print(f'  {key}: expected={target!r}, reference={a!r}, candidate={b!r}')
    print('AST name extraction can fail on malformed reconstructions; this is not an API or semantic correctness test.')


if __name__ == '__main__':
    main()
