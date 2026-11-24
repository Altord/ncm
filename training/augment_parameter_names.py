"""Build a training-only corpus with conservative, consistent C# parameter renames."""
import argparse
import json
from pathlib import Path
from training.csharp_structure import csharp_parser, _walk
from training.data import sha256_file


def augment(text):
    raw = text.encode('utf-8')
    tree = csharp_parser().parse(raw)
    if tree.root_node.has_error: return None
    changes = []
    for method in (n for n in _walk(tree.root_node) if n.type == 'method_declaration'):
        nodes = list(_walk(method))
        if any(n.type in ('lambda_expression', 'anonymous_method_expression', 'local_function_statement') for n in nodes):
            continue
        ids = [n for n in nodes if n.type == 'identifier']
        names = [n.child_by_field_name('name') for n in nodes if n.type == 'parameter']
        for index, declaration in enumerate(n for n in names if n is not None and n.type == 'identifier'):
            original = raw[declaration.start_byte:declaration.end_byte]
            matches = [n for n in ids if raw[n.start_byte:n.end_byte] == original]
            # Multiple declarations or member/named-argument references are ambiguous without binding.
            if sum(raw[n.start_byte:n.end_byte] == original for n in names if n is not None) != 1:
                continue
            if any(n != declaration and (n.parent.type in ('parameter', 'variable_declarator', 'method_declaration', 'name_colon') or
                   (n.parent.type == 'member_access_expression' and n.parent.child_by_field_name('name') == n)) for n in matches):
                continue
            replacement = f'ncmParameter{index}'.encode()
            if any(raw[n.start_byte:n.end_byte] == replacement for n in ids): continue
            changes.extend((n.start_byte, n.end_byte, replacement) for n in matches)
    if not changes: return None
    for start, end, replacement in sorted(changes, reverse=True): raw = raw[:start] + replacement + raw[end:]
    if csharp_parser().parse(raw).root_node.has_error: return None
    return raw.decode('utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--text-key', default='text')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit(f'Output exists: {out}')
    rows = [json.loads(s) for s in Path(args.input).read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows: raise SystemExit('Input is empty')
    extra = []
    for row in rows:
        if not isinstance(row.get(args.text_key), str): raise ValueError('Missing text')
        if Path(str(row.get('source', ''))).suffix.lower() != '.cs': continue
        changed = augment(row[args.text_key])
        if changed is not None:
            candidate = dict(row); candidate[args.text_key] = changed
            extra.append(candidate)
    if len(extra) < max(1, len(rows) // 100):
        raise SystemExit(f'Only {len(extra)}/{len(rows)} records could be augmented; stopping before training')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows + extra: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps({'original_records': len(rows), 'augmented_records': len(extra),
                      'input_sha256': sha256_file(args.input), 'output_sha256': sha256_file(out),
                      'note': 'Syntax-aware spelling augmentation, not compiler binding; comments and strings are not renamed.'}, indent=2), flush=True)


if __name__ == '__main__': main()
