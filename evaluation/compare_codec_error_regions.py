"""Offline character-diff diagnostics by reference C# syntax region."""
import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from evaluation.codec_prompt_signature_copy import WRAPPER
from training.csharp_structure import csharp_parser, _walk, byte_to_char_offsets
from training.data import sha256_file


def regions(text):
    raw = (WRAPPER + text + '\n}').encode('utf-8')
    tree = csharp_parser().parse(raw)
    offset = len(WRAPPER.encode('utf-8'))
    byte_map = byte_to_char_offsets(text)
    labels = ['whitespace' if ch.isspace() else 'other_syntax' for ch in text]
    priority = {'other_syntax': 0, 'whitespace': 0, 'identifier': 1, 'literal': 2, 'comment': 3}
    for n in _walk(tree.root_node):
        kind = n.type
        if 'comment' in kind:
            label = 'comment'
        elif 'string' in kind or kind in ('character_literal', 'integer_literal', 'real_literal', 'boolean_literal', 'null_literal'):
            label = 'literal'
        elif kind in ('identifier', 'verbatim_identifier'):
            label = 'identifier'
        else:
            continue
        start = max(0, min(len(byte_map)-1, n.start_byte-offset))
        end = max(0, min(len(byte_map)-1, n.end_byte-offset))
        for i in range(byte_map[start], byte_map[end]):
            if priority[label] > priority[labels[i]]:
                labels[i] = label
    return labels, bool(tree.root_node.has_error)


def summarize(rows):
    characters, methods = Counter(), Counter()
    exact = insertions = parse_errors = 0
    for row in rows:
        target, reconstruction = row['target'], row['reconstructed']
        labels, failed = regions(target)
        parse_errors += failed
        exact += target == reconstruction
        seen = set()
        for op, start, end, _, _ in difflib.SequenceMatcher(None, target, reconstruction, autojunk=False).get_opcodes():
            if op == 'equal': continue
            if op == 'insert':
                insertions += 1
                seen.add('insertions')
            else:
                for label in labels[start:end]:
                    characters[label] += 1
                    seen.add(label)
        methods.update(seen)
    return {'examples': len(rows), 'exact': exact, 'reference_parse_errors': parse_errors,
            'changed_reference_characters_by_region': dict(characters),
            'methods_with_change_by_region': dict(methods), 'insertion_spans': insertions}


def load(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    d = {r['id']: r for r in rows}
    if not rows or len(rows) != len(d): raise ValueError(f'Empty or duplicate IDs: {path}')
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit('Output exists')
    ref, cand = load(args.reference), load(args.candidate)
    if set(ref) != set(cand) or any(ref[k]['target'] != cand[k]['target'] for k in ref):
        raise SystemExit('IDs or targets differ')
    report = {'reference': summarize(list(ref.values())), 'candidate': summarize(list(cand.values())),
              'reference_sha256': sha256_file(args.reference), 'candidate_sha256': sha256_file(args.candidate),
              'scope': 'Offline character-diff analysis using the known original as reference.',
              'limitations': 'Diff alignment is heuristic; AST labels come from target syntax, not inference. Region counts overlap across methods, and parse failures can affect labels. This does not measure semantics or generation.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== Codec error region comparison to share ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
