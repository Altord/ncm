"""Copy an explicitly provided method signature into a codec reconstruction.

This is a deterministic prompt-conditioned diagnostic, not a trained pointer.
The prompt and decoded text alone determine the result; targets score it offline.
"""
from __future__ import annotations
import argparse
import difflib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.csharp_structure import csharp_parser, _walk

PROMPT_PREFIX = 'Complete this C# method: '
WRAPPER = 'class __NcmDiagnostic {\n'


def parse_method(text, *, require_clean=False):
    source = (WRAPPER + text + '\n}').encode('utf-8')
    tree = csharp_parser().parse(source)
    if require_clean and tree.root_node.has_error:
        return None
    methods = [n for n in _walk(tree.root_node) if n.type == 'method_declaration']
    if len(methods) != 1:
        return None
    method = methods[0]
    body = method.child_by_field_name('body')
    name = method.child_by_field_name('name')
    prefix = len(WRAPPER.encode('utf-8'))
    if body is None or name is None or method.start_byte < prefix:
        return None
    original_size = len(text.encode('utf-8'))
    if body.start_byte - prefix > original_size:
        return None
    return {'start': method.start_byte - prefix, 'body': body.start_byte - prefix,
            'name': source[name.start_byte:name.end_byte].decode('utf-8')}


def fallback_body_offset(text):
    """Find an unambiguous top-level method body boundary or abstain.

    This deliberately rejects comments, strings, directives and unmatched
    delimiters instead of pretending to implement a full C# lexer.
    """
    if any(s in text for s in ('//', '/*', '"', "'", '#')):
        return None
    stack = []
    seen_parameters = False
    for i, char in enumerate(text):
        if char in '([':
            stack.append(char)
        elif char in ')]':
            wanted = '(' if char == ')' else '['
            if not stack or stack.pop() != wanted:
                return None
            if char == ')' and not stack:
                seen_parameters = True
        elif not stack and seen_parameters and (char == '{' or text[i:i+2] == '=>'):
            return i
        elif not stack and char in ';}':
            return None
    return None


def copy_signature(prompt, decoded, *, fallback=False):
    if not isinstance(prompt, str) or not prompt.startswith(PROMPT_PREFIX):
        return decoded, 'prompt_not_signature'
    signature = prompt[len(PROMPT_PREFIX):].strip()
    # Require a single syntactically valid declaration in the user-provided prompt.
    # Its original body is never accessed for candidate construction.
    prompt_parsed = parse_method(signature + ' {}', require_clean=True)
    decoded_parsed = parse_method(decoded)
    if not signature or prompt_parsed is None:
        return decoded, 'invalid_prompt_signature'
    if decoded_parsed is None:
        offset = fallback_body_offset(decoded) if fallback else None
        if offset is None:
            return decoded, 'decoded_method_unavailable'
        header = decoded[:offset]
        trailing = header[len(header.rstrip()):]
        restored = signature + trailing + decoded[offset:]
        return restored, 'fallback_signature_copied'
    raw = decoded.encode('utf-8')
    start, body = decoded_parsed['start'], decoded_parsed['body']
    header = raw[start:body]
    trailing = header[len(header.rstrip()):]
    result = raw[:start] + signature.encode('utf-8') + trailing + raw[body:]
    restored = result.decode('utf-8')
    return restored, 'unchanged' if restored == decoded else 'signature_copied'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='Saved codec reconstruction JSONL with prompts')
    p.add_argument('--output', required=True)
    p.add_argument('--fallback', action='store_true',
                   help='Try conservative body-boundary detection on malformed decoded methods')
    args = p.parse_args()
    inp, out = Path(args.input), Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists():
        raise SystemExit('Output exists; choose a new path')
    rows = [json.loads(line) for line in inp.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise SystemExit('Input must contain unique, nonempty rows')
    results = []
    start = time.perf_counter()
    for row in rows:
        if not isinstance(row.get('prompt'), str) or not isinstance(row.get('target'), str):
            raise ValueError('Rows must contain prompt and target strings')
        decoded = row['reconstructed']
        # Candidate creation uses no target or compiler result.
        selected, reason = copy_signature(row['prompt'], decoded, fallback=args.fallback)
        target = row['target']  # Offline scoring only.
        target_method = parse_method(target)
        decoded_method = parse_method(decoded)
        selected_method = parse_method(selected)
        expected_name = target_method['name'] if target_method else None
        before_name = decoded_method['name'] if decoded_method else None
        after_name = selected_method['name'] if selected_method else None
        results.append({'id': row['id'], 'selection': reason, 'selected': selected,
                        'before_exact': decoded == target, 'after_exact': selected == target,
                        'name_scorable': expected_name is not None,
                        'before_name_exact': expected_name is not None and before_name == expected_name,
                        'after_name_exact': expected_name is not None and after_name == expected_name,
                        'before_similarity': difflib.SequenceMatcher(None, target, decoded, autojunk=False).ratio(),
                        'after_similarity': difflib.SequenceMatcher(None, target, selected, autojunk=False).ratio()})
    counts = {key: sum(r['selection'] == key for r in results) for key in (
        'signature_copied', 'fallback_signature_copied', 'unchanged', 'prompt_not_signature', 'invalid_prompt_signature', 'decoded_method_unavailable')}
    report = {'examples': len(results), 'before_exact_text': sum(r['before_exact'] for r in results),
              'after_exact_text': sum(r['after_exact'] for r in results),
              'new_exact_text': sum(not r['before_exact'] and r['after_exact'] for r in results),
              'lost_exact_text': sum(r['before_exact'] and not r['after_exact'] for r in results),
              'scorable_names': sum(r['name_scorable'] for r in results),
              'before_exact_names': sum(r['before_name_exact'] for r in results),
              'after_exact_names': sum(r['after_name_exact'] for r in results),
              'before_mean_character_similarity': sum(r['before_similarity'] for r in results) / len(results),
              'after_mean_character_similarity': sum(r['after_similarity'] for r in results) / len(results),
              'selection_counts': counts, 'elapsed_seconds': time.perf_counter()-start,
              'fallback_enabled': args.fallback,
              'scope': 'Synthetic signature-to-method prompts. Copies the supplied signature into a frozen known-target codec reconstruction. Prompt and decoded text choose edits; targets are offline scoring only. Not generator quality or a learned pointer.',
              'limitations': 'Prompt contains the expected signature by dataset construction; not representative of natural-language-only prompts. Conservative fallback may misidentify a body boundary and abstains on many malformed cases. Does not fix body references or verify semantics. Original target was encoded into codec latents.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('\n=== Prompt signature copy experiment ===\n' + json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
