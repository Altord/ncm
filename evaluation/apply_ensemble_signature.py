"""Apply a supplied signature to saved known-target codec ensemble outputs."""
import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from evaluation.codec_prompt_signature_copy import copy_signature, parse_method
from training.data import sha256_file


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    data = {r['id']: r for r in rows}
    if not rows or len(data) != len(rows): raise ValueError(f'Empty or duplicate IDs: {path}')
    return data


def evaluate(source, ensemble, fallback=True):
    if set(source) != set(ensemble): raise ValueError('Input IDs differ')
    result = []
    for key, row in source.items():
        prompt, target = row.get('prompt'), row.get('target')
        if not isinstance(prompt, str) or not isinstance(target, str): raise ValueError(f'Missing prompt or target: {key}')
        before = ensemble[key]['selected']
        if bool(ensemble[key]['ensemble_exact']) != (before == target):
            raise ValueError(f'Ensemble offline score differs: {key}')
        selected, reason = copy_signature(prompt, before, fallback=fallback)
        expected = parse_method(target)
        previous = parse_method(before)
        new = parse_method(selected)
        name = expected['name'] if expected else None
        result.append({'id': key, 'before': before, 'selected': selected, 'selection': reason,
            'before_exact': before == target, 'after_exact': selected == target,
            'name_scorable': name is not None,
            'before_name_exact': name is not None and previous is not None and previous['name'] == name,
            'after_name_exact': name is not None and new is not None and new['name'] == name,
            'before_similarity': difflib.SequenceMatcher(None, target, before, autojunk=False).ratio(),
            'after_similarity': difflib.SequenceMatcher(None, target, selected, autojunk=False).ratio()})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--ensemble', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    ensemble_report = json.loads(Path(args.ensemble).with_suffix('.report.json').read_text(encoding='utf-8'))
    if ensemble_report['input_sha256'] != sha256_file(args.input): raise SystemExit('Ensemble input hash differs')
    source, ensemble = load(args.input), load(args.ensemble)
    if ensemble_report['examples'] != len(source): raise SystemExit('Ensemble report count differs')
    rows = evaluate(source, ensemble)
    if sum(r['before_exact'] for r in rows) != ensemble_report['ensemble_exact']:
        raise SystemExit('Saved ensemble exact count differs')
    result = {'examples': len(rows), 'before_exact': sum(r['before_exact'] for r in rows),
        'after_exact': sum(r['after_exact'] for r in rows),
        'gained_exact': sum(not r['before_exact'] and r['after_exact'] for r in rows),
        'lost_exact': sum(r['before_exact'] and not r['after_exact'] for r in rows),
        'scorable_names': sum(r['name_scorable'] for r in rows),
        'before_exact_names': sum(r['before_name_exact'] for r in rows),
        'after_exact_names': sum(r['after_name_exact'] for r in rows),
        'before_character_similarity': sum(r['before_similarity'] for r in rows)/len(rows),
        'after_character_similarity': sum(r['after_similarity'] for r in rows)/len(rows),
        'selection_counts': dict(Counter(r['selection'] for r in rows)),
        'input_sha256': sha256_file(args.input), 'ensemble_sha256': sha256_file(args.ensemble),
        'scope': 'Known-target two-codec ensemble reconstruction with synthetic signature supplied in the input prompt; target body is used for offline scoring only.',
        'limitations': 'No new model inference, compilation, behavioral check, or generator test. Dataset repeatedly used. Two codec passes and latent streams remain necessary.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    out.with_suffix('.report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print('\n=== Ensemble + supplied-signature results to share ===\n'+json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
