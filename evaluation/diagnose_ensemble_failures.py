"""Describe saved ensemble reconstruction errors by length, project and source region."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from evaluation.compare_codec_error_regions import summarize
from evaluation.ensemble_codec_methods import load_rows
from training.data import encode_text, load_base_tokenizer, sha256_file


def analyze(source, ensemble, lengths):
    if source.keys() != ensemble.keys() or source.keys() != lengths.keys():
        raise ValueError('Source, ensemble and token-length IDs differ')
    grouped = defaultdict(list)
    for key, row in source.items():
        target, selected = row['target'], ensemble[key]['selected']
        if bool(ensemble[key]['ensemble_exact']) != (target == selected):
            raise ValueError(f'Saved exactness does not match target: {key}')
        n = lengths[key]
        bucket = '1-64' if n <= 64 else '65-128' if n <= 128 else '129-192' if n <= 192 else '193-256' if n <= 256 else '>256'
        record = {'target': target, 'reconstructed': selected}
        grouped['all'].append(record)
        grouped['length:' + bucket].append(record)
        if row.get('source'):
            grouped['project:' + row['source'].split('/')[0]].append(record)
    return {name: summarize(rows) for name, rows in sorted(grouped.items())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'ensemble', 'config', 'output'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    output = Path(args.output)
    if output.exists(): raise SystemExit(f'Output exists: {output}')
    source, ensemble = load_rows(args.input), load_rows(args.ensemble)
    ensemble_report = json.loads(Path(args.ensemble).with_suffix('.report.json').read_text(encoding='utf-8'))
    if ensemble_report['input_sha256'] != sha256_file(args.input) or ensemble_report['examples'] != len(source):
        raise SystemExit('Saved ensemble report does not match input')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    lengths = {key: len(encode_text(tok, row['target'], append_eos=True)) for key, row in source.items()}
    groups = analyze(source, ensemble, lengths)
    if groups['all']['exact'] != ensemble_report['ensemble_exact']:
        raise SystemExit('Ensemble report exactness differs')
    report = {'groups': groups, 'input_sha256': sha256_file(args.input),
              'ensemble_sha256': sha256_file(args.ensemble),
              'scope': 'Offline known-target analysis; no new model inference.',
              'limitations': 'Target AST and original text are used only for diagnosis. Diff alignment is heuristic, labels overlap at method level, and parse errors can affect them. Length includes EOS. Reused datasets; no compilation, behavioral or generator test.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
