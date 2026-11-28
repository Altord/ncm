"""Compare two saved ensemble outputs; target is used for scoring only."""
import argparse
import json
from pathlib import Path
from evaluation.ensemble_codec_methods import load_rows
from training.data import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'reference', 'candidate'):
        p.add_argument('--' + key, required=True)
    args = p.parse_args()
    source, old, new = (load_rows(x) for x in (args.input, args.reference, args.candidate))
    if source.keys() != old.keys() or source.keys() != new.keys():
        raise SystemExit('Input IDs differ')
    for path in (args.reference, args.candidate):
        meta = json.loads(Path(path).with_suffix('.report.json').read_text())
        if meta['input_sha256'] != sha256_file(args.input):
            raise SystemExit('Input hashes differ')
    old_only = new_only = both = neither = 0
    for key, row in source.items():
        a = old[key]['selected'] == row['target']
        b = new[key]['selected'] == row['target']
        both += a and b
        old_only += a and not b
        new_only += b and not a
        neither += not a and not b
    print(json.dumps({'examples': len(source), 'both_exact': both, 'reference_only_exact': old_only,
                      'candidate_only_exact': new_only, 'neither_exact': neither,
                      'scope': 'Offline scoring; knowing which is exact would require the original target.'}, indent=2))


if __name__ == '__main__':
    main()
