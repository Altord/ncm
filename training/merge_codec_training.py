"""Append provenance-checked supplemental method examples to codec training data."""
import argparse
import json
from pathlib import Path
from training.data import sha256_file


def merge(original, supplemental, text_key, stream):
    original_count = supplemental_count = 0
    with Path(original).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row.get(text_key), str): raise ValueError('Original training record lacks text')
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); original_count += 1
    with Path(supplemental).open(encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row.get('target'), str) or not row['target'] or Path(str(row.get('source',''))).suffix.lower() != '.cs':
                raise ValueError('Supplemental record lacks C# target or source')
            row[text_key] = row['target']
            stream.write(json.dumps(row, ensure_ascii=False)+'\n'); supplemental_count += 1
    if not original_count or not supplemental_count: raise ValueError('Original and supplemental data must both be nonempty')
    return original_count, supplemental_count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'supplemental', 'manifest', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--text-key', default='text'); p.add_argument('--min-supplemental', type=int, default=200)
    args = p.parse_args()
    out = Path(args.output); report = out.with_suffix('.manifest.json')
    if out.exists() or report.exists(): raise SystemExit('Merged output exists')
    metadata = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if metadata['dataset_sha256'] != sha256_file(args.supplemental): raise SystemExit('Supplemental dataset hash differs')
    if metadata['examples'] < args.min_supplemental: raise SystemExit(f'Only {metadata["examples"]} supplemental methods; minimum is {args.min_supplemental}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: original_count, supplemental_count = merge(args.original, args.supplemental, args.text_key, f)
    if supplemental_count != metadata['examples']: raise ValueError('Supplemental row count differs from manifest')
    result = {'original_records': original_count, 'supplemental_records': supplemental_count,
              'original_sha256': sha256_file(args.original), 'supplemental_sha256': sha256_file(args.supplemental),
              'supplemental_manifest_sha256': sha256_file(args.manifest), 'merged_sha256': sha256_file(out),
              'scope': 'Training corpus only; supplemental methods were lexically screened against protected datasets.',
              'limitations': 'Lexical overlap checks cannot exclude all semantic duplicates; public training content may include upstream code already seen elsewhere.'}
    with report.open('x', encoding='utf-8') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
