"""Build a source-separated exploratory codec set from pinned external projects."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
import subprocess
from pathlib import Path
from omegaconf import OmegaConf
from training.build_csharp_method_pairs import method_pairs
from training.csharp_structure import csharp_parser
from training.data import encode_text, load_base_tokenizer, sha256_file


def compact(text): return ''.join(text.split())


def shingles(text):
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return {tuple(tokens[i:i+5]) for i in range(len(tokens)-4)}


def build_index(texts):
    index = defaultdict(set)
    for i, text in enumerate(texts):
        for gram in shingles(text): index[gram].add(i)
    return index


def near_overlap(text, index):
    grams = shingles(text)
    if len(grams) < 15: return False
    matches = Counter(i for gram in grams for i in index.get(gram, ()))
    return bool(matches and max(matches.values()) / len(grams) >= .8)


def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'reference-config', 'comparison-config', 'method-train', 'method-validation', 'stress', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--max-per-repository', type=int, default=60)
    p.add_argument('--max-per-file', type=int, default=5)
    args = p.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output)
    report_path = out.with_suffix('.manifest.json')
    if out.exists() or report_path.exists(): raise SystemExit('Dataset output already exists')
    if min(args.max_per_file, args.max_per_repository) < 1: p.error('Limits must be positive')
    a, b = OmegaConf.load(args.reference_config), OmegaConf.load(args.comparison_config)
    OmegaConf.resolve(a); OmegaConf.resolve(b)
    if (OmegaConf.to_container(a.tokenizer, resolve=True) != OmegaConf.to_container(b.tokenizer, resolve=True) or
        int(a.model.context_length) != int(b.model.context_length) or sha256_file(a.data.input) != sha256_file(b.data.input)):
        raise SystemExit('Codec tokenizer configuration, context or training input differs')
    cache = json.loads((Path(a.data.cache_dir)/'manifest.json').read_text(encoding='utf-8'))
    valcache = json.loads((Path(a.train.validation_cache_dir)/'manifest.json').read_text(encoding='utf-8'))
    val = Path(valcache['source'])
    if cache['source_sha256'] != sha256_file(a.data.input) or valcache['source_sha256'] != sha256_file(val):
        raise SystemExit('Original training/validation inputs do not match cache manifests')
    protected_paths = [Path(a.data.input), val, Path(args.method_train), Path(args.method_validation), Path(args.stress)]
    protected = []
    for path in protected_paths:
        for row in records(path):
            value = row.get('target', row.get('text'))
            if isinstance(value, str) and value: protected.append(value)
    if not protected: raise SystemExit('No protected training or evaluation text found')
    print(f'Indexing {len(protected)} existing training/evaluation texts...', flush=True)
    norms = {compact(t) for t in protected}
    long_texts = [compact(t) for t in protected if len(compact(t)) >= 24]
    index = build_index(protected)
    tokenizer = load_base_tokenizer(a.tokenizer)
    hf = getattr(tokenizer, 'hf_tokenizer', None)
    if hf is None or not getattr(hf, 'is_fast', False): raise SystemExit('Expected fast HF tokenizer')
    parser = csharp_parser()
    repos, selected, counts = {}, [], Counter()
    for repo in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (repo/'.git').exists(): raise SystemExit(f'Expected pinned git repository: {repo}')
        rev = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
        if len(rev) != 40: raise SystemExit(f'Invalid git revision: {repo}')
        repos[repo.name] = {'commit': rev, 'files_scanned': 0, 'methods_selected': 0}
        for path in sorted(repo.rglob('*.cs')):
            if '.git' in path.parts: continue
            repos[repo.name]['files_scanned'] += 1
            per_file = 0
            try:
                candidates = method_pairs(path, root, parser, hf, 256, int(a.model.context_length), counts)
                for row in candidates:
                    text = row['target']; norm = compact(text)
                    if len(encode_text(tokenizer, text, True)) > int(a.model.context_length):
                        counts['actual_tokenizer_too_long'] += 1; continue
                    if norm in norms or (len(norm) >= 24 and any(norm in existing for existing in long_texts)):
                        counts['exact_or_contained_overlap'] += 1; continue
                    if near_overlap(text, index):
                        counts['high_lexical_overlap'] += 1; continue
                    if per_file >= args.max_per_file or repos[repo.name]['methods_selected'] >= args.max_per_repository:
                        counts['sampling_limit'] += 1; continue
                    row['license'] = 'MIT; see LICENSE in source repository'
                    selected.append(row); norms.add(norm)
                    per_file += 1; repos[repo.name]['methods_selected'] += 1
            except (UnicodeDecodeError, OSError):
                counts['unreadable'] += 1
        print(f"{repo.name}: {repos[repo.name]['files_scanned']} C# files; {repos[repo.name]['methods_selected']} methods", flush=True)
    if len(selected) < 10: raise SystemExit(f'Only {len(selected)} methods survived; refusing an underpowered screen')
    if len({r['id'] for r in selected}) != len(selected): raise SystemExit('Duplicate method IDs')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in selected: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    report = {'examples':len(selected), 'repositories':repos, 'excluded':dict(counts),
              'source_sha256':{str(p):sha256_file(p) for p in protected_paths}, 'dataset_sha256':sha256_file(out),
              'filter':'Exclude exact/contained normalized text and >=80% shared 5-token shingles with any indexed text (at least 15 unique shingles).',
              'limitations':'Exploratory public repositories; lexical screening misses renamed or semantic duplicates. Small methods get exact screening only. No guarantee a pretrained model has not seen the code. Not compiled or a generator evaluation.'}
    with report_path.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print('\n=== External method dataset ===\n'+json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
