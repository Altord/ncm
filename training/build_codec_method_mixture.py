"""Create a file-separated codec training mixture with complete methods."""
import argparse
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf
from training.data import encode_text, load_base_tokenizer, sha256_file


def read(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(r.get('source'), str) or not r['source'] for r in rows):
        raise ValueError(f'Empty data or missing source paths: {path}')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-config', 'methods', 'heldout-methods', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--method-fraction', type=float, default=0.25)
    args = p.parse_args()
    if not 0 < args.method_fraction < 1: p.error('method fraction must be in (0,1)')
    out = Path(args.output)
    manifest_path = out.with_suffix('.manifest.json')
    if out.exists() or manifest_path.exists(): raise SystemExit('Mixture output already exists')
    cfg = OmegaConf.load(args.reference_config)
    OmegaConf.resolve(cfg)
    base_path = Path(cfg.data.input)
    cache = json.loads((Path(cfg.data.cache_dir)/'manifest.json').read_text(encoding='utf-8'))
    if cache['source_sha256'] != sha256_file(base_path):
        raise SystemExit('Original training data no longer matches the baseline cache')
    val_manifest = json.loads((Path(cfg.train.validation_cache_dir)/'manifest.json').read_text(encoding='utf-8'))
    val_path = Path(val_manifest['source'])
    if sha256_file(val_path) != val_manifest['source_sha256']:
        raise SystemExit('Original codec validation data no longer matches its cache')
    base, methods, heldout, validation = (read(x) for x in (base_path, args.methods, args.heldout_methods, val_path))
    protected_sources = {r['source'] for r in heldout + validation}
    protected_text = {''.join(r.get('target', r.get('text', '')).split()) for r in heldout + validation}
    tok = load_base_tokenizer(cfg.tokenizer)
    context = int(cfg.model.context_length)
    eligible, excluded = [], {'source_overlap': 0, 'duplicate_text': 0, 'too_long': 0}
    for row in methods:
        target = row.get('target')
        if not isinstance(target, str) or not target: raise ValueError('Method missing target text')
        if row['source'] in protected_sources:
            excluded['source_overlap'] += 1
        elif ''.join(target.split()) in protected_text:
            excluded['duplicate_text'] += 1
        elif len(encode_text(tok, target, True)) > context:
            excluded['too_long'] += 1
        else:
            eligible.append(row)
    eligible.sort(key=lambda r: hashlib.sha256((r['source']+'\0'+r['target']).encode('utf-8')).hexdigest())
    requested = round(len(base)*args.method_fraction/(1-args.method_fraction))
    selected = eligible[:requested]
    if not selected or len(selected) < 0.8 * requested:
        raise SystemExit(f'Too few eligible methods after separation checks: {len(selected)}/{requested} requested')
    output = list(base) + [{'source': r['source'], 'text': r['target']} for r in selected]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    manifest = {'original_examples': len(base), 'method_examples': len(selected), 'requested_methods': requested,
                'actual_method_fraction': len(selected)/len(output), 'exclusions': excluded,
                'base_sha256': sha256_file(base_path), 'methods_sha256': sha256_file(args.methods),
                'heldout_sha256': sha256_file(args.heldout_methods), 'codec_validation_sha256': sha256_file(val_path),
                'mixture_sha256': sha256_file(out),
                'limitations': 'Source-path and whitespace-insensitive exact-text filtering do not detect near duplicates or renamed sources.'}
    with manifest_path.open('x', encoding='utf-8') as f: json.dump(manifest, f, indent=2); f.write('\n')
    print('\n=== Codec method mixture ===\n'+json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__': main()
