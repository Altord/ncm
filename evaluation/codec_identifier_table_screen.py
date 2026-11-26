"""Screen reversible identifier canonicalization with a frozen known-input codec."""
import argparse
import json
import re
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.csharp_structure import _walk, csharp_parser
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file
from evaluation.codec_prompt_signature_copy import WRAPPER
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text


def canonicalize(text):
    raw = text.encode('utf-8'); prefix = WRAPPER.encode('utf-8')
    wrapped = prefix + raw + b'\n}'
    tree = csharp_parser().parse(wrapped)
    if tree.root_node.has_error: return None
    nodes = [n for n in _walk(tree.root_node) if n.type in ('identifier', 'verbatim_identifier')
             and n.start_byte >= len(prefix) and n.end_byte <= len(prefix) + len(raw)]
    if not nodes: return None
    # Pick a prefix absent from all input bytes, including comments and literals.
    marker = '__ncmid_'
    while marker.encode() in raw: marker += 'x'
    table, spans = {}, []
    for node in nodes:
        start, end = node.start_byte - len(prefix), node.end_byte - len(prefix)
        original = raw[start:end].decode('utf-8')
        if original not in table: table[original] = f'{marker}{len(table)}__'
        spans.append((start, end, table[original].encode('utf-8')))
    if any(a[1] > b[0] for a, b in zip(sorted(spans), sorted(spans)[1:])):
        raise ValueError('Overlapping identifier spans')
    changed = raw
    for start, end, replacement in sorted(spans, reverse=True):
        changed = changed[:start] + replacement + changed[end:]
    mapping = {replacement: original for original, replacement in table.items()}
    return changed.decode('utf-8'), mapping


def restore(text, mapping):
    if not mapping: return text
    # Match complete identifiers only; corrupted placeholders remain unchanged.
    pattern = re.compile(r'(?<![\w])(?:' + '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)) + r')(?![\w])')
    return pattern.sub(lambda m: mapping[m.group(0)], text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'baseline', 'output'): p.add_argument('--'+name, required=True)
    p.add_argument('--max-examples', type=int, default=64)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.max_examples < 1: p.error('max-examples must be positive')
    out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    cfg = OmegaConf.load(args.config); OmegaConf.resolve(cfg)
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    report = json.loads(Path(args.baseline).with_suffix('.report.json').read_text(encoding='utf-8'))
    if (source.keys() != baseline.keys() or report['input_sha256'] != sha256_file(args.input) or
        Path(report['checkpoint']).resolve() != Path(args.checkpoint).resolve()): raise SystemExit('Baseline input, IDs or checkpoint differ')
    tok = load_base_tokenizer(cfg.tokenizer)
    eos, pad, context = int(tok.eos_token_id), int(tok.pad_token_id), int(cfg.model.context_length)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    vcfg = OmegaConf.to_container(cfg.vqvae, resolve=True); vcfg.pop('checkpoint_path', None)
    if vcfg.get('autoregressive_decoder', False): raise SystemExit('Parallel decoder required')
    model = VQVAE(**OmegaConf.create(vcfg))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    output, table_bytes, applied, skipped = [], 0, 0, 0
    rows = list(source.values())[:args.max_examples]
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            key, target = row['id'], row['target']
            if baseline[key]['target'] != target: raise SystemExit(f'Target differs: {key}')
            before = baseline[key]['reconstructed']; selected = before
            canonical = canonicalize(target)
            mapping = {}; reason = 'parse_or_no_identifiers'
            if canonical:
                text, mapping = canonical
                if restore(text, mapping) != target: raise SystemExit(f'Canonicalization is not reversible: {key}')
                ids = encode_text(tok, text, append_eos=True)
                if len(ids) <= context:
                    packed, _ = fixed_sequence(ids, context, pad, eos)
                    x = torch.tensor([packed], dtype=torch.long, device=device)
                    prediction = model.decode(model.encode(x)).argmax(-1)[0].tolist()
                    selected = restore(visible_text(tok, prediction, eos, pad), mapping)
                    applied += 1; reason = 'canonicalized'
                    table_bytes += len(json.dumps(mapping, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
                else: reason = 'canonicalized_input_exceeds_context'
            if reason != 'canonicalized': skipped += 1; mapping = {}
            output.append({'id': key, 'baseline_exact': before == target, 'selected_exact': selected == target,
                           'selected': selected, 'table': mapping, 'reason': reason})
            if index % 16 == 0 or index == len(rows): print(f'Processed {index}/{len(rows)}', flush=True)
    summary = {'examples': len(output), 'applied': applied, 'skipped': skipped,
               'baseline_exact': sum(r['baseline_exact'] for r in output),
               'selected_exact': sum(r['selected_exact'] for r in output),
               'gained_exact': sum(not r['baseline_exact'] and r['selected_exact'] for r in output),
               'lost_exact': sum(r['baseline_exact'] and not r['selected_exact'] for r in output),
               'identifier_table_json_utf8_bytes': table_bytes,
               'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint),
               'scope': 'Known-input codec: identifiers are replaced before encoding; the exact source-derived name table is needed to restore decoded text.',
               'limitations': 'Frozen codec was not trained on canonicalized text; repeatedly used first-N samples. JSON table size is descriptive, not packed storage; VQ bytes, framing, model, runtime and fallback bytes excluded. Not compilation, generation or semantic testing. Original-name table is unavailable for unknown generated targets unless supplied separately.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(summary, f, indent=2); f.write('\n')
    print('\n=== Identifier-table codec screen ===\n'+json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__': main()
