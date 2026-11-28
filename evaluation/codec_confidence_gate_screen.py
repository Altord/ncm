"""Compare frozen-codec disagreement gates on known-input round trips."""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file


def candidates(a, b):
    """Choose by inference-available logit margins; no target is used here."""
    av, ai = a.float().topk(2, dim=-1); bv, bi = b.float().topk(2, dim=-1)
    am, bm = av[..., 0] - av[..., 1], bv[..., 0] - bv[..., 1]
    result = {'equal': (a.float() + b.float()).argmax(-1)}
    for label, ratio in (('margin_half', .5), ('margin_one', 1.), ('margin_two', 2.)):
        result[label] = torch.where(am >= ratio * bm, ai[..., 0], bi[..., 0])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config-a', 'checkpoint-a', 'saved-a', 'config-b', 'checkpoint-b', 'saved-b', 'input', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--batch-size', type=int, default=8)
    args = p.parse_args(); out = Path(args.output)
    if out.exists() or out.with_suffix('.report.json').exists(): raise SystemExit('Output exists')
    ca, cb = OmegaConf.load(args.config_a), OmegaConf.load(args.config_b)
    OmegaConf.resolve(ca); OmegaConf.resolve(cb)
    if (OmegaConf.to_container(ca.tokenizer, resolve=True) != OmegaConf.to_container(cb.tokenizer, resolve=True)
        or int(ca.model.context_length) != int(cb.model.context_length)
        or int(ca.vqvae.in_vocab_size) != int(cb.vqvae.in_vocab_size)):
        raise SystemExit('Tokenizer configuration, context, or vocabulary differs')
    source, saved_a, saved_b = (load_rows(path) for path in (args.input, args.saved_a, args.saved_b))
    if saved_a.keys() != source.keys() or saved_b.keys() != source.keys(): raise SystemExit('IDs differ')
    for path, ckpt in ((args.saved_a, args.checkpoint_a), (args.saved_b, args.checkpoint_b)):
        report = json.loads(Path(path).with_suffix('.report.json').read_text())
        if report['input_sha256'] != sha256_file(args.input) or Path(report['checkpoint']).resolve() != Path(ckpt).resolve():
            raise SystemExit('Saved baseline provenance differs')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    _, a = load_codec(args.config_a, args.checkpoint_a, device); _, b = load_codec(args.config_b, args.checkpoint_b, device)
    tok = load_base_tokenizer(ca.tokenizer); context = int(ca.model.context_length)
    eos, pad = int(tok.eos_token_id), int(tok.pad_token_id)
    rows = list(source.values()); results = []; counts = {}
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            group = rows[start:start + args.batch_size]; packed = []
            for row in group:
                if any(saved[row['id']]['target'] != row['target'] for saved in (saved_a, saved_b)):
                    raise SystemExit('Target differs')
                ids = encode_text(tok, row['target'], True)
                if len(ids) > context: raise SystemExit('Truncation')
                packed.append(fixed_sequence(ids, context, pad, eos))
            x = torch.tensor([r[0] for r in packed], device=device, dtype=torch.long)
            mask = torch.tensor([r[1] for r in packed], device=device, dtype=torch.bool)
            la, lb = a.decode(a.encode(x)).float(), b.decode(b.encode(x)).float()
            pred = candidates(la, lb)
            for name, ids in pred.items():
                c, n = counts.get(name, (0, 0)); counts[name] = (c + int(((ids == x) & mask).sum()), n + int(mask.sum()))
            for i, row in enumerate(group):
                before_a = visible_text(tok, la[i].argmax(-1).tolist(), eos, pad)
                before_b = visible_text(tok, lb[i].argmax(-1).tolist(), eos, pad)
                if before_a != saved_a[row['id']]['reconstructed'] or before_b != saved_b[row['id']]['reconstructed']:
                    raise SystemExit('Recomputed baseline differs')
                outputs = {name: visible_text(tok, ids[i].tolist(), eos, pad) for name, ids in pred.items()}
                results.append({'id': row['id'], 'selected': outputs,
                                'exact': {name: text == row['target'] for name, text in outputs.items()}})
            print(f'Processed {min(start+len(group), len(rows))}/{len(rows)}', flush=True)
    report = {'examples': len(results), 'input_sha256': sha256_file(args.input),
              'checkpoint_a_sha256': sha256_file(args.checkpoint_a), 'checkpoint_b_sha256': sha256_file(args.checkpoint_b),
              'token_accuracy': {k: v[0]/v[1] for k, v in counts.items()},
              'scope': 'Known-input two-codec reconstruction; candidates use frozen logits only; original labels score offline.',
              'limitations': 'Reused datasets; two separate code streams. No compile, semantic or generator test. Tokenizer identity not independently verified.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f:
        for row in results: f.write(json.dumps(row, ensure_ascii=False)+'\n')
    with out.with_suffix('.report.json').open('x', encoding='utf-8') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
