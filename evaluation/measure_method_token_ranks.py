"""Offline true-token rank diagnostic for saved method codec failures."""
import argparse
import json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import load_base_tokenizer, encode_text, fixed_sequence, sha256_file
from evaluation.godot_csharp_codec_eval import visible_text
from evaluation.codec_prompt_signature_copy import WRAPPER
from training.csharp_structure import csharp_parser


def optimistic_rank(scores, target):
    return int((scores > scores[target]).sum()) + 1


def load(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    data = {r['id']: r for r in rows}
    if not rows or len(data) != len(rows): raise ValueError(f'Empty or duplicate IDs: {path}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'checkpoint', 'input', 'sample', 'signature-copy', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit('Output exists')
    source, sample, copies = (load(x) for x in (args.input, args.sample, args.signature_copy))
    if not set(sample).issubset(source) or not set(sample).issubset(copies): raise SystemExit('IDs differ')
    # Verify the saved codec evaluation belongs to this model and input when its report is available.
    report_file = Path(args.input).with_suffix('.report.json')
    report = json.loads(report_file.read_text(encoding='utf-8'))
    if Path(report['checkpoint']).resolve() != Path(args.checkpoint).resolve(): raise SystemExit('Codec checkpoint differs')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tok = load_base_tokenizer(cfg.tokenizer)
    vc = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vc.pop('checkpoint_path', None)
    device = torch.device(args.device)
    model = VQVAE(**OmegaConf.create(vc))
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    pad, eos, context = int(tok.pad_token_id), int(tok.eos_token_id), int(cfg.model.context_length)
    rows = []
    for key, saved in sample.items():
        original, copy = source[key], copies[key]
        if saved['baseline'] != original['reconstructed'] or bool(copy['before_exact']) != (original['reconstructed'] == original['target']):
            raise SystemExit(f'Baseline mismatch: {key}')
        if copy['selected'] == original['target']: continue
        truth = encode_text(tok, original['target'], append_eos=True)
        if len(truth) > context: raise SystemExit(f'Truncated target: {key}')
        ids, mask = fixed_sequence(truth, context, pad, eos)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model.decode(model.encode(x))[0].float()
            predicted = logits.argmax(-1)
            if visible_text(tok, predicted.tolist(), eos, pad) != original['reconstructed']:
                raise SystemExit(f'Frozen codec output differs from saved baseline: {key}')
            errors = []
            for pos in range(sum(mask)):
                target_id, got = int(ids[pos]), int(predicted[pos])
                if target_id != got:
                    scores = logits[pos]
                    errors.append({'position': pos, 'rank': optimistic_rank(scores, target_id),
                        'target_piece': tok.decode([target_id], skip_special_tokens=False),
                        'predicted_piece': tok.decode([got], skip_special_tokens=False)})
        syntax_error = csharp_parser().parse((WRAPPER + copy['selected'] + '\n}').encode('utf-8')).root_node.has_error
        rows.append({'id': key, 'parser_has_error_after_signature': syntax_error, 'errors': errors})
        print(f'Processed {len(rows)} failures; latest={key}, token errors={len(errors)}', flush=True)
    ranks = [e['rank'] for row in rows for e in row['errors']]
    report = {'methods': len(rows), 'token_errors': len(ranks),
        'methods_with_parser_error': sum(r['parser_has_error_after_signature'] for r in rows),
        'methods_with_no_token_errors': sum(not r['errors'] for r in rows),
        'error_ranks_at_most': {str(k): sum(r <= k for r in ranks) for k in (2, 5, 10, 100, 1000)},
        'methods_all_errors_at_most': {str(k): sum(bool(r['errors']) and all(e['rank'] <= k for e in r['errors']) for r in rows) for k in (2, 5, 10, 100, 1000)},
        'input_sha256': sha256_file(args.input), 'checkpoint_sha256': sha256_file(args.checkpoint), 'rows': rows,
        'scope': 'Offline oracle diagnosis only: uses original token IDs and known-target VQ encoding. Ranks are from the original codec output, not the signature-edited text.',
        'limitations': 'True ranks cannot be used to select repairs at inference. Optimistic ties; reused sample. Token agreement does not guarantee byte-exact text or semantic correctness.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2, ensure_ascii=False); f.write('\n')
    print('\n=== Real-method token rank diagnostic to share ===')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2))
    print(f'Report: {out}')


if __name__ == '__main__': main()
