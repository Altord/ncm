"""Offline rank analysis for unresolved known-input codec reconstructions."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import load_base_tokenizer, encode_text, fixed_sequence
from evaluation.godot_csharp_codec_eval import file_sha256, visible_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--selected', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    baseline_path = Path(args.baseline)
    report = json.loads(baseline_path.with_suffix('.report.json').read_text(encoding='utf-8'))
    if (report.get('input_sha256') != file_sha256(Path(args.input)) or
        Path(report.get('checkpoint', '')).resolve() != Path(args.checkpoint).resolve()):
        raise SystemExit('Baseline input or checkpoint does not match')
    baseline = [json.loads(line) for line in baseline_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = [json.loads(line) for line in Path(args.selected).read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {r['id']: r for r in selected}
    if len(by_id) != len(selected) or len(selected) != len(baseline):
        raise SystemExit('Selected records do not match baseline count or have duplicate IDs')
    for row in baseline:
        other = by_id.get(row['id'])
        if not other or other['baseline'] != row['reconstructed']:
            raise SystemExit(f'Selected record does not match baseline: {row["id"]}')
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    tokenizer = load_base_tokenizer(cfg.tokenizer)
    vqcfg = OmegaConf.to_container(cfg.vqvae, resolve=True)
    vqcfg.pop('checkpoint_path', None)
    model = VQVAE(**OmegaConf.create(vqcfg))
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('GPU unavailable')
    model.to(device).eval().requires_grad_(False)
    eos, pad = int(tokenizer.eos_token_id), int(tokenizer.pad_token_id)
    context = int(cfg.model.context_length)
    unresolved = [r for r in baseline if by_id[r['id']]['compile_status'] != 'passed']
    result_rows = []
    for n, row in enumerate(unresolved, 1):
        truth = encode_text(tokenizer, row['target'], append_eos=True)
        if len(truth) > context:
            raise SystemExit(f'Target does not fit codec context: {row["id"]}')
        ids, mask = fixed_sequence(truth, context, pad, eos)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model.decode(model.encode(x))[0].float()
            predicted = logits.argmax(-1)
            if visible_text(tokenizer, predicted.tolist(), eos, pad) != row['reconstructed']:
                raise SystemExit(f'Baseline reconstruction mismatch: {row["id"]}')
            errors = []
            for pos in range(sum(mask)):
                target = int(ids[pos])
                got = int(predicted[pos])
                if got != target:
                    scores = logits[pos]
                    # Rank one is best; ties use optimistic rank.
                    rank = int((scores > scores[target]).sum()) + 1
                    errors.append({'position': pos, 'target_token_id': target,
                                   'predicted_token_id': got, 'target_rank': rank,
                                   'target_piece': tokenizer.decode([target], skip_special_tokens=False),
                                   'predicted_piece': tokenizer.decode([got], skip_special_tokens=False)})
        result_rows.append({'id': row['id'], 'valid_tokens': sum(mask), 'token_errors': len(errors),
                            'reference': row['target'], 'reconstruction': row['reconstructed'],
                            'errors': errors})
        print(f"{row['id']}: {len(errors)} token errors; target ranks "
              f"{[e['target_rank'] for e in errors]}", flush=True)
        for error in errors:
            print(f"  position {error['position']}: {error['predicted_piece']!r} -> "
                  f"{error['target_piece']!r} (original token rank {error['target_rank']})", flush=True)
        print(f'Progress {n}/{len(unresolved)}', flush=True)
    result = {'unresolved_examples': len(result_rows),
              'single_token_error_examples': sum(r['token_errors'] == 1 for r in result_rows),
              'multiple_token_error_examples': sum(r['token_errors'] > 1 for r in result_rows),
              'zero_token_error_examples': sum(r['token_errors'] == 0 for r in result_rows),
              'examples': result_rows,
              'note': 'Uses original target token IDs only for offline diagnosis; this is not an inference repair. '
                      'Rank uses optimistic tie handling. One token error can affect multiple characters, '
                      'and text decoding or EOS behavior may differ from token-level counts.'}
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Output exists; choose a new path')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('\n=== Remaining codec failure diagnosis ===')
    print(f"Unresolved: {len(result_rows)}; single-token errors: {result['single_token_error_examples']}; "
          f"multiple-token errors: {result['multiple_token_error_examples']}; "
          f"zero-token errors: {result['zero_token_error_examples']}")
    print(result['note'])
    print(f'Report: {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
