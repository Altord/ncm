"""Compare aggregate codec reports when per-example IDs are unavailable."""
import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf


def compare(ref, candidate, ref_cfg, candidate_cfg):
    for key in ('input_sha256', 'text_key', 'sequence_length', 'code_only', 'code_extensions', 'examples', 'valid_tokens'):
        if key not in ref or ref[key] != candidate.get(key):
            raise ValueError(f'Reports differ or omit {key}')
    for config in (ref_cfg, candidate_cfg):
        OmegaConf.resolve(config)
    if OmegaConf.to_container(ref_cfg.tokenizer, resolve=True) != OmegaConf.to_container(candidate_cfg.tokenizer, resolve=True):
        raise ValueError('Tokenizer configurations differ')
    return {name: {'examples': r['examples'], 'valid_tokens': r['valid_tokens'],
                   'token_accuracy': r['valid_token_accuracy'], 'exact_sequences': r['exact_sequence_matches'],
                   'character_similarity': r['mean_character_similarity']}
            for name, r in [('reference', ref), ('candidate', candidate)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-report', 'candidate-report', 'reference-config', 'candidate-config'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    ref = json.loads(Path(args.reference_report).read_text(encoding='utf-8'))
    candidate = json.loads(Path(args.candidate_report).read_text(encoding='utf-8'))
    summary = compare(ref, candidate, OmegaConf.load(args.reference_config), OmegaConf.load(args.candidate_config))
    print('\n=== Aggregate general-validation comparison ===')
    for name, r in summary.items():
        print(f"{name}: token accuracy={r['token_accuracy']:.2%}; exact={r['exact_sequences']}/{r['examples']}; character similarity={r['character_similarity']:.2%}")
    print('Aggregate reports only; duplicate/missing row IDs prevent a paired comparison. Tokenizer configuration matches; tokenizer file identity is not independently verified.')


if __name__ == '__main__': main()
