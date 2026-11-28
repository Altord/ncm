"""Choose a pair of saved encoder-optimized codec streams by target token errors."""
import argparse
import json
import time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from evaluation.codec_index_search import score
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config-a','checkpoint-a','search-a','config-b','checkpoint-b','search-b','input','output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args(); out = Path(args.output); codefile = out.with_suffix('.codes.pt')
    if any(p.exists() for p in (out,out.with_suffix('.report.json'),codefile)): raise SystemExit('Output exists')
    def meta(path): return json.loads(Path(path).with_suffix('.report.json').read_text())
    for path, ckpt in ((args.search_a,args.checkpoint_a),(args.search_b,args.checkpoint_b)):
        report = meta(path)
        if report['input_sha256'] != sha256_file(args.input) or report['codec_sha256'] != sha256_file(ckpt):
            raise SystemExit('Search provenance differs')
    ca, cb = OmegaConf.load(args.config_a), OmegaConf.load(args.config_b)
    OmegaConf.resolve(ca); OmegaConf.resolve(cb)
    if (OmegaConf.to_container(ca.tokenizer, resolve=True) != OmegaConf.to_container(cb.tokenizer, resolve=True)
        or int(ca.model.context_length) != int(cb.model.context_length)
        or int(ca.vqvae.in_vocab_size) != int(cb.vqvae.in_vocab_size)):
        raise SystemExit('Tokenizer or context differs')
    src, sa, sb = (load_rows(p) for p in (args.input,args.search_a,args.search_b))
    if sa.keys() != src.keys() or sb.keys() != src.keys(): raise SystemExit('IDs differ')
    stored = torch.load(Path(args.search_b).with_suffix('.codes.pt'), map_location='cpu', weights_only=True)
    sb_codes = {r['id']:r for r in stored}
    if sb_codes.keys() != src.keys(): raise SystemExit('Saved code IDs differ')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    _, a = load_codec(args.config_a,args.checkpoint_a,device); _, b = load_codec(args.config_b,args.checkpoint_b,device)
    tok = load_base_tokenizer(ca.tokenizer); length = int(ca.model.context_length)
    eos,pad = int(tok.eos_token_id),int(tok.pad_token_id)
    output, codes = [], []; start = time.perf_counter(); errors_before = errors_after = 0
    with torch.inference_mode():
        for row in src.values():
            key,target = row['id'],row['target']; token_ids = encode_text(tok,target,True)
            if len(token_ids) > length: raise SystemExit('Truncated target')
            packed, valid = fixed_sequence(token_ids,length,pad,eos)
            x = torch.tensor([packed],device=device); mask = torch.tensor([valid],device=device,dtype=torch.bool)
            ai0,bi0 = a.encode(x),b.encode(x)
            ai1 = torch.tensor([sa[key]['codes']],device=device,dtype=torch.long)
            bi1 = sb_codes[key]['indices'].to(device).long().unsqueeze(0)
            if int(sb_codes[key]['valid_length']) != int(mask.sum()): raise SystemExit('Length mismatch')
            al,bl = [a.decode(i).float() for i in (ai0,ai1)], [b.decode(i).float() for i in (bi0,bi1)]
            if (visible_text(tok,al[1][0].argmax(-1).tolist(),eos,pad) != sa[key]['selected']
                or visible_text(tok,bl[1][0].argmax(-1).tolist(),eos,pad) != sb[key]['selected']):
                raise SystemExit('Saved codes do not reproduce search output')
            selections = []
            for ia,ib in ((0,0),(0,1),(1,0),(1,1)):
                logits = al[ia]+bl[ib]; err,ce = score(logits,x,mask)
                selections.append((int(err[0]),float(ce[0]),ia,ib,logits))
            baseline = selections[0]; best = min(selections,key=lambda z:(z[0],z[1]))
            # Baseline is retained on ties in both metrics.
            before = visible_text(tok,baseline[4][0].argmax(-1).tolist(),eos,pad)
            selected = visible_text(tok,best[4][0].argmax(-1).tolist(),eos,pad)
            errors_before += baseline[0]; errors_after += best[0]
            output.append({'id':key,'selected':selected,'baseline_exact':before == target,'ensemble_exact':selected == target,
                           'baseline_token_errors':baseline[0],'selected_token_errors':best[0],'choice':[best[2],best[3]]})
            codes.append({'id':key,'a':(ai0,ai1)[best[2]][0].cpu(),'b':(bi0,bi1)[best[3]][0].cpu(),'valid_length':int(mask.sum())})
            print(f'Processed {len(output)}/{len(src)}',flush=True)
    report = {'examples':len(output), 'baseline_exact':sum(r['baseline_exact'] for r in output),
              'selected_exact':sum(r['ensemble_exact'] for r in output),
              'baseline_token_errors':errors_before,'selected_token_errors':errors_after,
              'seconds':time.perf_counter()-start,'input_sha256':sha256_file(args.input),
              'checkpoint_a_sha256':sha256_file(args.checkpoint_a),'checkpoint_b_sha256':sha256_file(args.checkpoint_b),
              'scope':'Encoder uses known original tokens to select one of four pairs of code streams; decoder averages their logits.',
              'limitations':'Known-input reconstruction only. Two full latent streams; costly search. Reused stress; no generation or behavioral test.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        for row in output: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    torch.save(codes,codefile)
    with out.with_suffix('.report.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__': main()
