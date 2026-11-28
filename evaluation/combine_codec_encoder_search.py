"""Choose one saved codec stream from two known-input encoder searches."""
import argparse
import json
from pathlib import Path
import torch
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text
from training.data import encode_text, load_base_tokenizer, sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('config','checkpoint','input','first','second','output'): p.add_argument('--'+n,required=True)
    p.add_argument('--device',default='cuda'); args=p.parse_args(); out=Path(args.output)
    if any(p.exists() for p in (out,out.with_suffix('.report.json'),out.with_suffix('.codes.pt'))): raise SystemExit('Output exists')
    source,a,b=(load_rows(x) for x in (args.input,args.first,args.second))
    if a.keys()!=b.keys() or not a.keys()<=source.keys(): raise SystemExit('Input IDs differ')
    for path in (args.first,args.second):
        meta=json.loads(Path(path).with_suffix('.report.json').read_text())
        if meta['input_sha256']!=sha256_file(args.input) or meta['codec_sha256']!=sha256_file(args.checkpoint): raise SystemExit('Search provenance differs')
    def codes(path):
        rows=torch.load(Path(path).with_suffix('.codes.pt'),map_location='cpu',weights_only=True)
        result={r['id']:r for r in rows}
        if len(rows)!=len(result) or result.keys()!=a.keys(): raise SystemExit('Code IDs differ')
        return result
    ac,bc=codes(args.first),codes(args.second)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg,model=load_codec(args.config,args.checkpoint,device); tok=load_base_tokenizer(cfg.tokenizer)
    eos,pad=int(tok.eos_token_id),int(tok.pad_token_id)
    result,selected_codes=[],[]
    with torch.inference_mode():
        for key in a:
            ra,rb=a[key],b[key]; target=source[key]['target']
            better=rb['selected_errors']<ra['selected_errors']
            chosen,code=(rb,bc[key]) if better else (ra,ac[key])
            offset=int(code.get('offset',0)); indices=code['indices'].to(device).long().unsqueeze(0)
            predicted=model.decode(indices).argmax(-1)[0,offset:]
            token_ids=encode_text(tok,target,True); valid=int(code['valid_length'])
            if len(token_ids)!=valid: raise SystemExit('Length differs')
            errors=int((predicted[:valid]!=torch.tensor(token_ids,device=device)).sum())
            selected=visible_text(tok,predicted.tolist(),eos,pad)
            if selected!=chosen['selected'] or errors!=chosen['selected_errors']: raise SystemExit('Saved code mismatch')
            result.append({'id':key,'selected':selected,'selected_exact':selected==target,'selected_errors':errors,'offset':offset,'selection':'second' if better else 'first'})
            selected_codes.append({'id':key,'indices':code['indices'],'offset':offset,'valid_length':valid})
    report={'examples':len(result),'first_exact':sum(r['selected_exact'] for r in a.values()),
            'second_exact':sum(r['selected_exact'] for r in b.values()),'selected_exact':sum(r['selected_exact'] for r in result),
            'selected_errors':sum(r['selected_errors'] for r in result),'second_selected':sum(r['selection']=='second' for r in result),
            'input_sha256':sha256_file(args.input),'codec_sha256':sha256_file(args.checkpoint),
            'scope':'Known-input encoder compares token errors and transmits one selected VQ stream plus offset and length metadata.',
            'limitations':'Target required for encoder selection; two searches add compute. Reused sets; not generation or behavior.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        for row in result: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    torch.save(selected_codes,out.with_suffix('.codes.pt'))
    with out.with_suffix('.report.json').open('x',encoding='utf-8') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
