"""Screen encoder-chosen padding offsets with one index stream plus offset metadata."""
import argparse
import json
import time
from pathlib import Path
import torch
from evaluation.codec_latent_refiner_screen import load_codec
from evaluation.ensemble_codec_methods import load_rows
from evaluation.godot_csharp_codec_eval import visible_text
from training.data import encode_text, fixed_sequence, load_base_tokenizer, sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','input','baseline','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--offsets',type=int,nargs='+',default=[0,4,8,16])
    p.add_argument('--max-examples',type=int,default=0)
    args=p.parse_args(); out=Path(args.output); codefile=out.with_suffix('.codes.pt')
    if not args.offsets or args.offsets[0] != 0 or len(set(args.offsets)) != len(args.offsets) or min(args.offsets)<0: p.error('Offsets must start at zero, be unique and nonnegative')
    if any(x.exists() for x in (out,out.with_suffix('.report.json'),codefile)): raise SystemExit('Output exists')
    source, baseline = load_rows(args.input), load_rows(args.baseline)
    meta = json.loads(Path(args.baseline).with_suffix('.report.json').read_text())
    if (source.keys()!=baseline.keys() or meta['input_sha256']!=sha256_file(args.input)
        or Path(meta['checkpoint']).resolve()!=Path(args.checkpoint).resolve()): raise SystemExit('Baseline provenance differs')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg,model=load_codec(args.config,args.checkpoint,device)
    tok=load_base_tokenizer(cfg.tokenizer); context=int(cfg.model.context_length)
    eos,pad=int(tok.eos_token_id),int(tok.pad_token_id)
    rows=list(source.values()); rows=rows[:args.max_examples] if args.max_examples else rows
    output,codes=[],[]; attempts=0; before_total=after_total=0
    if device.type=='cuda': torch.cuda.synchronize(device)
    start=time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            key,target=row['id'],row['target']
            if baseline[key]['target']!=target: raise SystemExit('Target mismatch')
            tokens=encode_text(tok,target,True)
            if len(tokens)>context: raise SystemExit('Input truncated')
            ids,mask=fixed_sequence(tokens,context,pad,eos)
            actual=torch.tensor(ids,device=device,dtype=torch.long)
            mask=torch.tensor(mask,device=device,dtype=torch.bool)
            best_errors=None; best_text=None; best_codes=None; best_offset=0; before_text=None
            for offset in args.offsets:
                if offset+len(tokens)>context: continue
                # Offset is side metadata; the original token content is only used at the encoder.
                shifted=torch.full((1,context),pad,device=device,dtype=torch.long)
                shifted[0,offset:offset+len(tokens)]=torch.tensor(tokens,device=device)
                idx=model.encode(shifted); prediction=model.decode(idx).argmax(-1)[0,offset:]
                valid=int(mask.sum()); errors=int((prediction[:valid]!=actual[:valid]).sum())
                text=visible_text(tok,prediction.tolist(),eos,pad); attempts+=1
                if offset==0:
                    before_text=text; before_errors=errors
                    if text!=baseline[key]['reconstructed']: raise SystemExit('Baseline mismatch')
                if best_errors is None or errors<best_errors:
                    best_errors,best_text,best_codes,best_offset=errors,text,idx,offset
            verified=model.decode(best_codes).argmax(-1)[0,best_offset:]
            if visible_text(tok,verified.tolist(),eos,pad)!=best_text: raise SystemExit('Selected codes did not decode')
            before_total+=before_errors; after_total+=best_errors
            output.append({'id':key,'selected':best_text,'offset':best_offset,'baseline_exact':before_text==target,
                           'selected_exact':best_text==target,'baseline_errors':before_errors,'selected_errors':best_errors})
            codes.append({'id':key,'indices':best_codes[0].cpu(),'offset':best_offset,'valid_length':valid})
            if len(output)%16==0 or len(output)==len(rows): print(f'Processed {len(output)}/{len(rows)}',flush=True)
    if device.type=='cuda': torch.cuda.synchronize(device)
    summary={'examples':len(output),'baseline_exact':sum(r['baseline_exact'] for r in output),
             'selected_exact':sum(r['selected_exact'] for r in output),'baseline_token_errors':before_total,'selected_token_errors':after_total,
             'gained_exact':sum(not r['baseline_exact'] and r['selected_exact'] for r in output),
             'lost_exact':sum(r['baseline_exact'] and not r['selected_exact'] for r in output),
             'offsets':args.offsets,'attempts':attempts,'seconds':time.perf_counter()-start,
             'input_sha256':sha256_file(args.input),'codec_sha256':sha256_file(args.checkpoint),
             'scope':'Known-input encoder selects offset by target-token error; decoder needs one index stream plus offset and length metadata.',
             'limitations':'Extra encoding and offset metadata; reused sets. Not unknown-target generation, packed size or full runtime.'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8') as f:
        for r in output: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    torch.save(codes,codefile)
    with out.with_suffix('.report.json').open('x',encoding='utf-8') as f: json.dump(summary,f,indent=2); f.write('\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
