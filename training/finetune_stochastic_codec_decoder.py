"""Fine-tune a frozen encoder/codebook decoder on deterministic and sampled codes."""
import argparse
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from evaluation.codec_latent_refiner_screen import load_codec
from training.data import sha256_file
from training.engine import make_loader, seed_all


def validate(model, loader, device):
    model.eval(); model.vq.set_stochastic_encode(False)
    good = count = exact = examples = 0
    with torch.no_grad():
        for row in loader:
            x = row['tokens'].to(device).long(); mask = row['mask'].to(device).bool()
            logits = model.decode(model.encode(x)); pred = logits.argmax(-1)
            good += int(((pred == x) & mask).sum()); count += int(mask.sum())
            exact += int(((pred == x) | ~mask).all(-1).sum()); examples += x.shape[0]
    if not count: raise ValueError('Empty validation')
    return {'token_accuracy':good/count,'exact':exact,'examples':examples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config','checkpoint','output'): p.add_argument('--'+name, required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--temperature',type=float,default=.05); p.add_argument('--top-k',type=int,default=2)
    p.add_argument('--deterministic-only',action='store_true')
    p.add_argument('--validate-every',type=int,default=250); p.add_argument('--seed',type=int,default=42)
    args = p.parse_args(); out = Path(args.output)
    if min(args.steps,args.batch_size,args.top_k,args.validate_every) < 1 or args.lr <= 0 or args.temperature <= 0: p.error('Invalid settings')
    if out.exists(): raise SystemExit('Output exists')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise SystemExit('GPU unavailable')
    cfg, model = load_codec(args.config,args.checkpoint,device)
    manifest = json.loads((Path(cfg.data.cache_dir)/'manifest.json').read_text())
    if manifest['source_sha256'] != sha256_file(cfg.data.input): raise SystemExit('Training cache differs')
    seed_all(args.seed)
    training = make_loader(cfg.data.cache_dir,args.batch_size,0,True,args.seed)
    validation = make_loader(cfg.train.validation_cache_dir,args.batch_size,0,False,args.seed)
    model.decoder.requires_grad_(True)
    training_description = 'decoder-only deterministic codes' if args.deterministic_only else 'decoder-only alternating stochastic codes'
    optimizer = torch.optim.AdamW(model.decoder.parameters(),lr=args.lr)
    baseline = validate(model,validation,device); best = baseline['token_accuracy']; best_step = 0
    print(json.dumps({'step':0,'validation':baseline}),flush=True)
    it = iter(training); started = time.perf_counter()
    out.parent.mkdir(parents=True,exist_ok=True)
    # Save baseline if fine-tuning never improves validation; report selection explicitly.
    torch.save({'model':model.state_dict(),'step':0,'baseline_sha256':sha256_file(args.checkpoint),
                'validation':baseline,'training':training_description},out)
    for step in range(1,args.steps+1):
        try: row=next(it)
        except StopIteration: it=iter(training); row=next(it)
        x = row['tokens'].to(device).long(); mask = row['mask'].to(device).bool()
        with torch.no_grad():
            z = model.encode_z(x)
            model.vq.set_stochastic_encode(not args.deterministic_only and step % 2 == 0,temperature=args.temperature,top_k=args.top_k,top_p=1.)
            indices = model.vq.z_to_idx(z)
            q = model.vq.idx_to_z_q(indices)
        model.vq.set_stochastic_encode(False); model.decoder.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model.decoder(q)
        ce = F.cross_entropy(logits.float().transpose(1,2),x,reduction='none')
        loss = (ce*mask).sum()/mask.sum().clamp_min(1)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite decoder loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),1.0); optimizer.step()
        if step % args.validate_every == 0 or step == args.steps:
            metrics = validate(model,validation,device)
            print(json.dumps({'step':step,'train_ce':float(loss.detach()),'validation':metrics,'seconds':time.perf_counter()-started}),flush=True)
            if metrics['token_accuracy'] > best:
                best,best_step = metrics['token_accuracy'],step
                torch.save({'model':model.state_dict(),'step':step,'baseline_sha256':sha256_file(args.checkpoint),
                            'validation':metrics,'training':training_description},out)
    print(json.dumps({'selected_step':best_step,'baseline_validation':baseline,'selected_validation_accuracy':best,
                      'scope':'Decoder-only training; encoder and codebooks frozen; validation chooses checkpoint.'}),flush=True)


if __name__ == '__main__': main()
