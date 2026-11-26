"""Exercise source-grouped gate calibration and external scoring on a tiny codec."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.latent_refiner import LatentRefiner
from training.data import sha256_file
import evaluation.codec_latent_refiner_gate as screen

class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text): return [2,3,4]
    def decode(self, ids, skip_special_tokens=True): return 'int M() => 1;'

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    cfg = OmegaConf.create({'tokenizer': {}, 'model': {'context_length': 16},
      'vqvae': {'in_vocab_size':32, 'context_length':16, 'embed_dim':8, 'vqvae_vocab_size':[16,16],
       'n_layers':2, 'compression_factor':1, 'dropout':0.0,
       'vector_quantizer_config': {'clss':'multiscale_residual_vector_quantizer','decay':0.95,'epsilon':1e-5,
        'commitment_cost':0.25,'levels':{'use_manual_levels':True,'manual_levels':[4,16]},
        'quant_resi':{'enabled':True,'ratio':0.5,'share_mode':1},'aux':{'fine_drop':{'prob':0.0}}}}})
    config, ckpt, head = root/'config.yaml', root/'codec.pt', root/'adapter.pt'
    OmegaConf.save(cfg, config); torch.save({'model':VQVAE(**cfg.vqvae).state_dict()}, ckpt)
    torch.save({'adapter':LatentRefiner(8,16).state_dict(),'width':8,'hidden':16,
                'codec_sha256':sha256_file(ckpt),'config_sha256':sha256_file(config)}, head)
    sources = {}
    for i in range(100):
        source = f'project/{i}.cs'; sources.setdefault(screen.split_source(source), source)
    assert len(sources) == 2
    target = 'int M() => 1;'
    for name, paths in [('methods', list(sources.values())), ('external', ['external/file.cs'])]:
        src = root/(name+'.jsonl'); base = root/(name+'-base.jsonl'); adapter = root/(name+'-adapter.jsonl')
        rows = [{'id':str(i),'source':path,'target':target} for i,path in enumerate(paths)]
        src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        base.write_text(''.join(json.dumps({'id':r['id'],'target':target,'reconstructed':target})+'\n' for r in rows))
        adapter.write_text(''.join(json.dumps({'id':r['id'],'selected':target})+'\n' for r in rows))
        base.with_suffix('.report.json').write_text(json.dumps({'input_sha256':sha256_file(src),'checkpoint':str(ckpt)}))
        adapter.with_suffix('.report.json').write_text(json.dumps({'input_sha256':sha256_file(src),'codec_sha256':sha256_file(ckpt),'adapter_sha256':sha256_file(head)}))
    screen.load_base_tokenizer = lambda _: Tokenizer()
    sys.argv = ['test','--config',str(config),'--checkpoint',str(ckpt),'--adapter',str(head),
        '--methods-input',str(root/'methods.jsonl'),'--methods-baseline',str(root/'methods-base.jsonl'),
        '--methods-adapter',str(root/'methods-adapter.jsonl'),'--external-input',str(root/'external.jsonl'),
        '--external-baseline',str(root/'external-base.jsonl'),'--external-adapter',str(root/'external-adapter.jsonl'),
        '--output-dir',str(root/'out'),'--device','cpu']
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    report = json.loads((root/'out/report.json').read_text())
    assert report['threshold'] is None and report['calibration']['examples'] == report['methods_evaluation']['examples'] == 1
    assert report['external_evaluation']['selected_exact'] == 1
print('Latent adapter gate integration checks passed.')
