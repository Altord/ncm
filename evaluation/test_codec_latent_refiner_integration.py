"""Exercise training, validation selection, artifact loading and paired scoring."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from training.data import save_cache_manifest, sha256_file
import evaluation.codec_latent_refiner_screen as screen

class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text): return [2, 3, 4]
    def decode(self, ids, skip_special_tokens=True): return 'int M() => 1;'

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    for name in ('train', 'val'):
        cache = root/name; cache.mkdir()
        np.array([[2,3,4,1]+[0]*12, [2,3,4,1]+[0]*12], dtype=np.int32).tofile(cache/'tokens.bin')
        np.array([[1]*4+[0]*12, [1]*4+[0]*12], dtype=np.uint8).tofile(cache/'mask.bin')
        save_cache_manifest(cache, 'tokenizer', 2, {'tokens': {'file':'tokens.bin','dtype':'int32','shape':[16]},
                                                  'mask': {'file':'mask.bin','dtype':'uint8','shape':[16]}}, {})
    cfg = OmegaConf.create({'tokenizer': {}, 'model': {'context_length': 16}, 'data': {'cache_dir': str(root/'train')},
      'train': {'seed': 42, 'validation_cache_dir': str(root/'val')},
      'vqvae': {'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8, 'vqvae_vocab_size': [16,16],
       'n_layers': 2, 'compression_factor': 1, 'dropout': 0.0,
       'vector_quantizer_config': {'clss':'multiscale_residual_vector_quantizer', 'decay':0.95, 'epsilon':1e-5,
        'commitment_cost':0.25, 'levels':{'use_manual_levels':True,'manual_levels':[4,16]},
        'quant_resi':{'enabled':True,'ratio':0.5,'share_mode':1}, 'aux':{'fine_drop':{'prob':0.0}}}}})
    config, checkpoint, source = root/'config.yaml', root/'codec.pt', root/'source.jsonl'
    OmegaConf.save(cfg, config); torch.save({'model':VQVAE(**cfg.vqvae).state_dict()}, checkpoint)
    target = 'int M() => 1;'; source.write_text(json.dumps({'id':'one','target':target})+'\n')
    (root/'baseline.jsonl').write_text(json.dumps({'id':'one','target':target,'reconstructed':target})+'\n')
    (root/'baseline.report.json').write_text(json.dumps({'input_sha256':sha256_file(source),'checkpoint':str(checkpoint)}))
    screen.load_base_tokenizer = lambda _: Tokenizer()
    common = ['--config',str(config),'--checkpoint',str(checkpoint),'--device','cpu','--batch-size','2']
    sys.argv = ['test','--mode','train',*common,'--output',str(root/'head.pt'),'--steps','1','--validate-every','1']
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    sys.argv = ['test','--mode','evaluate',*common,'--adapter',str(root/'head.pt'),'--input',str(source),
                '--baseline',str(root/'baseline.jsonl'),'--output',str(root/'result.jsonl')]
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    report = json.loads((root/'result.report.json').read_text())
    assert report['examples'] == 1 and report['adapter_step'] == 1
    assert report['baseline_exact'] == report['adapter_exact'] == 1
    sys.argv = ['test','--mode','train',*common,'--output',str(root/'attention.pt'),'--steps','1',
                '--validate-every','1','--architecture','attention','--heads','2']
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    assert torch.load(root/'attention.pt', weights_only=False)['architecture'] == 'attention'
    sys.argv = ['test','--mode','evaluate',*common,'--adapter',str(root/'attention.pt'),'--input',str(source),
                '--baseline',str(root/'baseline.jsonl'),'--output',str(root/'attention-result.jsonl')]
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    report = json.loads((root/'attention-result.report.json').read_text())
    assert report['examples'] == report['adapter_exact'] == 1
    sys.argv = ['test','--mode','train',*common,'--output',str(root/'latent.pt'),'--steps','1',
                '--validate-every','1','--ce-weight','0','--latent-weight','1']
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    assert torch.load(root/'latent.pt', weights_only=False)['ce_weight'] == 0
print('Latent refiner training and evaluation integration checks passed.')
