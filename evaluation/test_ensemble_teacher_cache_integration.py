"""Build an ensemble cache from tiny codecs and verify the source stays intact."""
import contextlib
import gc
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
from training.data import MemmapCache, save_cache_manifest, sha256_file
import training.prepare_ensemble_teacher_cache as screen

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    source = root/'source.jsonl'
    source.write_text('{"text":"x"}\n')
    cache = root/'cache'
    cache.mkdir()
    np.array([[1,2,3,0,0,0,0,0]], dtype=np.int32).tofile(cache/'tokens.bin')
    np.array([[1,1,1,0,0,0,0,0]], dtype=np.uint8).tofile(cache/'mask.bin')
    np.ones((1,8), dtype=np.float32).tofile(cache/'token_weights.bin')
    save_cache_manifest(cache, 'tokenizer', 1, {
        'tokens': {'file':'tokens.bin', 'dtype':'int32', 'shape':[8]},
        'mask': {'file':'mask.bin', 'dtype':'uint8', 'shape':[8]},
        'token_weights': {'file':'token_weights.bin', 'dtype':'float32', 'shape':[8]}},
        {'source_sha256':sha256_file(source), 'sequence_length':8})
    cfg = OmegaConf.create({'tokenizer':{}, 'model':{'context_length':8},
        'data':{'input':str(source), 'cache_dir':str(cache)},
        'vqvae':{'in_vocab_size':32, 'context_length':8, 'embed_dim':8, 'vqvae_vocab_size':[16,16],
            'n_layers':2, 'compression_factor':1, 'dropout':0.,
            'vector_quantizer_config': {'clss':'multiscale_residual_vector_quantizer', 'decay':.95, 'epsilon':1e-5,
                'commitment_cost':.25, 'levels':{'use_manual_levels':True, 'manual_levels':[4,8]},
                'quant_resi':{'enabled':True, 'ratio':.5, 'share_mode':1}, 'aux':{'fine_drop':{'prob':0.}}}}})
    before = sha256_file(cache/'tokens.bin')
    for name in ('a', 'b'):
        OmegaConf.save(cfg, root/f'{name}.yaml')
        with contextlib.redirect_stdout(io.StringIO()): model = VQVAE(**cfg.vqvae)
        torch.save({'model':model.state_dict()}, root/f'{name}.pt')
    sys.argv = ['test', '--config-a',str(root/'a.yaml'), '--checkpoint-a',str(root/'a.pt'),
                '--config-b',str(root/'b.yaml'), '--checkpoint-b',str(root/'b.pt'),
                '--output',str(root/'out'), '--topk','2', '--batch-size','1', '--device','cpu']
    with contextlib.redirect_stdout(io.StringIO()): screen.main()
    result = MemmapCache(root/'out')
    assert result[0]['ensemble_topk_ids'].shape == (8,2)
    assert torch.allclose(result[0]['ensemble_topk_probs'].sum(-1), torch.ones(8), atol=1e-6)
    assert result.manifest['ensemble_distillation']['checkpoint_sha256'] == [sha256_file(root/'a.pt'), sha256_file(root/'b.pt')]
    assert sha256_file(cache/'tokens.bin') == before == sha256_file(root/'out'/'tokens.bin')
    del result
    gc.collect()
print('Ensemble teacher-cache integration checks passed.')
