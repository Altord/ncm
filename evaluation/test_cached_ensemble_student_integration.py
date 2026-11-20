"""Train one tiny CPU student step from a cached teacher and export it."""
import contextlib
import gc
import io
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from omegaconf import OmegaConf
from training.data import save_cache_manifest, sha256_file
import training.train_tokenizer as trainer

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    cache = root/'cache'; cache.mkdir()
    teachers = [root/'teacher-a.pt', root/'teacher-b.pt']
    for i, path in enumerate(teachers): path.write_bytes(f'teacher-{i}'.encode())
    tokens = np.array([[1,2,3,0,0,0,0,0]], dtype=np.int32)
    mask = np.array([[1,1,1,0,0,0,0,0]], dtype=np.uint8)
    indices = np.stack((tokens, np.zeros_like(tokens)), axis=-1)
    probabilities = np.empty((1,8,2), dtype=np.float32); probabilities[:] = [0.8,0.2]
    arrays = {}
    for name, data in [('tokens',tokens), ('mask',mask), ('ensemble_topk_ids',indices), ('ensemble_topk_probs',probabilities)]:
        filename = name+'.bin'; data.tofile(cache/filename)
        arrays[name] = {'file':filename, 'dtype':str(data.dtype), 'shape':list(data.shape[1:])}
    save_cache_manifest(cache, 'tokenizer', 1, arrays, {'ensemble_distillation': {'topk':2,
        'checkpoint_sha256':[sha256_file(p) for p in teachers]}})
    cfg = OmegaConf.create({'task':'test', 'tokenizer':{}, 'model':{'context_length':8}, 'data':{'cache_dir':str(cache)},
        'vqvae':{'in_vocab_size':32, 'context_length':8, 'embed_dim':8, 'vqvae_vocab_size':[16,16], 'n_layers':2,
            'compression_factor':1, 'dropout':0., 'vector_quantizer_config': {'clss':'multiscale_residual_vector_quantizer',
            'decay':.95, 'epsilon':1e-5, 'commitment_cost':.25,
            'levels':{'use_manual_levels':True,'manual_levels':[4,8]},
            'quant_resi':{'enabled':True,'ratio':.5,'share_mode':1},'aux':{'fine_drop':{'prob':0.}}}},
        'train':{'seed':42, 'device':'cpu', 'precision':'fp32', 'batch_size':1, 'grad_accum_steps':1,
            'num_workers':0, 'max_steps':1, 'warmup_steps':0, 'lr':.0003, 'min_lr':.00003,
            'betas':[.9,.95], 'weight_decay':.01, 'grad_clip':1., 'log_every':1, 'save_every':1,
            'run_dir':str(root/'run'), 'artifact_dir':str(root/'artifact'),
            'distillation':{'weight':.1, 'topk':2, 'cached_ensemble':True,
                            'teachers':[{'checkpoint':str(p)} for p in teachers]}}})
    OmegaConf.save(cfg, root/'config.yaml')
    sys.argv = ['test', '--config', str(root/'config.yaml')]
    with contextlib.redirect_stdout(io.StringIO()): trainer.main()
    assert (root/'artifact'/'tokenizer.pt').is_file()
    record = json.loads((root/'run'/'logs'/'train.jsonl').read_text().splitlines()[0])
    assert 'distill_ce' in record and 'teacher_correct_fraction' in record
    gc.collect()
print('Cached ensemble student training checks passed.')
