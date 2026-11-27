"""Check deterministic sampling and frozen-model cache metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from models.vqvae import VQVAE
from models.latent_refiner import LatentRefiner
from evaluation.codec_cache_generalization import sample_indices, measure

assert sample_indices(10, 3, 42) == sample_indices(10, 3, 42)
assert len(set(sample_indices(10, 3, 42))) == 3
assert sample_indices(3, 10, 42) == [0,1,2]
cfg = OmegaConf.create({'in_vocab_size': 32, 'context_length': 16, 'embed_dim': 8,
 'vqvae_vocab_size': [16,16], 'n_layers':2, 'compression_factor':1, 'dropout':0.0,
 'vector_quantizer_config': {'clss':'multiscale_residual_vector_quantizer','decay':0.95,'epsilon':1e-5,
  'commitment_cost':0.25,'levels':{'use_manual_levels':True,'manual_levels':[4,16]},
  'quant_resi':{'enabled':True,'ratio':0.5,'share_mode':1},'aux':{'fine_drop':{'prob':0.0}}}})
model = VQVAE(**cfg).eval().requires_grad_(False); adapter = LatentRefiner(8,16).eval()
x = torch.randint(0,32,(2,16)); mask = torch.tensor([[1]*10+[0]*6,[1]*8+[0]*8], dtype=torch.bool)
result = measure(model, adapter, [{'tokens':x,'mask':mask}], torch.device('cpu'))
assert result['examples'] == 2 and result['valid_tokens'] == 18
assert result['codec'] == result['adapter']
print('Codec cache generalization checks passed.')
