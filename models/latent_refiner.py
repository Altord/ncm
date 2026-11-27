"""Small residual adapter that consumes decoded VQ latents only."""
import torch
from torch import nn


class LatentRefiner(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, quantized):
        return quantized + self.net(quantized)


class ContextualLatentRefiner(nn.Module):
    """Residual self-attention followed by the original per-position adapter."""
    def __init__(self, width, hidden, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('Adapter width must be divisible by attention heads')
        self.norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        self.positionwise = LatentRefiner(width, hidden)

    def forward(self, quantized):
        inputs = self.norm(quantized)
        attended = self.attention(inputs, inputs, inputs, need_weights=False)[0]
        return self.positionwise(quantized + attended)


def make_latent_refiner(width, hidden, architecture='mlp', heads=4):
    if architecture == 'mlp': return LatentRefiner(width, hidden)
    if architecture == 'attention': return ContextualLatentRefiner(width, hidden, heads)
    raise ValueError(f'Unknown latent adapter architecture: {architecture}')
