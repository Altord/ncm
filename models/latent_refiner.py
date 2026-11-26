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
