from __future__ import annotations

import torch
import torch.nn as nn


class SimplePerturbationGenerator(nn.Module):
    """
    Lightweight generator that applies learnable frequency-domain and color perturbations
    to increase difficulty for the discriminator. Not a full image-synthesis GAN.
    """

    def __init__(self, strength: float = 0.1) -> None:
        super().__init__()
        self.strength = nn.Parameter(torch.tensor(strength))
        self.color = nn.Sequential(
            nn.Conv2d(3, 16, 1), nn.ReLU(), nn.Conv2d(16, 3, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.color(x)
        return torch.clamp(x + self.strength.tanh() * y, 0.0, 1.0)


