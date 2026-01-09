from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import SpatialAttention


class SmallAttentionDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
        )
        self.attn = SpatialAttention(128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        feats = self.features(x)
        if prior is not None:
            prior = F.interpolate(prior, size=feats.shape[-2:], mode="bilinear", align_corners=False)
        feats = self.attn(feats, prior)
        logits = self.head(feats)
        return logits


