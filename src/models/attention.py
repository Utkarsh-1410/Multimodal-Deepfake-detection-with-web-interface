from __future__ import annotations

import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        attn = self.conv(x)  # (B,1,H,W)
        if prior is not None:
            attn = attn + prior  # prior expected (B,1,H,W)
        b, _, h, w = attn.shape
        attn = attn.view(b, 1, h * w)
        attn = torch.softmax(attn, dim=-1)
        attn = attn.view(b, 1, h, w)
        return x * attn


