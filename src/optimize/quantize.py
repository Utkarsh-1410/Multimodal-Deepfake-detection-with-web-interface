from __future__ import annotations

import torch
from torch.ao.quantization import quantize_dynamic

from src.models.discriminator import SmallAttentionDiscriminator


def dynamic_quantize_discriminator() -> torch.nn.Module:
    model = SmallAttentionDiscriminator().eval()
    qmodel = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return qmodel


if __name__ == "__main__":
    qm = dynamic_quantize_discriminator()
    print(qm)


