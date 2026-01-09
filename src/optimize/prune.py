from __future__ import annotations

import torch
import torch.nn.utils.prune as prune

from src.models.discriminator import SmallAttentionDiscriminator


def prune_discriminator(amount: float = 0.2) -> torch.nn.Module:
    model = SmallAttentionDiscriminator().eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


if __name__ == "__main__":
    m = prune_discriminator(0.2)
    print("Pruned model ready")


