from __future__ import annotations

import torch
import onnx

from src.models.discriminator import SmallAttentionDiscriminator


def export_discriminator_onnx(path: str, opset: int = 17) -> None:
    model = SmallAttentionDiscriminator().eval()
    x = torch.randn(1, 3, 224, 224)
    prior = torch.randn(1, 1, 224, 224)
    torch.onnx.export(
        model,
        (x, prior),
        path,
        input_names=["image", "prior"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes={"image": {0: "batch"}, "prior": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx.load(path)  # validate loadable


if __name__ == "__main__":
    export_discriminator_onnx("exports/discriminator.onnx", opset=17)


