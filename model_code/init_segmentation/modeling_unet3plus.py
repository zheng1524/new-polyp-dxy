import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig

from models import create_model


class UNet3PlusConfig(PretrainedConfig):
    model_type = "unet3plus"

    def __init__(
        self,
        backbone: str = "efficientnet",
        in_channels: int = 3,
        out_channels: int = 1,
        inter_ch: int = 64,
        img_size: int = 256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_ch = inter_ch
        self.img_size = img_size


class UNet3PlusForSegmentation(PreTrainedModel):
    """UNet 3+ segmentation model with EfficientNet-B0 backbone.

    Returns a dict with key "logits" (raw sigmoid input, shape B×1×H×W).
    Pass pixel_values as a float32 tensor normalised to [0, 1], shape B×3×H×W.
    """
    config_class = UNet3PlusConfig
    _tied_weights_keys = None
    # Class-level fallback so transformers v5 finalization never triggers
    # nn.Module.__getattr__ for this attribute (instance attr set in __init__ takes priority)
    all_tied_weights_keys = {}

    def __init__(self, config: UNet3PlusConfig):
        super().__init__(config)
        self.model = create_model(
            architecture="unet3plus",
            backbone=config.backbone,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            inter_ch=config.inter_ch,
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        logits = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        return {"loss": loss, "logits": logits}
