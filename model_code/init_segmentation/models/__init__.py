import torch.nn as nn

from .attention_unet import AttentionUNet
from .resunet import ResUNet
from .transunet import TransUNet
from .unet import UNet
from .unet3plus import UNet3Plus
from .unetplusplus import UNetPlusPlus

# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

ARCHITECTURES: dict[str, type] = {
    "unet": UNet,
    "unetplusplus": UNetPlusPlus,
    "unet3plus": UNet3Plus,
    "attention_unet": AttentionUNet,
    "resunet": ResUNet,
    "transunet": TransUNet,
}

# Backbones available for all custom hierarchical architectures
BACKBONES: list[str] = ["default", "efficientnet", "convnext", "swin", "siglip"]

# Valid (architecture, backbone) combinations
VALID_COMBINATIONS: dict[str, list[str]] = {
    arch: BACKBONES for arch in ARCHITECTURES
}

# All architecture names
ALL_ARCHITECTURES: list[str] = list(ARCHITECTURES)


# ---------------------------------------------------------------------------
# create_model — unified factory
# ---------------------------------------------------------------------------

def create_model(
    architecture: str,
    backbone: str,
    in_channels: int = 3,
    out_channels: int = 1,
    **kwargs,
) -> nn.Module:
    valid_bbs = VALID_COMBINATIONS.get(architecture)
    if valid_bbs is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from {ALL_ARCHITECTURES}"
        )
    if backbone not in valid_bbs:
        raise ValueError(
            f"Backbone '{backbone}' is not compatible with '{architecture}'. "
            f"Valid choices: {valid_bbs}"
        )
    return ARCHITECTURES[architecture](
        backbone_name=backbone,
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs,
    )
