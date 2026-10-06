import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone
from .blocks import UNetDecoderBlock


class UNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "default",
        in_channels: int = 3,
        out_channels: int = 1,
        **_kw,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone_name, in_channels)
        channels = self.backbone.out_channels
        skip_channels, bottleneck_ch = channels[:-1], channels[-1]

        self.decoders = nn.ModuleList()
        in_ch = bottleneck_ch
        for s_ch in reversed(skip_channels):
            self.decoders.append(UNetDecoderBlock(in_ch, s_ch, s_ch))
            in_ch = s_ch

        self.head = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor | None = None, pixel_values: torch.Tensor | None = None, **_kw,
    ) -> torch.Tensor:
        if pixel_values is not None:
            x = pixel_values
        input_size = x.shape[2:]

        skips, x = self.backbone(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        x = self.head(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
