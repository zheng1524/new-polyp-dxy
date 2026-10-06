import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone
from .blocks import TransformerBlock, UNetDecoderBlock


class TransUNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "default",
        in_channels: int = 3,
        out_channels: int = 1,
        img_size: int = 256,
        transformer_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        **_kw,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone_name, in_channels)
        channels = self.backbone.out_channels
        skip_channels, bottleneck_ch = channels[:-1], channels[-1]

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            _, dummy_bn = self.backbone(dummy)
            num_patches = dummy_bn.shape[2] * dummy_bn.shape[3]

        self.proj_in = nn.Conv2d(bottleneck_ch, transformer_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, transformer_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.transformer = nn.Sequential(
            *[TransformerBlock(transformer_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.proj_out = nn.Conv2d(transformer_dim, bottleneck_ch, kernel_size=1)

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

        skips, bottleneck = self.backbone(x)

        B, _C, H, W = bottleneck.shape
        t = self.proj_in(bottleneck)               # B, D, H, W
        t = t.flatten(2).transpose(1, 2)           # B, N, D
        t = t + self.pos_embed
        t = self.transformer(t)                    # B, N, D
        t = t.transpose(1, 2).view(B, -1, H, W)   # B, D, H, W
        x = self.proj_out(t)                       # B, bottleneck_ch, H, W

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        x = self.head(x)
        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
