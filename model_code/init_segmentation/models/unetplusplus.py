import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone
from .blocks import ConvBlock


class UNetPlusPlus(nn.Module):
    """UNet++ with dense nested skip connections.

    Each node x[i][j] aggregates all previous same-scale nodes x[i][0..j-1]
    plus an upsampled feature from the level below x[i+1][j-1], enabling
    the decoder to learn progressively richer skip representations before
    the final prediction at x[0][D].
    See this: https://arxiv.org/abs/1807.10165
    """

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
        skip_channels = list(channels[:-1])  # shallowest → deepest
        bottleneck_ch = channels[-1]
        self._D = len(skip_channels)

        # all_ch[i] = channel count of the raw encoder node x[i][0]
        all_ch = skip_channels + [bottleneck_ch]

        # Dense nodes: x[i][j] for j > 0.
        # x[i][j] concatenates: j same-scale predecessors + 1 upsampled from below.
        # All intermediate node outputs keep skip_channels[i] channels.
        self.nodes = nn.ModuleDict()
        for j in range(1, self._D + 1):
            for i in range(self._D - j + 1):
                # channels coming from same scale: j tensors each with skip_channels[i] channels
                from_same = j * skip_channels[i]
                # channels coming from one level deeper (upsampled)
                from_below = skip_channels[i + 1] if j > 1 else all_ch[i + 1]
                self.nodes[f"{i}_{j}"] = ConvBlock(from_same + from_below, skip_channels[i])

        self.head = nn.Conv2d(skip_channels[0], out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        **_kw,
    ) -> torch.Tensor:
        if pixel_values is not None:
            x = pixel_values
        input_size = x.shape[2:]

        skips, bottleneck = self.backbone(x)
        D = self._D

        # Initialise node cache with encoder outputs
        cache: dict[tuple[int, int], torch.Tensor] = {}
        for i, s in enumerate(skips):
            cache[(i, 0)] = s
        cache[(D, 0)] = bottleneck

        # Fill the dense grid column by column (increasing j)
        for j in range(1, D + 1):
            for i in range(D - j + 1):
                prev = [cache[(i, k)] for k in range(j)]
                target_size = prev[0].shape[2:]
                below = F.interpolate(
                    cache[(i + 1, j - 1)], size=target_size,
                    mode="bilinear", align_corners=False,
                )
                cache[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat(prev + [below], dim=1))

        out = self.head(cache[(0, D)])
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out
