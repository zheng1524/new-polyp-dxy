import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import create_backbone
from .blocks import ConvBlock


class UNet3Plus(nn.Module):
    """UNet 3+ with full-scale skip connections.

    Every decoder node aggregates features from ALL encoder levels (not just
    the matching scale), giving each node simultaneous access to fine-grained
    detail and deep semantic context. Each incoming stream is projected to
    `inter_ch` channels before concatenation so total input channels are
    predictable regardless of backbone channel widths.

    Reference: Huang et al., "UNet 3+: A Full-Scale Connected UNet for
    Medical Image Segmentation", ICASSP 2020.
    """

    def __init__(
        self,
        backbone_name: str = "default",
        in_channels: int = 3,
        out_channels: int = 1,
        inter_ch: int = 64,
        **_kw,
    ):
        super().__init__()
        self.backbone = create_backbone(backbone_name, in_channels)
        all_ch = list(self.backbone.out_channels)  # [skip0, ..., skip_{D-1}, bottleneck]
        D = len(all_ch) - 1                        # number of skip levels = decoder levels
        self._D = D
        self._inter_ch = inter_ch
        self._dec_ch = (D + 1) * inter_ch          # fixed output channels for every decoder node

        # --- projections: each incoming stream is independently projected to inter_ch ---
        # enc_projs[k][j]: encoder/bottleneck level j → inter_ch, for use at decoder k
        self.enc_projs = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(all_ch[j], inter_ch, 1, bias=False),
                    nn.BatchNorm2d(inter_ch),
                    nn.ReLU(inplace=True),
                )
                for j in range(D + 1)
            ])
            for k in range(D)
        ])

        # dec_projs[k][i]: prior decoder level i → inter_ch, for use at decoder k  (i < k)
        # All prior decoder outputs have self._dec_ch channels.
        self.dec_projs = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self._dec_ch, inter_ch, 1, bias=False),
                    nn.BatchNorm2d(inter_ch),
                    nn.ReLU(inplace=True),
                )
                for _ in range(k)           # one projection per prior decoder level
            ])
            for k in range(D)
        ])

        # Fusion conv at each decoder level: (D+1+k)*inter_ch → dec_ch
        self.dec_convs = nn.ModuleList([
            ConvBlock((D + 1 + k) * inter_ch, self._dec_ch)
            for k in range(D)
        ])

        self.head = nn.Conv2d(self._dec_ch, out_channels, kernel_size=1)

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
        # enc_feats: shallowest encoder first, bottleneck last  (D+1 tensors)
        enc_feats = list(skips) + [bottleneck]

        dec_outputs: list[torch.Tensor] = []

        # Build decoder from deepest (k=0) to shallowest (k=D-1).
        # Decoder k targets the same spatial resolution as skips[D-1-k].
        for k in range(D):
            target = skips[D - 1 - k].shape[2:]
            parts: list[torch.Tensor] = []

            # All encoder / bottleneck streams
            for j, feat in enumerate(enc_feats):
                feat = _resize(feat, target)
                parts.append(self.enc_projs[k][j](feat))

            # All prior decoder streams (deeper → current scale = always upsample)
            for i, df in enumerate(dec_outputs):
                feat = F.interpolate(df, size=target, mode="bilinear", align_corners=False)
                parts.append(self.dec_projs[k][i](feat))

            dec_outputs.append(self.dec_convs[k](torch.cat(parts, dim=1)))

        out = self.head(dec_outputs[-1])
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out


def _resize(feat: torch.Tensor, target: tuple[int, int] | torch.Size) -> torch.Tensor:
    if feat.shape[2:] == torch.Size(target):
        return feat
    if feat.shape[2] > target[0]:
        return F.adaptive_max_pool2d(feat, target)
    return F.interpolate(feat, size=target, mode="bilinear", align_corners=False)
