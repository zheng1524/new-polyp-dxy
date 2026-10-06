import ssl
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    Swin_T_Weights,
    convnext_tiny,
    efficientnet_b0,
    swin_t,
)
from torchvision.models.feature_extraction import create_feature_extractor

from .blocks import ConvBlock


def _disable_ssl_verification():
    """Create an unverified SSL context for downloading pretrained weights."""
    return ssl._create_unverified_context()


# Workaround for SSL certificate verification issues when downloading pretrained weights
_original_create_default_https_context = ssl._create_default_https_context


def _enable_unverified_ssl():
    ssl._create_default_https_context = _disable_ssl_verification


def _restore_ssl():
    ssl._create_default_https_context = _original_create_default_https_context


class Backbone(ABC):
    """All backbones return (skip_features high→low, bottleneck)."""

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Channel counts from highest-res skip to bottleneck (last element)."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        ...


class DefaultBackbone(Backbone, nn.Module):
    """Plain conv encoder identical to the classic U-Net."""

    def __init__(self, in_channels: int = 3, block_cls: type | None = None, **_kw):
        nn.Module.__init__(self)
        block = block_cls or ConvBlock
        self.enc1 = block(in_channels, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.enc4 = block(256, 512)
        self.bottleneck = block(512, 1024)
        self.pool = nn.MaxPool2d(2, 2)

    @property
    def out_channels(self) -> list[int]:
        return [64, 128, 256, 512, 1024]

    def forward(self, x: torch.Tensor):
        skips = []
        for enc in [self.enc1, self.enc2, self.enc3, self.enc4]:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        return skips, self.bottleneck(x)


class EfficientNetBackbone(Backbone, nn.Module):
    """EfficientNet-B0 pretrained encoder (ImageNet)."""

    _RETURN_NODES = {
        "features.1": "s1",   # H/2,  16 ch
        "features.2": "s2",   # H/4,  24 ch
        "features.3": "s3",   # H/8,  40 ch
        "features.5": "s4",   # H/16, 112 ch
        "features.8": "bottleneck",  # H/32, 1280 ch
    }

    def __init__(self, in_channels: int = 3, **_kw):
        nn.Module.__init__(self)
        base = efficientnet_b0(weights=None)
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels, old.kernel_size,
                old.stride, old.padding, bias=False,
            )
        self.body = create_feature_extractor(base, return_nodes=self._RETURN_NODES)

    @property
    def out_channels(self) -> list[int]:
        return [16, 24, 40, 112, 1280]

    def forward(self, x: torch.Tensor):
        f = self.body(x)
        return [f["s1"], f["s2"], f["s3"], f["s4"]], f["bottleneck"]


class ConvNeXTBackbone(Backbone, nn.Module):
    """ConvNeXT-Tiny pretrained encoder (ImageNet)."""

    _RETURN_NODES = {
        "features.1": "s1",   # H/4,  96 ch
        "features.3": "s2",   # H/8,  192 ch
        "features.5": "s3",   # H/16, 384 ch
        "features.7": "bottleneck",  # H/32, 768 ch
    }

    def __init__(self, in_channels: int = 3, **_kw):
        nn.Module.__init__(self)
        base = convnext_tiny(weights=None)
        if in_channels != 3:
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels, old.kernel_size,
                old.stride, old.padding, bias=False,
            )
        self.body = create_feature_extractor(base, return_nodes=self._RETURN_NODES)

    @property
    def out_channels(self) -> list[int]:
        return [96, 192, 384, 768]

    def forward(self, x: torch.Tensor):
        f = self.body(x)
        return [f["s1"], f["s2"], f["s3"]], f["bottleneck"]


class SwinBackbone(Backbone, nn.Module):
    """Swin Transformer Tiny pretrained encoder (ImageNet).

    Produces hierarchical features at four scales — identical channel widths
    to ConvNeXt-Tiny (96 → 192 → 384 → 768) — so it slots into every
    existing decoder without any code changes.

    Input (B, 3, H, W) → skips [(B,96,H/4,W/4), (B,192,H/8,W/8),
                                  (B,384,H/16,W/16)], bottleneck (B,768,H/32,W/32)

    Note: torchvision Swin outputs tensors in (B, H, W, C) layout;
    the backbone permutes them to the standard (B, C, H, W) before returning.
    """

    _RETURN_NODES = {
        "features.1": "s1",         # H/4,  96 ch
        "features.3": "s2",         # H/8,  192 ch
        "features.5": "s3",         # H/16, 384 ch
        "features.7": "bottleneck", # H/32, 768 ch
    }

    def __init__(self, in_channels: int = 3, **_kw):
        nn.Module.__init__(self)
        base = swin_t(weights=None)
        if in_channels != 3:
            # Replace the first patch-embedding conv
            old = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size, stride=old.stride,
                padding=old.padding, bias=False,
            )
        self.body = create_feature_extractor(base, return_nodes=self._RETURN_NODES)

    @property
    def out_channels(self) -> list[int]:
        return [96, 192, 384, 768]

    def forward(self, x: torch.Tensor):
        f = self.body(x)
        # Swin-T stores features as (B, H, W, C) → convert to (B, C, H, W)
        s1 = f["s1"].permute(0, 3, 1, 2).contiguous()
        s2 = f["s2"].permute(0, 3, 1, 2).contiguous()
        s3 = f["s3"].permute(0, 3, 1, 2).contiguous()
        bn = f["bottleneck"].permute(0, 3, 1, 2).contiguous()
        return [s1, s2, s3], bn


class SigLIPBackbone(Backbone, nn.Module):
    """SigLIP-Base/16 (93 M) pretrained vision encoder as a flat-ViT backbone.

    Gemma 3 uses the larger ``google/siglip-so400m-patch14-384`` (400 M); this
    class defaults to the practical ``google/siglip-base-patch16-224`` variant
    (93 M) that fits comfortably alongside a UNet decoder. Swap MODEL_ID for
    the Gemma 3 variant when VRAM permits.

    Architecture note
    -----------------
    SigLIP is a pure Vision Transformer — every layer produces tokens at the
    **same** spatial resolution (H/16 × W/16 = 14×14 for 224-px input). This
    backbone therefore returns four feature maps that are **all at 14×14** but
    at different semantic depths (layers 3 / 6 / 9 / 12 of the 12-layer ViT):

        [z3, z6, z9]  →  skips  (each B, 768, 14, 14)
        z12           →  bottleneck (B, 768, 14, 14)

    This is intentionally different from the hierarchical CNN/Swin backbones.
    Use the ``ViTUNet`` architecture (``models/vit_unet.py``) which handles the
    flat-resolution skips by projecting and bilinearly upsampling them to match
    each decoder stage. Standard UNet/AttentionUNet/ResUNet/TransUNet decoders
    will NOT work correctly with this backbone.

    Gemma 3 variant
    ---------------
    Replace MODEL_ID with ``"google/siglip-so400m-patch14-384"`` and set
    ``PATCH_SIZE = 14``, ``INPUT_SIZE = 384`` to use the exact Gemma 3 encoder.
    You will also need to adjust ``EXTRACT_LAYERS`` (the model has 27 layers).
    """

    MODEL_ID = "google/siglip-base-patch16-224"
    PATCH_SIZE = 16
    INPUT_SIZE = 224
    # EXTRACT_LAYERS: quarter-intervals of the 12-layer ViT (1-indexed after embedding)
    EXTRACT_LAYERS = (3, 6, 9, 12)

    def __init__(self, in_channels: int = 3, **_kw):
        nn.Module.__init__(self)
        # google/siglip-base-patch16-224 is a full CLIP-style model (vision +
        # text).  Loading with SiglipVisionModel.from_pretrained fails because
        # the repo config is a SiglipConfig, not a SiglipVisionConfig.
        # Solution: load the full SiglipModel and keep only the vision encoder.
        from transformers import SiglipModel

        full = SiglipModel.from_pretrained(self.MODEL_ID)
        self._vit = full.vision_model        # SiglipVisionTransformer
        self._hidden_dim: int = full.config.vision_config.hidden_size   # 768
        del full                             # free text encoder weights

        self._patch_grid = self.INPUT_SIZE // self.PATCH_SIZE           # 14

        if in_channels != 3:
            old = self._vit.embeddings.patch_embedding
            self._vit.embeddings.patch_embedding = nn.Conv2d(
                in_channels, self._hidden_dim,
                kernel_size=self.PATCH_SIZE, stride=self.PATCH_SIZE, bias=False,
            )

        # Register forward hooks on the target transformer layers to capture
        # intermediate hidden states.  This is version-agnostic — it works
        # regardless of whether the transformers library honours
        # `output_hidden_states=True` via its **kwargs API.
        self._hooked: dict[int, torch.Tensor] = {}
        self._hook_handles: list = []
        for layer_idx in self.EXTRACT_LAYERS:
            layer = self._vit.encoder.layers[layer_idx - 1]  # 0-indexed
            handle = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hook_handles.append(handle)

    def _make_hook(self, idx: int):
        def _hook(module, inp, out):
            # SiglipEncoderLayer returns a tuple; the first element is the
            # hidden state tensor (B, N_patches, hidden_dim).
            self._hooked[idx] = out[0] if isinstance(out, tuple) else out
        return _hook

    @property
    def out_channels(self) -> list[int]:
        return [self._hidden_dim] * len(self.EXTRACT_LAYERS)  # [768, 768, 768, 768]

    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        if x.shape[2] != self.INPUT_SIZE or x.shape[3] != self.INPUT_SIZE:
            x = F.interpolate(
                x, size=(self.INPUT_SIZE, self.INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )

        self._hooked.clear()
        self._vit(pixel_values=x)     # hooks populate self._hooked

        G = self._patch_grid           # 14
        features = []
        for layer_idx in self.EXTRACT_LAYERS:
            hs = self._hooked[layer_idx]   # (B, 196, 768)
            feat = (
                hs.reshape(B, G, G, self._hidden_dim)
                .permute(0, 3, 1, 2)
                .contiguous()
            )                              # (B, 768, 14, 14)
            features.append(feat)

        # z3=features[0] (early/local), z12=features[3] (final/global)
        return features[:-1], features[-1]  # skips, bottleneck


BACKBONE_REGISTRY: dict[str, type] = {
    "default": DefaultBackbone,
    "efficientnet": EfficientNetBackbone,
    "convnext": ConvNeXTBackbone,
    "swin": SwinBackbone,
    "siglip": SigLIPBackbone,
}


def create_backbone(name: str, in_channels: int = 3, **kwargs) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(BACKBONE_REGISTRY)}")
    return BACKBONE_REGISTRY[name](in_channels=in_channels, **kwargs)
