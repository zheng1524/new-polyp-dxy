from transformers import PretrainedConfig


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
