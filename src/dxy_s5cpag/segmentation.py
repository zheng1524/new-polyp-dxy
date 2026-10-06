"""Frozen INIT multiclass inference used for raw-video and Ip8 processing."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from polypseg.common import preprocess_pil_image


def run_segmentation(model, device: torch.device, images: Iterable[tuple[str, Path]],
                     masks_dir: Path, image_size: int = 256, threshold: float = 0.5) -> None:
    """Run the historical argmax INIT inference; `threshold` is retained for API parity."""
    del threshold
    masks_dir.mkdir(parents=True, exist_ok=True)
    for stem, image_path in images:
        output = masks_dir / f"{stem}.png"
        if output.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        tensor = preprocess_pil_image(image, image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(pixel_values=tensor)["logits"].cpu()
        labels = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
        Image.fromarray(torch.argmax(labels, dim=1).squeeze(0).numpy().astype("uint8"), mode="L").save(output)
