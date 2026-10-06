import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IGNORE_INDEX = 255
DEFAULT_MODEL_ID = "andreribeiro87/unet3plus-efficientnet-kvasir-seg"
BINARY_LABEL_MAP = {
    "_background_": 0,
    "ring": 0,
    "polyp": 1,
    "__ignore__": IGNORE_INDEX,
}
MULTICLASS_LABEL_MAP = {
    "_background_": 0,
    "polyp": 1,
    "ring": 2,
    "__ignore__": IGNORE_INDEX,
}
OVERLAY_COLORS = {
    1: (255, 64, 64),
    2: (64, 200, 255),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_hf_cache_dirs() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    hf_home = ensure_dir(project_root / ".hf_cache")
    ensure_dir(hf_home / "hub")
    ensure_dir(hf_home / "modules")
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_MODULES_CACHE", str(hf_home / "modules"))
    return hf_home


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_pretrained_model(
    model_source: str,
    device: torch.device,
    local_files_only: bool = False,
    config_overrides: Optional[Dict] = None,
    strict: bool = True,
) -> torch.nn.Module:
    hf_home = ensure_hf_cache_dirs()
    model_path = Path(model_source)
    from transformers import AutoModel
    from transformers import AutoConfig
    from transformers import dynamic_module_utils as dmu

    dmu.HF_MODULES_CACHE = str(hf_home / "modules")
    with temporary_model_import_path(model_source):
        if model_path.exists() and model_path.is_dir() and (model_path / "model.safetensors").exists():
            from safetensors.torch import load_file as safe_load_file

            config = AutoConfig.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=True,
            )
            if config_overrides:
                for key, value in config_overrides.items():
                    setattr(config, key, value)
            model = AutoModel.from_config(config, trust_remote_code=True)
            state_dict = safe_load_file(str(model_path / "model.safetensors"))
            if not strict:
                model_state = model.state_dict()
                filtered = {}
                for key, value in state_dict.items():
                    target = model_state.get(key)
                    if target is not None and target.shape == value.shape:
                        filtered[key] = value
                state_dict = filtered
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
            if strict and (missing_keys or unexpected_keys):
                raise RuntimeError(
                    "State dict mismatch while loading local model. "
                    f"missing_keys={missing_keys[:10]}, unexpected_keys={unexpected_keys[:10]}"
                )
        else:
            model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
    model.to(device)
    model.eval()
    return model


@contextmanager
def temporary_model_import_path(model_source: str):
    model_path = Path(model_source)
    inserted = False
    if model_path.exists() and model_path.is_dir():
        resolved = str(model_path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
            inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(resolved)
            except ValueError:
                pass


def validate_local_model_dir(model_dir: Optional[Path]) -> None:
    if model_dir is None:
        return
    candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"No model weights found under {model_dir}. "
            "Run scripts/download_pretrained_model.py first."
        )
    for path in existing:
        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Weight file is incomplete: {path}. "
                "Delete it and re-run scripts/download_pretrained_model.py."
            )


def preprocess_pil_image(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def resize_prob_map(prob_map: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    resized = F.interpolate(
        prob_map.unsqueeze(0).unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def tensor_to_pil_mask(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def overlay_binary_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    overlay[mask > 0] = (
        (1.0 - alpha) * overlay[mask > 0] + alpha * np.array(color, dtype=np.float32)
    )
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def overlay_label_mask(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float = 0.45,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
) -> Image.Image:
    color_map = color_map or OVERLAY_COLORS
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    for class_id, color in color_map.items():
        region = mask == class_id
        if not np.any(region):
            continue
        overlay[region] = (1.0 - alpha) * overlay[region] + alpha * np.array(color, dtype=np.float32)
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_stems(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def iter_image_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        for path in sorted(input_path.glob(suffix)):
            yield path


def iter_image_paths_from_stems(input_dir: Path, stems: List[str]) -> Iterable[Path]:
    suffixes = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for stem in stems:
        found = None
        for suffix in suffixes:
            candidate = input_dir / f"{stem}{suffix}"
            if candidate.exists():
                found = candidate
                break
        if found is not None:
            yield found


def copy_model_artifacts(source_dir: Path, target_dir: Path) -> None:
    ensure_dir(target_dir)
    for name in ["configuration_unet3plus.py", "modeling_unet3plus.py", "loss.py", ".gitattributes", "README.md"]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)
    source_models = source_dir / "models"
    target_models = target_dir / "models"
    if source_models.exists():
        if target_models.exists():
            shutil.rmtree(target_models)
        shutil.copytree(source_models, target_models)
