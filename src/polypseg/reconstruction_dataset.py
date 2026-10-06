import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PARIS_TYPE_TO_MODEL_PREFIX = {
    "Ip": "01p",
    "Is": "01s",
    "IIa": "02a",
    "IIb": "02b",
}


def _normalize_numeric_token(token: str) -> float:
    cleaned = token.strip().lower().replace("mm", "")
    return float(cleaned)


def _format_size_key(value: float) -> str:
    return f"{value:.3f}"


def _choose_primary_blend_path(model_dir: Path) -> Optional[Path]:
    preferred = model_dir / "息肉.blend"
    if preferred.exists():
        return preferred

    candidates = sorted(model_dir.glob("*.blend"))
    if not candidates:
        return None

    for candidate in candidates:
        if "息肉" in candidate.name and "负模" not in candidate.name:
            return candidate
    return candidates[0]


@dataclass
class CaptureFilenameMetadata:
    stem: str
    lesion_code: str
    paris_type: str
    nominal_diameter_mm: float
    stomach_state: str
    ring_code: str
    ring_mode: str
    reference_token: str
    has_reference: bool
    expected_rings: int
    ring_inner_diameters_mm: List[float]
    view_index: int
    capture_group_key: str
    model_prefix: str
    canonical_model_key: str

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["nominal_diameter_mm"] = float(self.nominal_diameter_mm)
        payload["ring_inner_diameters_mm"] = [float(v) for v in self.ring_inner_diameters_mm]
        return payload


@dataclass
class BlenderModelRecord:
    model_key: str
    model_prefix: str
    nominal_diameter_mm: float
    model_dir: Path
    blend_path: Optional[Path]
    source_name: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "model_key": self.model_key,
            "model_prefix": self.model_prefix,
            "nominal_diameter_mm": float(self.nominal_diameter_mm),
            "model_dir": str(self.model_dir),
            "blend_path": str(self.blend_path) if self.blend_path is not None else None,
            "source_name": self.source_name,
        }


def parse_capture_filename(stem: str) -> CaptureFilenameMetadata:
    parts = stem.split("_")
    if len(parts) < 7:
        raise ValueError(
            f"Filename stem does not match expected pattern "
            f"`GP_Ip_12.0mm_closed_R10_single_refY_002`: {stem}"
        )

    lesion_code = parts[0]
    paris_type = parts[1]
    nominal_diameter_mm = _normalize_numeric_token(parts[2])
    stomach_state = parts[3]
    ring_code = parts[4]
    ring_mode = parts[5]
    reference_token = parts[6]
    view_index = -1
    for token in parts[7:]:
        if token.isdigit():
            view_index = int(token)
            break

    has_reference = reference_token.lower() == "refy"
    if not has_reference:
        expected_rings = 0
        ring_inner_diameters_mm: List[float] = []
    elif ring_mode == "double":
        expected_rings = 2
        # 现有命名中 double 统一表示 5 mm 和 10 mm 内径各一个。
        ring_inner_diameters_mm = [5.0, 10.0]
    else:
        expected_rings = 1
        if not ring_code.startswith("R"):
            raise ValueError(f"Invalid ring code in stem: {stem}")
        ring_inner_diameters_mm = [_normalize_numeric_token(ring_code[1:])]

    model_prefix = PARIS_TYPE_TO_MODEL_PREFIX.get(paris_type)
    if model_prefix is None:
        raise ValueError(f"Unsupported Paris type `{paris_type}` in stem: {stem}")

    # Keep every derived frame or duplicate media stem under the same acquisition group.
    capture_group_key = "_".join(parts[:7]) if len(parts) >= 7 else stem
    canonical_model_key = f"{model_prefix}:{_format_size_key(nominal_diameter_mm)}"

    return CaptureFilenameMetadata(
        stem=stem,
        lesion_code=lesion_code,
        paris_type=paris_type,
        nominal_diameter_mm=nominal_diameter_mm,
        stomach_state=stomach_state,
        ring_code=ring_code,
        ring_mode=ring_mode,
        reference_token=reference_token,
        has_reference=has_reference,
        expected_rings=expected_rings,
        ring_inner_diameters_mm=ring_inner_diameters_mm,
        view_index=view_index,
        capture_group_key=capture_group_key,
        model_prefix=model_prefix,
        canonical_model_key=canonical_model_key,
    )


def build_blender_model_registry(blend_root: Path) -> Dict[str, List[BlenderModelRecord]]:
    registry: Dict[str, List[BlenderModelRecord]] = {}
    if not blend_root.exists():
        raise FileNotFoundError(f"Blend root does not exist: {blend_root}")

    for model_dir in sorted(path for path in blend_root.iterdir() if path.is_dir()):
        name = model_dir.name
        if "-" not in name:
            continue

        parts = name.split("-")
        if len(parts) < 2:
            continue

        model_prefix = parts[0]
        try:
            nominal_diameter_mm = _normalize_numeric_token(parts[1])
        except ValueError:
            continue

        model_key = f"{model_prefix}:{_format_size_key(nominal_diameter_mm)}"
        record = BlenderModelRecord(
            model_key=model_key,
            model_prefix=model_prefix,
            nominal_diameter_mm=nominal_diameter_mm,
            model_dir=model_dir,
            blend_path=_choose_primary_blend_path(model_dir),
            source_name=name,
        )
        registry.setdefault(model_prefix, []).append(record)

    for records in registry.values():
        records.sort(key=lambda item: (item.nominal_diameter_mm, item.source_name))
    return registry


def resolve_blender_model(
    metadata: CaptureFilenameMetadata,
    registry: Dict[str, List[BlenderModelRecord]],
    tolerance_mm: float = 0.2,
) -> Optional[BlenderModelRecord]:
    candidates = registry.get(metadata.model_prefix, [])
    if not candidates:
        return None

    best: Optional[BlenderModelRecord] = None
    best_delta = math.inf
    for candidate in candidates:
        delta = abs(candidate.nominal_diameter_mm - metadata.nominal_diameter_mm)
        if delta < best_delta:
            best_delta = delta
            best = candidate

    if best is None or best_delta > tolerance_mm:
        return None
    return best


def build_manifest_records(
    image_root: Path,
    blend_root: Path,
    mask_root: Optional[Path] = None,
    ring_fit_json_dir: Optional[Path] = None,
    tolerance_mm: float = 0.2,
) -> List[Dict[str, object]]:
    registry = build_blender_model_registry(blend_root)
    image_paths = sorted(image_root.glob("*.jpg")) + sorted(image_root.glob("*.jpeg")) + sorted(image_root.glob("*.png"))
    records: List[Dict[str, object]] = []

    for image_path in image_paths:
        meta = parse_capture_filename(image_path.stem)
        model = resolve_blender_model(meta, registry, tolerance_mm=tolerance_mm)
        mask_path = (mask_root / f"{image_path.stem}.png") if mask_root is not None else None
        ring_fit_json_path = (ring_fit_json_dir / f"{image_path.stem}.json") if ring_fit_json_dir is not None else None

        record: Dict[str, object] = {
            **meta.to_dict(),
            "image_path": str(image_path),
            "mask_path": str(mask_path) if mask_path is not None and mask_path.exists() else None,
            "ring_fit_json_path": str(ring_fit_json_path) if ring_fit_json_path is not None and ring_fit_json_path.exists() else None,
            "blender_model_matched": model is not None,
            "blender_model_dir": str(model.model_dir) if model is not None else None,
            "blender_model_blend": str(model.blend_path) if model is not None and model.blend_path is not None else None,
            "blender_model_name": model.source_name if model is not None else None,
            "blender_model_nominal_diameter_mm": float(model.nominal_diameter_mm) if model is not None else None,
        }
        records.append(record)

    capture_group_counts: Dict[str, int] = {}
    model_group_counts: Dict[str, int] = {}
    for record in records:
        capture_group_counts[record["capture_group_key"]] = capture_group_counts.get(record["capture_group_key"], 0) + 1
        model_key = str(record["canonical_model_key"])
        model_group_counts[model_key] = model_group_counts.get(model_key, 0) + 1

    for record in records:
        record["capture_group_size"] = capture_group_counts[record["capture_group_key"]]
        record["model_group_size"] = model_group_counts[str(record["canonical_model_key"])]

    return records


def summarize_manifest(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "num_images": len(records),
        "num_capture_groups": len({record["capture_group_key"] for record in records}),
        "num_canonical_models": len({record["canonical_model_key"] for record in records}),
        "num_with_reference": sum(1 for record in records if record["has_reference"]),
        "num_without_reference": sum(1 for record in records if not record["has_reference"]),
        "num_with_blender_match": sum(1 for record in records if record["blender_model_matched"]),
    }

    paris_type_counts: Dict[str, int] = {}
    ring_mode_counts: Dict[str, int] = {}
    for record in records:
        paris_type = str(record["paris_type"])
        ring_mode = str(record["ring_mode"])
        paris_type_counts[paris_type] = paris_type_counts.get(paris_type, 0) + 1
        ring_mode_counts[ring_mode] = ring_mode_counts.get(ring_mode, 0) + 1

    summary["paris_type_counts"] = paris_type_counts
    summary["ring_mode_counts"] = ring_mode_counts
    return summary


def _partition_groups(
    group_to_stems: Dict[str, List[str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    groups = list(group_to_stems.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    total_images = sum(len(stems) for _, stems in groups)
    target_val = total_images * val_ratio
    target_test = total_images * test_ratio

    split_groups = {"train": [], "val": [], "test": []}
    split_sizes = {"train": 0, "val": 0, "test": 0}

    for group_key, stems in groups:
        current_total = split_sizes["train"] + split_sizes["val"] + split_sizes["test"]
        remaining = total_images - current_total

        if split_sizes["val"] < target_val:
            split_name = "val"
        elif split_sizes["test"] < target_test:
            split_name = "test"
        else:
            split_name = "train"

        if remaining == len(stems) and split_name != "train":
            split_name = "train"

        split_groups[split_name].extend(sorted(stems))
        split_sizes[split_name] += len(stems)

    for split_name in split_groups:
        split_groups[split_name].sort()
    return split_groups


def build_group_splits(
    records: Sequence[Dict[str, object]],
    group_key: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    if group_key not in {"capture_group_key", "canonical_model_key", "blender_model_name"}:
        raise ValueError(f"Unsupported split group key: {group_key}")

    group_to_stems: Dict[str, List[str]] = {}
    for record in records:
        key = str(record[group_key])
        group_to_stems.setdefault(key, []).append(str(record["stem"]))

    return _partition_groups(group_to_stems, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


def write_manifest_json(path: Path, records: Sequence[Dict[str, object]], summary: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "records": list(records),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            if isinstance(row.get("ring_inner_diameters_mm"), list):
                row["ring_inner_diameters_mm"] = json.dumps(row["ring_inner_diameters_mm"], ensure_ascii=False)
            writer.writerow(row)


def write_split_files(output_dir: Path, splits: Dict[str, Iterable[str]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, stems in splits.items():
        lines = [str(stem) for stem in stems]
        (output_dir / f"{split_name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
