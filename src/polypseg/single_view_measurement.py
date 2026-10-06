import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .calibration import load_calibration, undistort_image, undistort_mask
from .reconstruction_dataset import parse_capture_filename


POLYP_CLASS_ID = 1


@dataclass
class MeasurementResult:
    stem: str
    status: str
    reason: str
    metrics: Dict[str, object]


@dataclass
class EllipseParams:
    cx: float
    cy: float
    major_radius: float
    minor_radius: float
    angle_deg: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def load_label_mask(mask_path: Path) -> np.ndarray:
    return np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)


def sample_ellipse_points(params: EllipseParams, num_points: int = 360) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * math.pi, num=num_points, endpoint=False, dtype=np.float64)
    angle = math.radians(params.angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    xs = params.major_radius * np.cos(theta)
    ys = params.minor_radius * np.sin(theta)
    xr = cos_a * xs - sin_a * ys + params.cx
    yr = sin_a * xs + cos_a * ys + params.cy
    return np.column_stack([xr, yr]).astype(np.float32)


def transform_points(points_xy: np.ndarray, matrix_3x3: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    warped = homog @ matrix_3x3.T
    warped_xy = warped[:, :2] / np.clip(warped[:, 2:3], 1e-8, None)
    return warped_xy.astype(np.float32)


def _mean_angle_deg(angles_deg: Sequence[float], weights: Sequence[float]) -> float:
    if not angles_deg:
        return 0.0
    xs = 0.0
    ys = 0.0
    for angle_deg, weight in zip(angles_deg, weights):
        theta = math.radians(angle_deg * 2.0)
        xs += float(weight) * math.cos(theta)
        ys += float(weight) * math.sin(theta)
    return math.degrees(math.atan2(ys, xs) / 2.0)


def ellipse_from_dict(payload: Optional[Dict[str, float]]) -> Optional[EllipseParams]:
    if payload is None:
        return None
    return EllipseParams(**payload)


def choose_polyp_component(polyp_mask: np.ndarray, reference_center_xy: Optional[Tuple[float, float]], min_area: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    mask_u8 = (polyp_mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    components: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        centroid = centroids[idx].astype(np.float32)
        component = labels == idx
        if reference_center_xy is None:
            score = float(area)
        else:
            dist = float(np.linalg.norm(centroid - np.asarray(reference_center_xy, dtype=np.float32)))
            score = float(area) / max(dist, 1.0)
        components.append((score, component, centroid))

    if not components:
        raise ValueError("No polyp component survived filtering.")

    components.sort(key=lambda item: item[0], reverse=True)
    _, component, centroid = components[0]
    return component, centroid


def contour_from_mask(mask_bool: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask.")
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2).astype(np.float32)


def fit_ellipse_to_points(points_xy: np.ndarray) -> Optional[EllipseParams]:
    if len(points_xy) < 5:
        return None
    ellipse = cv2.fitEllipse(points_xy.reshape(-1, 1, 2).astype(np.float32))
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    if axis_a >= axis_b:
        major_radius = axis_a / 2.0
        minor_radius = axis_b / 2.0
        major_angle = angle_deg
    else:
        major_radius = axis_b / 2.0
        minor_radius = axis_a / 2.0
        major_angle = angle_deg + 90.0
    return EllipseParams(
        cx=float(cx),
        cy=float(cy),
        major_radius=float(major_radius),
        minor_radius=float(minor_radius),
        angle_deg=float(major_angle),
    )


def polyline_perimeter(points_xy: np.ndarray) -> float:
    if len(points_xy) < 2:
        return 0.0
    closed = np.vstack([points_xy, points_xy[:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def contour_area(points_xy: np.ndarray) -> float:
    return float(cv2.contourArea(points_xy.reshape(-1, 1, 2).astype(np.float32)))


def build_affine_rectifier(inner_ellipse: EllipseParams, outer_ellipse: Optional[EllipseParams]) -> Tuple[np.ndarray, Dict[str, float]]:
    ellipses = [inner_ellipse]
    if outer_ellipse is not None:
        ellipses.append(outer_ellipse)

    weights = [ellipse.major_radius * ellipse.minor_radius for ellipse in ellipses]
    center_x = float(sum(w * ellipse.cx for w, ellipse in zip(weights, ellipses)) / max(sum(weights), 1e-8))
    center_y = float(sum(w * ellipse.cy for w, ellipse in zip(weights, ellipses)) / max(sum(weights), 1e-8))
    angle_deg = _mean_angle_deg([ellipse.angle_deg for ellipse in ellipses], weights)
    aspect_ratio = float(
        sum(w * (ellipse.major_radius / max(ellipse.minor_radius, 1e-8)) for w, ellipse in zip(weights, ellipses))
        / max(sum(weights), 1e-8)
    )

    angle = math.radians(angle_deg)
    translate = np.array(
        [
            [1.0, 0.0, -center_x],
            [0.0, 1.0, -center_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotate = np.array(
        [
            [math.cos(-angle), -math.sin(-angle), 0.0],
            [math.sin(-angle), math.cos(-angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    scale = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, aspect_ratio, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = scale @ rotate @ translate
    return transform, {
        "center_x": center_x,
        "center_y": center_y,
        "angle_deg": angle_deg,
        "aspect_ratio": aspect_ratio,
    }


def assign_ring_diameters(rings: Sequence[Dict[str, object]], ring_inner_diameters_mm: Sequence[float]) -> List[Dict[str, object]]:
    augmented: List[Dict[str, object]] = []
    for idx, ring in enumerate(rings):
        inner = ellipse_from_dict(ring.get("inner_ellipse"))
        radius_key = inner.major_radius if inner is not None else -1.0
        augmented.append(
            {
                "ring_index": idx + 1,
                "ring": ring,
                "inner_major_radius_px": float(radius_key),
                "assigned_inner_diameter_mm": None,
            }
        )

    if len(augmented) == 1 and len(ring_inner_diameters_mm) == 1:
        augmented[0]["assigned_inner_diameter_mm"] = float(ring_inner_diameters_mm[0])
        return augmented

    sorted_rings = sorted(augmented, key=lambda item: item["inner_major_radius_px"])
    sorted_sizes = sorted(float(v) for v in ring_inner_diameters_mm)
    for item, size_mm in zip(sorted_rings, sorted_sizes):
        item["assigned_inner_diameter_mm"] = float(size_mm)
    return augmented


def choose_reference_ring(
    assigned_rings: Sequence[Dict[str, object]],
    polyp_centroid_xy: np.ndarray,
) -> Dict[str, object]:
    best_item = None
    best_score = None
    for item in assigned_rings:
        ring = item["ring"]
        inner = ellipse_from_dict(ring.get("inner_ellipse"))
        outer = ellipse_from_dict(ring.get("outer_ellipse"))
        center = None
        if inner is not None:
            center = np.array([inner.cx, inner.cy], dtype=np.float32)
        elif outer is not None:
            center = np.array([outer.cx, outer.cy], dtype=np.float32)
        if center is None or item.get("assigned_inner_diameter_mm") is None:
            continue
        dist = float(np.linalg.norm(center - polyp_centroid_xy))
        score = dist
        if best_score is None or score < best_score:
            best_score = score
            best_item = item

    if best_item is None:
        raise ValueError("No usable reference ring found.")
    return best_item


def measure_single_view_sample(
    stem: str,
    mask_path: Path,
    ring_fit_json_path: Path,
    image_path: Optional[Path] = None,
    min_polyp_area: int = 200,
) -> MeasurementResult:
    if not mask_path.exists():
        return MeasurementResult(stem=stem, status="error", reason=f"missing_mask:{mask_path}", metrics={})
    if not ring_fit_json_path.exists():
        return MeasurementResult(stem=stem, status="error", reason=f"missing_ring_fit:{ring_fit_json_path}", metrics={})

    ring_fit = json.loads(ring_fit_json_path.read_text(encoding="utf-8"))
    filename_meta = parse_capture_filename(stem).to_dict()
    if not filename_meta.get("has_reference", False):
        return MeasurementResult(stem=stem, status="skipped", reason="no_reference_ring", metrics={})

    assigned_rings = assign_ring_diameters(
        ring_fit.get("rings", []),
        filename_meta.get("ring_inner_diameters_mm", []),
    )
    if not assigned_rings:
        return MeasurementResult(stem=stem, status="skipped", reason="no_reference_ring", metrics={})

    calib = None
    calibration_json = ring_fit.get("calibration_json")
    if calibration_json:
        calib_path = Path(calibration_json)
        if calib_path.exists():
            calib = load_calibration(calib_path)

    label_mask = load_label_mask(mask_path)
    if calib is not None:
        label_mask = undistort_mask(label_mask, calib)
    polyp_mask = label_mask == POLYP_CLASS_ID
    if not np.any(polyp_mask):
        return MeasurementResult(stem=stem, status="error", reason="empty_polyp_mask", metrics={})

    provisional_centers = []
    for item in assigned_rings:
        inner = ellipse_from_dict(item["ring"].get("inner_ellipse"))
        outer = ellipse_from_dict(item["ring"].get("outer_ellipse"))
        if inner is not None:
            provisional_centers.append([inner.cx, inner.cy])
        elif outer is not None:
            provisional_centers.append([outer.cx, outer.cy])
    provisional_center = tuple(np.mean(np.asarray(provisional_centers, dtype=np.float32), axis=0)) if provisional_centers else None

    polyp_component_mask, polyp_centroid_xy = choose_polyp_component(polyp_mask, provisional_center, min_area=min_polyp_area)
    reference_ring = choose_reference_ring(assigned_rings, polyp_centroid_xy)
    ring_payload = reference_ring["ring"]
    inner_ellipse = ellipse_from_dict(ring_payload.get("inner_ellipse"))
    outer_ellipse = ellipse_from_dict(ring_payload.get("outer_ellipse"))
    if inner_ellipse is None:
        return MeasurementResult(stem=stem, status="error", reason="missing_inner_ellipse", metrics={})

    rectifier, rectifier_meta = build_affine_rectifier(inner_ellipse, outer_ellipse)
    inner_points = sample_ellipse_points(inner_ellipse, num_points=360)
    inner_points_rect = transform_points(inner_points, rectifier)
    inner_center_rect = np.mean(inner_points_rect, axis=0)
    inner_radius_rect_px = float(np.mean(np.linalg.norm(inner_points_rect - inner_center_rect, axis=1)))
    if inner_radius_rect_px <= 0:
        return MeasurementResult(stem=stem, status="error", reason="invalid_inner_radius", metrics={})

    assigned_inner_diameter_mm = float(reference_ring["assigned_inner_diameter_mm"])
    mm_per_px = assigned_inner_diameter_mm / max(inner_radius_rect_px * 2.0, 1e-8)
    px_per_mm = 1.0 / mm_per_px

    polyp_contour = contour_from_mask(polyp_component_mask)
    polyp_contour_rect = transform_points(polyp_contour, rectifier)
    polyp_area_rect_px2 = contour_area(polyp_contour_rect)
    polyp_perimeter_rect_px = polyline_perimeter(polyp_contour_rect)
    polyp_ellipse_rect = fit_ellipse_to_points(polyp_contour_rect)

    if polyp_ellipse_rect is not None:
        major_diameter_mm = polyp_ellipse_rect.major_radius * 2.0 * mm_per_px
        minor_diameter_mm = polyp_ellipse_rect.minor_radius * 2.0 * mm_per_px
        fitted_angle_deg = polyp_ellipse_rect.angle_deg
    else:
        xs = polyp_contour_rect[:, 0]
        ys = polyp_contour_rect[:, 1]
        major_diameter_mm = (float(xs.max() - xs.min())) * mm_per_px
        minor_diameter_mm = (float(ys.max() - ys.min())) * mm_per_px
        fitted_angle_deg = 0.0

    area_mm2 = polyp_area_rect_px2 * (mm_per_px ** 2)
    perimeter_mm = polyp_perimeter_rect_px * mm_per_px
    equivalent_diameter_mm = math.sqrt(max(4.0 * area_mm2 / math.pi, 0.0))

    metrics: Dict[str, object] = {
        "stem": stem,
        "reference_ring_index": int(reference_ring["ring_index"]),
        "reference_ring_inner_diameter_mm": assigned_inner_diameter_mm,
        "reference_ring_inner_rectified_diameter_px": float(inner_radius_rect_px * 2.0),
        "mm_per_px": float(mm_per_px),
        "px_per_mm": float(px_per_mm),
        "rectifier": rectifier_meta,
        "polyp_pixel_area": int(np.sum(polyp_component_mask)),
        "polyp_centroid_xy": [float(polyp_centroid_xy[0]), float(polyp_centroid_xy[1])],
        "planar_major_diameter_mm": float(major_diameter_mm),
        "planar_minor_diameter_mm": float(minor_diameter_mm),
        "planar_equivalent_diameter_mm": float(equivalent_diameter_mm),
        "planar_area_mm2": float(area_mm2),
        "planar_perimeter_mm": float(perimeter_mm),
        "planar_ellipse_angle_deg": float(fitted_angle_deg),
        "ring_quality": ring_payload.get("quality", {}),
    }

    if image_path is not None and image_path.exists():
        metrics["image_path"] = str(image_path)
    metrics["mask_path"] = str(mask_path)
    metrics["ring_fit_json_path"] = str(ring_fit_json_path)
    return MeasurementResult(stem=stem, status="ok", reason="ok", metrics=metrics)


def write_measurement_json(output_path: Path, result: MeasurementResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stem": result.stem,
        "status": result.status,
        "reason": result.reason,
        "metrics": result.metrics,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_measurement_summary(output_csv: Path, results: Sequence[MeasurementResult]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "stem",
        "status",
        "reason",
        "reference_ring_index",
        "reference_ring_inner_diameter_mm",
        "mm_per_px",
        "planar_major_diameter_mm",
        "planar_minor_diameter_mm",
        "planar_equivalent_diameter_mm",
        "planar_area_mm2",
        "planar_perimeter_mm",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "stem": result.stem,
                "status": result.status,
                "reason": result.reason,
                "reference_ring_index": result.metrics.get("reference_ring_index", ""),
                "reference_ring_inner_diameter_mm": result.metrics.get("reference_ring_inner_diameter_mm", ""),
                "mm_per_px": result.metrics.get("mm_per_px", ""),
                "planar_major_diameter_mm": result.metrics.get("planar_major_diameter_mm", ""),
                "planar_minor_diameter_mm": result.metrics.get("planar_minor_diameter_mm", ""),
                "planar_equivalent_diameter_mm": result.metrics.get("planar_equivalent_diameter_mm", ""),
                "planar_area_mm2": result.metrics.get("planar_area_mm2", ""),
                "planar_perimeter_mm": result.metrics.get("planar_perimeter_mm", ""),
            }
            writer.writerow(row)


def _make_preview_canvas(image: np.ndarray, rectified_points: np.ndarray, padding: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    if len(rectified_points) == 0:
        return np.eye(3, dtype=np.float64), image
    min_xy = np.floor(rectified_points.min(axis=0) - padding).astype(int)
    max_xy = np.ceil(rectified_points.max(axis=0) + padding).astype(int)
    width = int(max_xy[0] - min_xy[0] + 1)
    height = int(max_xy[1] - min_xy[1] + 1)
    shift = np.array(
        [
            [1.0, 0.0, -float(min_xy[0])],
            [0.0, 1.0, -float(min_xy[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    canvas = np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
    return shift, canvas


def write_measurement_visualization(
    output_dir: Path,
    result: MeasurementResult,
    image_path: Optional[Path],
    mask_path: Path,
    ring_fit_json_path: Path,
) -> None:
    if result.status != "ok":
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    ring_fit = json.loads(ring_fit_json_path.read_text(encoding="utf-8"))
    calib = None
    calibration_json = ring_fit.get("calibration_json")
    if calibration_json:
        calib_path = Path(calibration_json)
        if calib_path.exists():
            calib = load_calibration(calib_path)

    label_mask = load_label_mask(mask_path)
    if calib is not None:
        label_mask = undistort_mask(label_mask, calib)
    polyp_mask = (label_mask == POLYP_CLASS_ID).astype(np.uint8)

    if image_path is not None and image_path.exists():
        image = cv2.imread(str(image_path))
        if image is not None and calib is not None:
            image = undistort_image(image, calib)
    else:
        image = np.dstack([label_mask] * 3)

    if image is None:
        return

    overlay = image.copy()
    overlay[polyp_mask > 0] = (
        0.55 * overlay[polyp_mask > 0] + 0.45 * np.array([0, 64, 255], dtype=np.float32)
    ).astype(np.uint8)

    contour = contour_from_mask(polyp_mask > 0)
    cv2.polylines(overlay, [contour.reshape(-1, 1, 2).astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)

    ring_index = int(result.metrics["reference_ring_index"]) - 1
    ring = ring_fit["rings"][ring_index]
    for key, color in [("outer_ellipse", (0, 255, 0)), ("inner_ellipse", (255, 255, 0))]:
        ellipse = ellipse_from_dict(ring.get(key))
        if ellipse is None:
            continue
        cv2.ellipse(
            overlay,
            ((ellipse.cx, ellipse.cy), (ellipse.major_radius * 2.0, ellipse.minor_radius * 2.0), ellipse.angle_deg),
            color,
            2,
            cv2.LINE_AA,
        )

    text_lines = [
        f"major={result.metrics['planar_major_diameter_mm']:.2f} mm",
        f"minor={result.metrics['planar_minor_diameter_mm']:.2f} mm",
        f"eq={result.metrics['planar_equivalent_diameter_mm']:.2f} mm",
        f"area={result.metrics['planar_area_mm2']:.2f} mm^2",
    ]
    for idx, line in enumerate(text_lines):
        cv2.putText(
            overlay,
            line,
            (24, 32 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_dir / f"{result.stem}.jpg"), overlay)


def load_manifest_records(manifest_json: Path) -> List[Dict[str, object]]:
    payload = json.loads(manifest_json.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported manifest format: {manifest_json}")


def iter_measurement_inputs(records: Sequence[Dict[str, object]], stems_filter: Optional[Iterable[str]] = None) -> Iterable[Tuple[str, Optional[Path], Path, Path]]:
    allow = set(stems_filter) if stems_filter is not None else None
    for record in records:
        stem = str(record["stem"])
        if allow is not None and stem not in allow:
            continue
        mask_path = record.get("mask_path")
        ring_fit_json_path = record.get("ring_fit_json_path")
        if not mask_path or not ring_fit_json_path:
            continue
        image_path = record.get("image_path")
        yield (
            stem,
            Path(image_path) if image_path else None,
            Path(mask_path),
            Path(ring_fit_json_path),
        )
