import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_max

from .calibration import CalibrationResult, load_calibration, undistort_image, undistort_mask


RING_CLASS_ID = 2


@dataclass
class EllipseParams:
    cx: float
    cy: float
    major_radius: float
    minor_radius: float
    angle_deg: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "cx": float(self.cx),
            "cy": float(self.cy),
            "major_radius": float(self.major_radius),
            "minor_radius": float(self.minor_radius),
            "angle_deg": float(self.angle_deg),
        }


def parse_filename_metadata(stem: str) -> Dict[str, object]:
    parts = stem.split("_")
    ring_code = parts[4] if len(parts) > 4 else ""
    ring_mode = parts[5] if len(parts) > 5 else "none"
    reference_token = parts[6] if len(parts) > 6 else "refN"

    if "refN" in stem:
        expected_rings = 0
    elif "double" in stem:
        expected_rings = 2
    elif "single" in stem:
        expected_rings = 1
    else:
        expected_rings = None

    ring_inner_diameters_mm: List[float]
    if expected_rings == 0:
        ring_inner_diameters_mm = []
    elif ring_mode == "double":
        # 当前数据命名中 double 表示 5 mm 与 10 mm 内径参考环各一个。
        ring_inner_diameters_mm = [5.0, 10.0]
    elif ring_code == "R5":
        ring_inner_diameters_mm = [5.0]
    elif ring_code == "R10":
        ring_inner_diameters_mm = [10.0]
    else:
        ring_inner_diameters_mm = []

    return {
        "stem": stem,
        "expected_rings": expected_rings,
        "has_reference": expected_rings not in (0, None),
        "reference_mode": "double" if "double" in stem else "single" if "single" in stem else "none",
        "ring_code": ring_code,
        "ring_mode": ring_mode,
        "reference_token": reference_token,
        "ring_inner_diameters_mm": ring_inner_diameters_mm,
    }


def load_ring_mask(mask_path: Path) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
    return mask == RING_CLASS_ID


def preprocess_ring_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


def split_mask_if_needed(mask: np.ndarray, expected_rings: Optional[int]) -> np.ndarray:
    if expected_rings is None or expected_rings <= 1:
        return mask

    labeled = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labeled)
    if len(regions) >= expected_rings:
        return mask

    distance = ndi.distance_transform_edt(mask)
    peak_coords = peak_local_max(
        distance,
        labels=mask.astype(np.uint8),
        num_peaks=expected_rings,
        min_distance=max(10, int(min(mask.shape) * 0.02)),
    )
    if len(peak_coords) < expected_rings:
        return mask

    markers = np.zeros_like(mask, dtype=np.int32)
    for idx, (r, c) in enumerate(peak_coords, start=1):
        markers[r, c] = idx
    markers = ndi.label(markers > 0)[0]
    labels = segmentation.watershed(-distance, markers, mask=mask)
    return labels > 0


def select_components(mask: np.ndarray, expected_rings: Optional[int], min_area: int = 300) -> List[np.ndarray]:
    mask = split_mask_if_needed(mask, expected_rings)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    comps: List[Tuple[int, np.ndarray]] = []
    for idx in range(1, num_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comps.append((area, labels == idx))
    comps.sort(key=lambda item: item[0], reverse=True)
    if expected_rings is not None and expected_rings >= 0:
        comps = comps[: max(expected_rings, len(comps) if expected_rings == 0 else expected_rings)]
    return [comp for _, comp in comps]


def component_area_ellipse(component_mask: np.ndarray) -> Optional[EllipseParams]:
    labeled = measure.label(component_mask.astype(np.uint8), connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return None
    prop = max(props, key=lambda p: p.area)
    cy, cx = prop.centroid
    major_radius = prop.axis_major_length / 2.0
    minor_radius = prop.axis_minor_length / 2.0
    angle_deg = -math.degrees(prop.orientation)
    return EllipseParams(cx=cx, cy=cy, major_radius=major_radius, minor_radius=minor_radius, angle_deg=angle_deg)


def ellipse_from_cv2(ellipse) -> EllipseParams:
    (cx, cy), (axis_a, axis_b), angle_deg = ellipse
    if axis_a >= axis_b:
        major_radius = axis_a / 2.0
        minor_radius = axis_b / 2.0
        major_angle = angle_deg
    else:
        major_radius = axis_b / 2.0
        minor_radius = axis_a / 2.0
        major_angle = angle_deg + 90.0
    return EllipseParams(cx=float(cx), cy=float(cy), major_radius=float(major_radius), minor_radius=float(minor_radius), angle_deg=float(major_angle))


def ellipse_to_cv2(params: EllipseParams):
    return (
        (float(params.cx), float(params.cy)),
        (float(params.major_radius * 2.0), float(params.minor_radius * 2.0)),
        float(params.angle_deg),
    )


def sample_contour_points(contour: np.ndarray, max_points: int = 800) -> np.ndarray:
    points = contour.reshape(-1, 2)
    if len(points) <= max_points:
        return points.astype(np.float32)
    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int32)
    return points[idx].astype(np.float32)


def ellipse_residuals(params: EllipseParams, points: np.ndarray) -> np.ndarray:
    theta = math.radians(params.angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = points[:, 0] - params.cx
    dy = points[:, 1] - params.cy
    xr = cos_t * dx + sin_t * dy
    yr = -sin_t * dx + cos_t * dy
    norm = np.sqrt((xr / max(params.major_radius, 1e-6)) ** 2 + (yr / max(params.minor_radius, 1e-6)) ** 2)
    scale = max(1.0, 0.5 * (params.major_radius + params.minor_radius))
    return np.abs(norm - 1.0) * scale


def fit_ellipse_ransac(points: np.ndarray, iterations: int = 200) -> Tuple[Optional[EllipseParams], Dict[str, float], np.ndarray]:
    if len(points) < 5:
        return None, {"mean_residual": float("inf"), "inlier_ratio": 0.0}, np.zeros(len(points), dtype=bool)

    best_inliers = None
    best_params = None
    best_score = (-1, float("inf"))
    rng = np.random.default_rng(42)
    subset_size = min(max(20, 5), len(points))
    for _ in range(iterations):
        subset_idx = rng.choice(len(points), size=subset_size, replace=False)
        subset = points[subset_idx].reshape(-1, 1, 2).astype(np.float32)
        try:
            ellipse = cv2.fitEllipseDirect(subset)
        except cv2.error:
            continue
        params = ellipse_from_cv2(ellipse)
        residuals = ellipse_residuals(params, points)
        threshold = max(2.5, 0.015 * max(params.major_radius * 2.0, params.minor_radius * 2.0))
        inliers = residuals < threshold
        score = (int(inliers.sum()), float(residuals[inliers].mean()) if np.any(inliers) else float("inf"))
        if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
            best_score = score
            best_inliers = inliers
            best_params = params

    if best_inliers is None or best_params is None:
        try:
            ellipse = cv2.fitEllipseDirect(points.reshape(-1, 1, 2).astype(np.float32))
            best_params = ellipse_from_cv2(ellipse)
            best_inliers = np.ones(len(points), dtype=bool)
        except cv2.error:
            return None, {"mean_residual": float("inf"), "inlier_ratio": 0.0}, np.zeros(len(points), dtype=bool)

    inlier_points = points[best_inliers]
    if len(inlier_points) >= 5:
        try:
            refined = cv2.fitEllipseDirect(inlier_points.reshape(-1, 1, 2).astype(np.float32))
            best_params = ellipse_from_cv2(refined)
        except cv2.error:
            pass

    residuals = ellipse_residuals(best_params, points)
    metrics = {
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "inlier_ratio": float(np.mean(best_inliers)),
        "eccentricity": float(math.sqrt(max(0.0, 1.0 - (best_params.minor_radius ** 2) / max(best_params.major_radius ** 2, 1e-6)))),
    }
    return best_params, metrics, best_inliers


def extract_component_boundaries(component_mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    mask_u8 = component_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        return None, None
    hierarchy = hierarchy[0]
    outer_candidates = [i for i, h in enumerate(hierarchy) if h[3] == -1]
    if not outer_candidates:
        return None, None
    outer_idx = max(outer_candidates, key=lambda i: cv2.contourArea(contours[i]))
    inner_candidates = [i for i, h in enumerate(hierarchy) if h[3] == outer_idx]
    inner_idx = max(inner_candidates, key=lambda i: cv2.contourArea(contours[i])) if inner_candidates else None
    outer = sample_contour_points(contours[outer_idx])
    inner = sample_contour_points(contours[inner_idx]) if inner_idx is not None else None
    return outer, inner


def skeleton_midline_ellipse(component_mask: np.ndarray) -> Optional[EllipseParams]:
    skeleton = morphology.medial_axis(component_mask)
    points_rc = np.column_stack(np.nonzero(skeleton))
    if len(points_rc) < 20:
        return None
    points_xy = np.column_stack((points_rc[:, 1], points_rc[:, 0])).astype(np.float32)
    params, _, _ = fit_ellipse_ransac(points_xy, iterations=120)
    return params


def quality_checks(
    area_ellipse: Optional[EllipseParams],
    outer_ellipse: Optional[EllipseParams],
    inner_ellipse: Optional[EllipseParams],
    outer_metrics: Optional[Dict[str, float]],
    inner_metrics: Optional[Dict[str, float]],
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    if outer_ellipse is not None and inner_ellipse is not None:
        center_gap = float(math.hypot(outer_ellipse.cx - inner_ellipse.cx, outer_ellipse.cy - inner_ellipse.cy))
        mean_radius = max(1.0, 0.5 * (outer_ellipse.major_radius + outer_ellipse.minor_radius))
        result["center_gap_px"] = center_gap
        result["relative_center_gap"] = float(center_gap / mean_radius)
        result["thickness_major"] = float(max(0.0, outer_ellipse.major_radius - inner_ellipse.major_radius))
        result["thickness_minor"] = float(max(0.0, outer_ellipse.minor_radius - inner_ellipse.minor_radius))
    if area_ellipse is not None and outer_ellipse is not None:
        gap = float(math.hypot(area_ellipse.cx - outer_ellipse.cx, area_ellipse.cy - outer_ellipse.cy))
        result["area_to_outer_center_gap_px"] = gap
    if outer_metrics is not None:
        result["outer_mean_residual"] = outer_metrics["mean_residual"]
        result["outer_inlier_ratio"] = outer_metrics["inlier_ratio"]
    if inner_metrics is not None:
        result["inner_mean_residual"] = inner_metrics["mean_residual"]
        result["inner_inlier_ratio"] = inner_metrics["inlier_ratio"]
    return result


def fit_single_component(component_mask: np.ndarray) -> Dict[str, object]:
    result: Dict[str, object] = {
        "area_ellipse": None,
        "outer_ellipse": None,
        "inner_ellipse": None,
        "mid_ellipse": None,
        "quality": {},
    }
    area_ellipse = component_area_ellipse(component_mask)
    result["area_ellipse"] = area_ellipse.as_dict() if area_ellipse is not None else None

    outer_points, inner_points = extract_component_boundaries(component_mask)
    outer_params = outer_metrics = None
    inner_params = inner_metrics = None
    if outer_points is not None and len(outer_points) >= 5:
        outer_params, outer_metrics, _ = fit_ellipse_ransac(outer_points)
        result["outer_ellipse"] = outer_params.as_dict() if outer_params is not None else None
    if inner_points is not None and len(inner_points) >= 5:
        inner_params, inner_metrics, _ = fit_ellipse_ransac(inner_points)
        result["inner_ellipse"] = inner_params.as_dict() if inner_params is not None else None

    if inner_params is None:
        mid = skeleton_midline_ellipse(component_mask)
        result["mid_ellipse"] = mid.as_dict() if mid is not None else None

    result["quality"] = quality_checks(area_ellipse, outer_params, inner_params, outer_metrics, inner_metrics)
    result["pixel_area"] = int(component_mask.sum())
    return result


def draw_ellipse(canvas: np.ndarray, params: EllipseParams, color: Tuple[int, int, int], thickness: int = 2) -> None:
    cv2.ellipse(canvas, ellipse_to_cv2(params), color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_ring_visualization(
    image_path: Path,
    mask: np.ndarray,
    rings: List[Dict[str, object]],
    metadata: Dict[str, object],
    output_path: Path,
    calib: Optional[CalibrationResult] = None,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    if calib is not None:
        image = undistort_image(image, calib)
    overlay = image.copy()
    ring_region = mask.astype(bool)
    overlay[ring_region] = (0.7 * overlay[ring_region] + 0.3 * np.array([255, 200, 0])).astype(np.uint8)

    for idx, ring in enumerate(rings, start=1):
        if ring.get("area_ellipse") is not None:
            draw_ellipse(overlay, EllipseParams(**ring["area_ellipse"]), (0, 255, 255), 1)
        if ring.get("outer_ellipse") is not None:
            draw_ellipse(overlay, EllipseParams(**ring["outer_ellipse"]), (0, 255, 0), 2)
        if ring.get("inner_ellipse") is not None:
            draw_ellipse(overlay, EllipseParams(**ring["inner_ellipse"]), (255, 255, 0), 2)
        elif ring.get("mid_ellipse") is not None:
            draw_ellipse(overlay, EllipseParams(**ring["mid_ellipse"]), (0, 165, 255), 1)

        center = None
        if ring.get("outer_ellipse") is not None:
            center = ring["outer_ellipse"]["cx"], ring["outer_ellipse"]["cy"]
        elif ring.get("area_ellipse") is not None:
            center = ring["area_ellipse"]["cx"], ring["area_ellipse"]["cy"]
        if center is not None:
            cv2.putText(
                overlay,
                f"ring{idx}",
                (int(center[0]) + 8, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    footer = f"expected={metadata['expected_rings']} detected={len(rings)} mode={metadata['reference_mode']}"
    cv2.putText(overlay, footer, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def process_mask(mask_path: Path, image_path: Path, output_dir: Path) -> Dict[str, object]:
    return process_mask_with_calibration(mask_path, image_path, output_dir, calibration_json=None)


def process_mask_with_calibration(
    mask_path: Path,
    image_path: Path,
    output_dir: Path,
    calibration_json: Optional[Path],
) -> Dict[str, object]:
    metadata = parse_filename_metadata(mask_path.stem)
    calib = load_calibration(calibration_json) if calibration_json is not None else None
    ring_mask = load_ring_mask(mask_path)
    if calib is not None:
        ring_mask = undistort_mask(ring_mask.astype(np.uint8) * 255, calib) > 127
    ring_mask = preprocess_ring_mask(ring_mask)
    components = select_components(ring_mask, metadata["expected_rings"])

    rings = []
    if metadata["expected_rings"] != 0:
        for component in components:
            rings.append(fit_single_component(component))

    result = {
        "stem": mask_path.stem,
        "metadata": metadata,
        "ring_pixel_count": int(ring_mask.sum()),
        "detected_components": len(components),
        "rings": rings,
        "calibration_json": str(calibration_json) if calibration_json is not None else None,
    }

    draw_ring_visualization(
        image_path=image_path,
        mask=ring_mask,
        rings=rings,
        metadata=metadata,
        output_path=output_dir / "overlays" / f"{mask_path.stem}.jpg",
        calib=calib,
    )
    (output_dir / "json").mkdir(parents=True, exist_ok=True)
    (output_dir / "json" / f"{mask_path.stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def write_summary_csv(results: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stem",
                "expected_rings",
                "detected_components",
                "ring_index",
                "pixel_area",
                "outer_cx",
                "outer_cy",
                "outer_major_radius",
                "outer_minor_radius",
                "outer_angle_deg",
                "inner_cx",
                "inner_cy",
                "inner_major_radius",
                "inner_minor_radius",
                "inner_angle_deg",
                "center_gap_px",
                "outer_mean_residual",
                "inner_mean_residual",
            ]
        )
        for result in results:
            rings = result["rings"]
            if not rings:
                writer.writerow(
                    [result["stem"], result["metadata"]["expected_rings"], result["detected_components"], "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
                )
                continue
            for idx, ring in enumerate(rings, start=1):
                outer = ring.get("outer_ellipse") or {}
                inner = ring.get("inner_ellipse") or {}
                quality = ring.get("quality") or {}
                writer.writerow(
                    [
                        result["stem"],
                        result["metadata"]["expected_rings"],
                        result["detected_components"],
                        idx,
                        ring.get("pixel_area", ""),
                        outer.get("cx", ""),
                        outer.get("cy", ""),
                        outer.get("major_radius", ""),
                        outer.get("minor_radius", ""),
                        outer.get("angle_deg", ""),
                        inner.get("cx", ""),
                        inner.get("cy", ""),
                        inner.get("major_radius", ""),
                        inner.get("minor_radius", ""),
                        inner.get("angle_deg", ""),
                        quality.get("center_gap_px", ""),
                        quality.get("outer_mean_residual", ""),
                        quality.get("inner_mean_residual", ""),
                    ]
                )


def _bounded_penalty(value: float, warn: float, fail: float, warn_penalty: float, fail_penalty: float) -> Tuple[float, bool]:
    if value >= fail:
        return fail_penalty, True
    if value >= warn:
        return warn_penalty, False
    return 0.0, False


def evaluate_ring_quality(ring: Dict[str, object]) -> Dict[str, object]:
    reasons: List[str] = []
    score = 100.0
    hard_reject = False

    outer = ring.get("outer_ellipse")
    inner = ring.get("inner_ellipse")
    mid = ring.get("mid_ellipse")
    quality = ring.get("quality") or {}

    if outer is None:
        return {
            "status": "reject",
            "score": 0.0,
            "reasons": ["missing_outer_ellipse"],
        }

    outer_res = quality.get("outer_mean_residual")
    if isinstance(outer_res, (int, float)):
        penalty, failed = _bounded_penalty(float(outer_res), warn=12.0, fail=20.0, warn_penalty=25.0, fail_penalty=45.0)
        score -= penalty
        hard_reject = hard_reject or failed
        if failed:
            reasons.append("outer_residual_too_high")
        elif penalty > 0:
            reasons.append("outer_residual_high")

    outer_inlier = quality.get("outer_inlier_ratio")
    if isinstance(outer_inlier, (int, float)):
        outer_inlier = float(outer_inlier)
        if outer_inlier < 0.45:
            score -= 35.0
            hard_reject = True
            reasons.append("outer_inlier_too_low")
        elif outer_inlier < 0.60:
            score -= 20.0
            reasons.append("outer_inlier_low")
        elif outer_inlier < 0.75:
            score -= 8.0
            reasons.append("outer_inlier_marginal")

    if inner is None:
        score -= 15.0
        reasons.append("missing_inner_ellipse")
        if mid is not None:
            reasons.append("using_mid_ellipse_fallback")
    else:
        inner_res = quality.get("inner_mean_residual")
        if isinstance(inner_res, (int, float)):
            penalty, failed = _bounded_penalty(float(inner_res), warn=5.0, fail=8.0, warn_penalty=20.0, fail_penalty=35.0)
            score -= penalty
            hard_reject = hard_reject or failed
            if failed:
                reasons.append("inner_residual_too_high")
            elif penalty > 0:
                reasons.append("inner_residual_high")

        inner_inlier = quality.get("inner_inlier_ratio")
        if isinstance(inner_inlier, (int, float)):
            inner_inlier = float(inner_inlier)
            if inner_inlier < 0.55:
                score -= 28.0
                hard_reject = True
                reasons.append("inner_inlier_too_low")
            elif inner_inlier < 0.70:
                score -= 12.0
                reasons.append("inner_inlier_low")

        rel_gap = quality.get("relative_center_gap")
        if isinstance(rel_gap, (int, float)):
            rel_gap = float(rel_gap)
            if rel_gap > 0.12:
                score -= 35.0
                hard_reject = True
                reasons.append("center_gap_too_large")
            elif rel_gap > 0.08:
                score -= 18.0
                reasons.append("center_gap_large")
            elif rel_gap > 0.05:
                score -= 8.0
                reasons.append("center_gap_marginal")

        th_major = quality.get("thickness_major")
        th_minor = quality.get("thickness_minor")
        if isinstance(th_major, (int, float)) and isinstance(th_minor, (int, float)) and th_major > 0 and th_minor > 0:
            thickness_ratio = float(th_major) / float(th_minor)
            if thickness_ratio < 0.60 or thickness_ratio > 1.80:
                score -= 28.0
                hard_reject = True
                reasons.append("thickness_ratio_invalid")
            elif thickness_ratio < 0.75 or thickness_ratio > 1.40:
                score -= 12.0
                reasons.append("thickness_ratio_large")
            elif thickness_ratio < 0.85 or thickness_ratio > 1.20:
                score -= 5.0
                reasons.append("thickness_ratio_marginal")

            inner_outer_major_ratio = float(inner["major_radius"]) / max(float(outer["major_radius"]), 1e-6)
            inner_outer_minor_ratio = float(inner["minor_radius"]) / max(float(outer["minor_radius"]), 1e-6)
            if inner_outer_major_ratio < 0.55 or inner_outer_minor_ratio < 0.55:
                score -= 20.0
                hard_reject = True
                reasons.append("inner_outer_radius_ratio_invalid")
        else:
            score -= 10.0
            reasons.append("thickness_unavailable")

    score = max(0.0, min(100.0, score))
    if hard_reject or score < 50.0:
        status = "reject"
    elif score < 80.0 or inner is None:
        status = "review"
    else:
        status = "accept"

    if not reasons:
        reasons.append("good_fit")
    return {
        "status": status,
        "score": score,
        "reasons": reasons,
    }


def evaluate_result_quality(result: Dict[str, object]) -> Dict[str, object]:
    metadata = result.get("metadata") or {}
    expected = metadata.get("expected_rings")
    detected = int(result.get("detected_components", 0))
    rings = result.get("rings") or []

    ring_assessments = []
    for idx, ring in enumerate(rings, start=1):
        assessment = evaluate_ring_quality(ring)
        assessment["ring_index"] = idx
        ring_assessments.append(assessment)

    reasons: List[str] = []
    if expected == 0:
        if detected == 0:
            return {
                "status": "accept",
                "score": 100.0,
                "reasons": ["no_ring_expected_and_detected"],
                "ring_assessments": [],
            }
        return {
            "status": "reject",
            "score": 0.0,
            "reasons": ["unexpected_ring_detected"],
            "ring_assessments": ring_assessments,
        }

    if expected is not None and detected != expected:
        reasons.append("detected_count_mismatch")

    if not ring_assessments:
        reasons.append("no_ring_fit_available")
        return {
            "status": "reject",
            "score": 0.0,
            "reasons": reasons,
            "ring_assessments": [],
        }

    scores = [float(item["score"]) for item in ring_assessments]
    score = float(sum(scores) / len(scores))
    statuses = [item["status"] for item in ring_assessments]

    if "detected_count_mismatch" in reasons or "reject" in statuses:
        status = "reject"
    elif "review" in statuses:
        status = "review"
    else:
        status = "accept"

    for item in ring_assessments:
        for reason in item["reasons"]:
            reasons.append(f"ring{item['ring_index']}:{reason}")

    return {
        "status": status,
        "score": score,
        "reasons": reasons,
        "ring_assessments": ring_assessments,
    }
