"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from dataclasses import dataclass as dataclass
from skimage.feature.peak import peak_local_max as peak_local_max
from typing import Dict as Dict
import cv2 as cv2
import math as math
import numpy as np
import scipy.ndimage as ndi
import skimage.measure as measure
import skimage.morphology as morphology
import skimage.segmentation as segmentation


def preprocess_ring_mask(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened.astype(bool)


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


def sample_contour_points(contour: np.ndarray, max_points: int = 800) -> np.ndarray:
    points = contour.reshape(-1, 2)
    if len(points) <= max_points:
        return points.astype(np.float32)
    idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int32)
    return points[idx].astype(np.float32)


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


# Bind relocated dependencies after defining functions (including override hooks).
