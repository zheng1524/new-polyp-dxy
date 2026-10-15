"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from typing import Dict as Dict
import math as math
import numpy as np


def transform_points(points_xy: np.ndarray, matrix_3x3: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float64)
    homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    warped = homog @ matrix_3x3.T
    warped_xy = warped[:, :2] / np.clip(warped[:, 2:3], 1e-8, None)
    return warped_xy.astype(np.float32)


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


def ellipse_from_dict(payload: Optional[Dict[str, float]]) -> Optional[EllipseParams]:
    if payload is None:
        return None
    return EllipseParams(**payload)


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


@dataclass
class EllipseParams:
    cx: float
    cy: float
    major_radius: float
    minor_radius: float
    angle_deg: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


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


# Bind relocated dependencies after defining functions (including override hooks).
