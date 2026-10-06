"""Current affine-rectified ring scale used by S5-CPAG-39."""
from __future__ import annotations

import numpy as np

from polypseg.single_view_measurement import (
    build_affine_rectifier,
    ellipse_from_dict,
    sample_ellipse_points,
    transform_points,
)


def current_rectified_scale(payload: dict) -> tuple[float, float, dict, dict]:
    """Return `(mm_per_px, rectified_ring_diameter_px, ring, quality)`.

    This is byte-for-byte equivalent in formula to the function used by the
    original dense-video audit.  It intentionally uses the known physical
    inner-ring diameter encoded by the capture filename.
    """
    ring = payload["rings"][0]
    inner = ellipse_from_dict(ring["inner_ellipse"])
    outer = ellipse_from_dict(ring.get("outer_ellipse"))
    matrix, _ = build_affine_rectifier(inner, outer)
    points = transform_points(sample_ellipse_points(inner, 360), matrix)
    center = np.mean(points, axis=0)
    diameter = 2 * float(np.mean(np.linalg.norm(points - center, axis=1)))
    ring_mm = float(payload["metadata"]["ring_inner_diameters_mm"][0])
    return ring_mm / diameter, diameter, ring, ring.get("quality") or {}
