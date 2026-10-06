"""Two-way observed-area support used by the post-capture guard."""
from __future__ import annotations

import cv2
import numpy as np

from . import v2


def area_consistency(body: np.ndarray, ellipse: np.ndarray, unknown: np.ndarray):
    """Return C-like recall, P-like occupancy and observed pixel counts.

    Unknown pixels are excluded from both denominators.  The function is the
    same implementation used by the frozen core-occupancy audit.
    """
    observed_body = body & ~unknown
    observed_ellipse = ellipse & ~unknown
    intersection = observed_body & observed_ellipse
    body_n = int(observed_body.sum())
    ellipse_n = int(observed_ellipse.sum())
    match_n = int(intersection.sum())
    return (
        match_n / body_n if body_n else np.nan,
        match_n / ellipse_n if ellipse_n else np.nan,
        body_n,
        ellipse_n,
        match_n,
    )


def unknown_mask(label: np.ndarray, ring: dict) -> np.ndarray:
    """Existing V2 ring-occlusion-plus-annulus definition of U."""
    inner, outer = ring["inner_ellipse"], ring["outer_ellipse"]
    rr = float(np.sqrt(inner["major_radius"] * inner["minor_radius"]))
    inner_fill = v2.ellipse_mask(label.shape, inner).astype(bool)
    outer_fill = v2.ellipse_mask(label.shape, outer).astype(bool)
    return (
        cv2.dilate((label == 2).astype(np.uint8), v2.disk(max(3, 0.08 * rr))).astype(bool)
        | (outer_fill & ~inner_fill)
    )


def support_for_geometry(label: np.ndarray, ring: dict, cx: float, cy: float,
                         major: float, minor: float, angle: float) -> dict:
    """Compute C and P for an already chosen V2 circle or ellipse."""
    body = label == 1
    if not body.any():
        raise ValueError("empty_polyp_mask")
    raster = np.zeros(label.shape, np.uint8)
    cv2.ellipse(raster, (round(cx), round(cy)),
                (max(1, round(major / 2)), max(1, round(minor / 2))),
                angle, 0, 360, 1, -1)
    ellipse = raster.astype(bool)
    unknown = unknown_mask(label, ring)
    zeros = np.zeros(label.shape, bool)
    c, _, body_n, ellipse_n, overlap_n = area_consistency(body, ellipse, zeros)
    _, p, _, observed_ellipse_n, observed_overlap_n = area_consistency(body, ellipse, unknown)
    return {
        "mask_coverage_C": c,
        "fit_occupancy_P": p,
        "mask_area_px": body_n,
        "ellipse_area_px": ellipse_n,
        "intersection_px": overlap_n,
        "observed_ellipse_area_px": observed_ellipse_n,
        "observed_intersection_px": observed_overlap_n,
        "unknown_area_px": int(unknown.sum()),
    }
