"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import cv2 as cv2
import numpy as np


def candidate_fit(mask, band, center, rr, stem, name):
    guard = cv2.dilate(band, disk(max(2, 0.06 * rr)))
    pts = contour(mask)
    visible = pts[~guard[np.clip(pts[:, 1].astype(int), 0, guard.shape[0] - 1), np.clip(pts[:, 0].astype(int), 0, guard.shape[1] - 1)].astype(bool)] if len(pts) else pts
    removed = len(pts) - len(visible)
    (ys, xs) = np.nonzero(mask)
    if len(xs):
        geom = np.array([float(xs.mean()), float(ys.mean())])
        extent = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
    else:
        geom = center
        extent = rr
    if len(visible) > 2500:
        visible = visible[np.linspace(0, len(visible) - 1, 2500).astype(int)]
    return (visible, removed, geom, extent)

# Bind relocated dependencies after defining functions (including override hooks).
from .v2 import contour as contour
from .v2 import disk as disk
