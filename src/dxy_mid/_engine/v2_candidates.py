"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import cv2 as cv2
import math as math
import numpy as np


def process_candidates(labels, geometry, source_id):
    original = labels
    rings = [dict(inner_ellipse=geometry['inner'], outer_ellipse=geometry['outer'])]
    r = dict(stem=source_id, group_key='current_frame')
    ringinfo = rings[0]
    inner = ringinfo['inner_ellipse']
    outer = ringinfo['outer_ellipse']
    (yy, xx) = np.nonzero((original == 1) | (original == 2))
    pad = 40
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(original.shape[1], int(xx.max()) + pad + 1)
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(original.shape[0], int(yy.max()) + pad + 1)
    polyp = (original[y0:y1, x0:x1] == 1).astype(np.uint8)
    ring = (original[y0:y1, x0:x1] == 2).astype(np.uint8)

    def local(e):
        return {**e, 'cx': e['cx'] - x0, 'cy': e['cy'] - y0}
    inner = local(inner)
    outer = local(outer)
    center = np.array([inner['cx'], inner['cy']])
    rr = math.sqrt(inner['major_radius'] * inner['minor_radius'])
    innerfill = ellipse_mask(polyp.shape, inner)
    outerfill = ellipse_mask(polyp.shape, outer)
    inside = innerfill.astype(np.uint8)
    outside = (1 - outerfill).astype(np.uint8)
    band = cv2.dilate(ring, disk(max(3, 0.08 * rr))).astype(np.uint8)
    band = np.maximum(band, (1 - inside & 1 - outside).astype(np.uint8))
    (comps, edges) = component_records(polyp, inside, outside, band, center, rr)
    descriptors = [{'stem': r['stem'], 'group_key': r['group_key'], **d} for (d, *_) in comps]
    graph = [{'stem': r['stem'], 'group_key': r['group_key'], **edge} for edge in edges]
    broad_out = [(d, cm) for (d, cm, *_) in comps if d['region'] == 'outside' and d['broad_descriptor'] and (d['visible_contour_points'] >= 12)]
    all_substantial = [(d, cm) for (d, cm, *_) in comps if d['visible_contour_points'] >= 12]
    A = np.zeros_like(polyp)
    for (d, cm) in broad_out if broad_out else all_substantial:
        A = np.maximum(A, cm)
    allmask = np.zeros_like(polyp)
    for (d, cm) in all_substantial:
        allmask = np.maximum(allmask, cm)
    B = cv2.morphologyEx(allmask, cv2.MORPH_OPEN, disk(max(2, 0.12 * rr)))
    return dict(A=A, B=B, band=band, center=center, rr=rr, crop=(x0, y0, x1, y1), components=comps, edges=edges, candidate_A_source='broad_outside' if broad_out else 'all_substantial')

# Bind relocated dependencies after defining functions (including override hooks).
from .v2 import component_records as component_records
from .v2 import disk as disk
from .v2 import ellipse_mask as ellipse_mask
