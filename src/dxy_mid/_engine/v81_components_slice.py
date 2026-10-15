"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import copy as copy
import cv2 as cv2
import numpy as np
RUN_VERSION = 'V8.1-observed-boundary-evidence-gate-1.0'
RULES = {'normal_probe_px': [3.0, 5.0], 'inside_min': 0.8, 'background_min': 0.8, 'ring_clearance_px': 6.0, 'image_contrast_min': 5.0, 'strong_P_min': 80, 'strong_P_sectors': 12, 'strong_P_per_sector': 4, 'serious_Q_width_multiplier': 1.0, 'serious_Q_pixel_margin': 2.0, 'serious_Q_rr_margin': 0.01, 'serious_Q_arc_fraction_max': 0.2}

def repair_boundary(frame, base, raw, old):
    """Only guard-affected X with observed exterior evidence can become P."""
    d = copy.deepcopy(old)
    labels = frame.labels
    polyp = labels == 1
    bg = labels == 0
    rm = labels == 2
    distance = cv2.distanceTransform((~rm).astype(np.uint8), cv2.DIST_L2, 5)
    gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
    (_, cc) = cv2.connectedComponents(polyp.astype(np.uint8), 8)
    body_ids = {c['name']: {int(cc[int(q[1]), int(q[0])]) for q in c['trusted'] if polyp[int(q[1]), int(q[0])]} - {0} for c in base.candidates}
    changes = []
    checks = []
    for rec in d['records']:
        if rec['class_'] != 'X':
            continue
        q = np.array(rec['q'], float)
        (x, y) = map(int, q)
        why = rec['reason']
        check = dict(global_index=rec['global_index'], old_reason=why)
        rec['old_class_'] = rec['class_']
        rec['old_reason'] = why
        if rec['parent_contour'] >= 0 or why in ['internal_polyp_hole', 'not_in_V2_selected_body_or_sample', 'original_component_not_supported_by_V2_body']:
            check['repair_reject'] = 'preserve_holes_and_V2_body_exclusions'
            checks.append(check)
            continue
        support = [name for (name, ids) in body_ids.items() if int(cc[y, x]) in ids]
        if not support or not base.unknown_mask[y, x]:
            check['repair_reject'] = 'not_guard_affected_P_supported_body'
            checks.append(check)
            continue
        check['actual_ring_distance_px'] = float(distance[y, x])
        if distance[y, x] <= RULES['ring_clearance_px']:
            if not near_ring_evidence(rec, d, polyp, rm, q, check):
                checks.append(check)
                continue
        contour = d['contours'][rec['contour_id']]
        (n, normal_reason, metrics) = boundary_v7.normal(contour, rec['contour_index'], polyp)
        check.update(metrics)
        if n is None:
            check['repair_reject'] = normal_reason
            checks.append(check)
            continue
        ins = [boundary_v7.RAW.frac(polyp, q - n * hop, 1) for hop in RULES['normal_probe_px']]
        outs = [boundary_v7.RAW.frac(bg, q + n * hop, 1) for hop in RULES['normal_probe_px']]
        a = np.rint(q + n * 3).astype(int)
        b = np.rint(q - n * 3).astype(int)
        a = np.clip(a, [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
        b = np.clip(b, [0, 0], [gray.shape[1] - 1, gray.shape[0] - 1])
        contrast = abs(float(gray[a[1], a[0]]) - float(gray[b[1], b[0]]))
        check.update(inside_fraction=min(ins), background_fraction=min(outs), image_contrast=contrast)
        if min(ins) < RULES['inside_min'] or min(outs) < RULES['background_min']:
            check['repair_reject'] = 'polyp_background_normal_evidence_insufficient'
            checks.append(check)
            continue
        if contrast < RULES['image_contrast_min']:
            check['repair_reject'] = 'image_edge_contrast_insufficient'
            checks.append(check)
            continue
        rec.update(class_='P', reason='observed_polyp_background_outside_actual_ring', repair_body_support=support, repair_normal=n.tolist(), repair_evidence=check)
        changes.append(dict(global_index=rec['global_index'], contour_id=rec['contour_id'], contour_index=rec['contour_index'], q=rec['q'], old_class='X', new_class='P', old_reason=why, body_support=support, **{k: v for (k, v) in check.items() if k not in ['global_index', 'old_reason']}))
        check['repair_reject'] = ''
        checks.append(check)
    newbase = copy.deepcopy(base)
    for c in newbase.candidates:
        extra = [r['q'] for r in d['records'] if r.get('repair_body_support') and c['name'] in r['repair_body_support']]
        if extra:
            c['trusted'] = np.vstack([c['trusted'], np.array(extra)])
            c['points_local'] = c['trusted'] - c['crop_offset']
    points = np.array([r['q'] for r in d['records']], int).reshape(-1, 2)
    states = np.array([r['class_'] for r in d['records']])
    result = BoundaryResult(newbase.candidates, points[states == 'P'], points[states == 'X'], points[states == 'Q'], base.unknown_mask, dict(base.provenance, method='V8.1_guard_evidence_repair', implementation_version=RUN_VERSION))
    d.update(version=RUN_VERSION, base_version=old['version'], repair_rules=RULES, classification_changes=changes, repair_checks=checks)
    return (result, d)

# Bind relocated dependencies after defining functions (including override hooks).
from . import boundary_v7 as boundary_v7
from .contracts import BoundaryResult as BoundaryResult
from .v82_boundary import near_ring_evidence as near_ring_evidence
