"""Opt-in, read-only observation of the frozen pipeline for scientific figures.

No fitting, sampling, gating or aggregation is implemented here. A thread-local
Python return profiler observes the real numerical call's existing local values.
The original profiler is restored even on failure; no engine hook is replaced.
The result is intentionally separate from the normal measurement API/cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

from . import pipeline
from . import _engine as engine


@dataclass
class ObservedMeasurement:
    measurement: pipeline.Measurement
    native_image: np.ndarray
    original_labels: np.ndarray
    effective_labels: np.ndarray
    filled: np.ndarray
    boundary: dict | None
    boundary_before_filter: dict | None
    selected: dict | None
    results: dict | None
    base: object | None
    checks: dict


def observe_files(image_path, mask_path, *, inner_diameter_mm, source_id, stem):
    """Recompute a real frame and retain its actual transient numerical evidence.

    This diagnostic opt-in currently requires the frozen V9.1 Python call shape.
    Missing evidence is an explicit error, never reconstructed from predictions.
    """
    image = cv2.imread(str(image_path))
    labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    captured = {}
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event != 'return':
            return
        if frame.f_code is pipeline._measure.__code__:
            captured['pipeline'] = frame.f_locals.copy()
        elif frame.f_code is engine.short_Q.__code__:
            captured['before_filter'] = frame.f_locals['evidence']

    try:
        sys.setprofile(observe)
        measurement = pipeline.measure_frame(
            image, labels, inner_diameter_mm=float(inner_diameter_mm),
            source_id=source_id, stem=stem,
            image_sha256=pipeline.sha256(image_path),
            mask_sha256=pipeline.sha256(mask_path))
    finally:
        sys.setprofile(previous)
    if 'pipeline' not in captured:
        raise RuntimeError('Frozen numerical return evidence was not observed')
    v = captured['pipeline']
    observed = ObservedMeasurement(
        measurement, image, labels, v['effective'], v['filled'], v.get('e'),
        captured.get('before_filter'), v.get('selected'), v.get('results'),
        v.get('base'), {})
    observed.checks = validate_observation(observed)
    return observed


def ellipse_points(e, count=361):
    """Render the given ellipse parameters; never refit points."""
    t = np.linspace(0, 2*np.pi, count)
    angle = np.deg2rad(e['angle_deg'])
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return np.array([e['cx'], e['cy']]) + np.c_[
        e['major_radius']*np.cos(t), e['minor_radius']*np.sin(t)] @ rotation.T


def candidate_points(candidate, count=361):
    t = np.linspace(0, 2*np.pi, count)
    return candidate.center_native + np.c_[np.cos(t), np.sin(t)] @ candidate.basis_native.T


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def validate_observation(observed):
    m = observed.measurement
    a = m.audit
    checks = dict(same_image_mask_dimensions=observed.native_image.shape[:2] == observed.original_labels.shape,
                  polyp_mask_unchanged=bool(np.array_equal(observed.original_labels == 1, observed.effective_labels == 1)),
                  original_coordinate_system=a['coordinate_system'] == 'native_image_px',
                  zero_historical_fallback=a['history_prediction_fallback'] == 0,
                  GT_not_read=a['GT_read'] is False)
    if not m.ring.valid:
        checks.update(failure_has_no_geometry=m.candidate is None,
                      hard_failure_is_NaN=bool(np.isnan(a['D_frame_mm'])),
                      failure_excluded=not a['measurement_eligible'])
    else:
        if observed.selected is None or observed.boundary is None:
            raise AssertionError('Missing actual solver evidence')
        d = observed.selected['data']
        contours = observed.boundary['original_boundary']['contours']
        original, _ = cv2.findContours((observed.original_labels == 1).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        checks['raw_contour_order_identical'] = len(contours) == len(original) and all(
            np.array_equal(c.reshape(-1, 2), p.reshape(-1, 2)) for c, p in zip(contours, original))
        records = observed.boundary['original_boundary']['records']
        indexed = {z['global_index']: z for z in records}
        checks['PQX_partition_original_contours'] = len(records) == sum(len(c) for c in contours) and all(z['class_'] in ['P', 'Q', 'X'] for z in records)
        checks['Q_has_real_contour_index'] = all(
            np.array_equal(q['q'], contours[q['contour_id']][q['contour_index']]) and
            indexed[q['global_index']]['contour_id'] == q['contour_id'] and
            indexed[q['global_index']]['contour_index'] == q['contour_index'] for q in d['Q'])
        checks['Q_is_original_polyp_pixel'] = all(observed.original_labels[int(q['q'][1]), int(q['q'][0])] == 1 for q in d['Q'])
        checks['Q_unit_normal_finite_interval'] = all(
            np.isclose(np.linalg.norm(q['n']), 1) and np.isfinite([q['lo'], q['hi']]).all() and q['lo'] <= 0 <= q['hi'] for q in d['Q'])
        checks['Q_removed_not_in_solver'] = not set(observed.boundary['short_Q_filter']['removed_global_indices']) & {q['global_index'] for q in d['Q']}
        mid = m.ring.geometry['centerline']
        checks['Mid_axis_is_scale_source'] = bool(np.isclose(2*mid['major_radius'], a['L_mid_px'], atol=1e-8, rtol=1e-10))
        checks['Mid_physical_diameter'] = a['D_mid_mm'] in (6.2, 11.2)
        checks['Mid_formula'] = bool(np.isclose(a['scale_mm_per_px'], a['D_mid_mm']/a['L_mid_px'], atol=1e-12))
        if m.candidate is not None:
            _, _, axes, ends = engine.axes(m.candidate)
            audit_ends = np.array([[a['axis_x1'], a['axis_y1']], [a['axis_x2'], a['axis_y2']]])
            checks['chosen_axis_matches_audit'] = bool(np.allclose(ends, audit_ends, atol=1e-8))
            checks['axis_endpoints_length_L_px'] = bool(np.isclose(np.linalg.norm(ends[1]-ends[0]), a['L_px'], atol=1e-8))
            checks['chosen_singular_axis_length'] = bool(np.isclose(2*axes[0], a['L_px'], atol=1e-8))
            checks['same_frame_mm_formula'] = bool(np.isclose(a['D_frame_mm'], a['L_px']*a['scale_mm_per_px'], atol=1e-12))
    checks = {k: bool(v) for k, v in checks.items()}
    if not all(checks.values()):
        raise AssertionError({k: v for k, v in checks.items() if not v})
    return checks


def candidate_records(observed):
    """Every actual default A/B pool record, including rejected/profile records."""
    if observed.results is None:
        return []
    rows = []
    for body in observed.results['default']:
        for index, rec in enumerate(body['records']):
            p = rec['p']
            metrics = rec['metrics']
            original_reasons = rec.get('original_rejection_reasons', rec['reasons'])
            rows.append(dict(body=body['data']['name'], index=index, shape=rec['shape'],
                origin=rec['origin'], cx=p[0], cy=p[1], radius_1_px=np.exp(p[2]),
                radius_2_px=np.exp(p[3]), angle_rad=p[4],
                diameter_px=metrics['diameter_px'], P_loss=metrics['P_loss'],
                Q_loss=metrics['Q_loss'], score=metrics['score'],
                Q_support=metrics['Q_support_arc_weighted'],
                original_rejection_reasons=';'.join(original_reasons),
                current_reasons=';'.join(rec['reasons']),
                diagnostic_profile=rec['origin'].startswith('identifiability_')))
    return rows


def export_observation(observed, output):
    """Export numerical provenance separate from PNGs; no cache is published."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    m = observed.measurement
    summary = dict(source_id=m.audit['source_id'], audit=m.audit, checks=observed.checks,
                   ring=dict(valid=m.ring.valid, reason=m.ring.reason, geometry=m.ring.geometry),
                   evidence_source='Actual frozen pipeline return locals; no diagnostic fitting',
                   Q_semantic='original_contour_Q_constraint; never postfit Q_inferred')
    (out/'audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json))
    points = []
    for name in ['inner_points', 'outer_points', 'rejected_points', 'mid_points']:
        for i, q in enumerate(m.ring.evidence.get(name, [])):
            points.append(dict(kind=name, index=i, x=q[0], y=q[1]))
    pd.DataFrame(points, columns=['kind', 'index', 'x', 'y']).to_csv(out/'ring_points.csv', index=False)
    if observed.boundary is None:
        return
    d = observed.selected['data']
    q_map = {q['global_index']: (i, q) for i, q in enumerate(d['Q'])}
    original = observed.boundary['original_boundary']
    rows = []
    sampled_p = set(observed.selected['actual_sampled_P_indices'])
    p_map = {g: i for i, g in enumerate(d['P_global_indices'])}
    for z in original['records']:
        gid = z['global_index']
        row = dict(global_index=gid, contour_id=z['contour_id'], contour_index=z['contour_index'],
                   x=z['q'][0], y=z['q'][1], classification=z['class_'], reason=z['reason'],
                   solver_P=gid in p_map, sampled_P=p_map.get(gid, -1) in sampled_p, solver_Q=gid in q_map,
                   sampled_Q=False, scored_Q=False, normal_x=np.nan, normal_y=np.nan, lo_px=np.nan, hi_px=np.nan)
        if gid in q_map:
            i, q = q_map[gid]
            row.update(sampled_Q=i in observed.selected['actual_sampled_Q_indices'],
                       scored_Q=i in observed.selected['plan']['score_qi'], normal_x=q['n'][0],
                       normal_y=q['n'][1], lo_px=q['lo'], hi_px=q['hi'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'boundary_solver_points.csv', index=False)
    pd.DataFrame(candidate_records(observed)).to_csv(out/'geometry_candidates.csv', index=False)
    before = observed.boundary_before_filter
    if before is not None:
        pd.DataFrame([dict(global_index=q['global_index'], contour_id=q['contour_id'], contour_index=q['contour_index'],
                           x=q['q'][0], y=q['q'][1], removed=q['global_index'] in observed.boundary['short_Q_filter']['removed_global_indices'])
                      for q in before['source_Q_constraints']]).to_csv(out/'Q_before_short_arc_filter.csv', index=False)
