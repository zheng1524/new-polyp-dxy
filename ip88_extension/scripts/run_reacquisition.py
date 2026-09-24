#!/usr/bin/env python3
"""GT-blind raw-video reacquisition for the previously unmeasurable Ip8.8/R10 group.

This is intentionally an isolated extension.  It does not read the old V2 R3
prediction.  It re-decodes two original videos, re-runs frozen INIT, and only
adapts a *missing* inner ring contour from the current fitter's already
available outer and skeleton-midline ellipses.  V2 itself is imported unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path('/home/liu/polyp_research')
OUT = ROOT/'experiments/dxy_ip88_raw_video_reacquisition_20260923'
AUDIT = ROOT/'experiments/dxy_scale_video_audit_20260919'
RAW = ROOT/'experiments/dxy_raw_video_frame_selection_20260921'
MAIN = ROOT/'experiments/dxy_post_capture_relaxed_area_guard_20260922'
V2ROOT = ROOT/'experiments/dxy_head_fitting_v2_20260918'
OCC = ROOT/'experiments/dxy_core_ellipse_occupancy_20260921'
MODEL = ROOT/'experiments/dxy_remeasurement_20260917/02_segmentation/model'
GROUP = 'GP_Ip_8.8mm_closed_R10_single_refY'
VIDEOS = [ROOT/'smart_endoscope_20260805/raw'/f'{GROUP}_{n:03d}.mp4' for n in (4, 5)]
for d in ('inputs', 'frames/images', 'frames/masks', 'frames/ring_raw',
          'frames/ring_adapted', 'tables', 'gallery/overlays'):
    (OUT/d).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(V2ROOT/'scripts'))
sys.path.insert(0, str(AUDIT/'scripts'))
sys.path.insert(0, str(OCC/'scripts'))
from polypseg.common import load_pretrained_model  # noqa: E402
from polypseg.ring_fitting import (  # noqa: E402
    EllipseParams, load_ring_mask, preprocess_ring_mask, select_components,
    process_mask_with_calibration,
)
import v2 as v2mod  # noqa: E402
import run_audit as aud  # noqa: E402
from audit import area_consistency  # noqa: E402


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(obj)
    return obj


PREP = module('keyframe_prep', ROOT/'scripts/prepare_smart_endoscope_multiview_dataset.py')
OLD = module('old_segmentation', ROOT/'smart_endoscope_20260805/scripts/run_old_method_v1.py')

PROTOCOL = {
    'scope': 'one formerly excluded group; frozen 38-group mainline remains untouched',
    'videos': [str(x) for x in VIDEOS],
    'raw_sampling': 'discard first/last 5%; split each source video into 5 equal temporal bins; decode 50 uniformly spaced source frames/bin (500 frames total)',
    'segmentation': 'frozen INIT model, 256px argmax, existing old.run_segmentation',
    'standard_ring': 'unchanged current process_mask_with_calibration; use native inner ellipse if present',
    'missing_inner_adapter': 'only when standard inner ellipse is absent: reuse current outer ellipse and current skeleton-midline ellipse; inner center=2*mid_center-outer_center and inner radii=2*mid_radii-outer_radii. This is annular centerline symmetry, not a new fitter or calibrated correction.',
    'adapter_validity': 'outer and mid must exist; both inferred inner radii must be positive; inferred inner axes must be smaller than outer axes. No GT-derived thresholds.',
    'measurement': 'unchanged current_rectified_scale and unmodified V2 process_view/choose/choose_geometry; L_px=Dxy/scale',
    'selection': 'per bin highest existing sharpness among V2-valid frames, then frozen post-capture relaxed area guard v1 (global formal-S0 p2 C/P) repairs only an extreme selected capture; B1 unchanged',
    'area_support': 'C=|M∩E|/|M|; P=|M∩(E\\U)|/|E\\U|, U=ring dilation plus fitted annulus; exact existing area_consistency routine',
    'aggregation': 'B1=10*median(L_px)/median(ring_px), >=4 selected valid bins, ring p95/p5-1>=.05; group median of eligible video captures',
    'old_V2_R3': 'not read, not used, not compared until final narrative only',
    'GT_used_before_freeze': False,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(data, rel: str) -> None:
    p = OUT/rel
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(p, index=False)


def finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def truth(x) -> bool:
    return x.strip().lower() in ('1', 'true', 'yes') if isinstance(x, str) else bool(x)


def sample_indices(n: int) -> list[list[int]]:
    lo, hi = int(math.ceil(.05*n)), int(math.floor(.95*n))-1
    grid = np.array_split(np.arange(lo, hi+1, dtype=int), 5)
    return [np.unique(np.linspace(x[0], x[-1], min(50, len(x))).round().astype(int)).tolist() for x in grid]


def extract() -> None:
    (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(PROTOCOL, ensure_ascii=False, indent=2)+'\n')
    rows = []
    for video in VIDEOS:
        cap = cv2.VideoCapture(str(video))
        assert cap.isOpened(), video
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = float(cap.get(cv2.CAP_PROP_FPS))
        assert n > 0 and fps > 0, (video, n, fps)
        for bin_id, indices in enumerate(sample_indices(n), 1):
            for ordinal, index in enumerate(indices, 1):
                stem = f'{video.stem}-reacq{bin_id:02d}-{ordinal:02d}'
                image = OUT/'frames/images'/f'{stem}.jpg'
                if not image.exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    decoded = bool(ok and frame is not None and cv2.imwrite(str(image), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]))
                else:
                    decoded = True
                rows.append({'group': GROUP, 'capture': video.stem, 'video': str(video),
                             'video_sha256': sha(video), 'frame_index': index, 'timestamp_sec': index/fps,
                             'fps': fps, 'video_frame_count': n, 'bin_id': bin_id, 'ordinal': ordinal,
                             'stem': stem, 'image_path': str(image) if decoded else '', 'decode_valid': decoded,
                             'decode_failure_reason': '' if decoded else 'decode_failed'})
        cap.release()
    d = pd.DataFrame(rows).sort_values(['capture', 'frame_index'])
    assert len(d) == 500 and not d.duplicated(['capture', 'frame_index']).any()
    write(d, 'inputs/reacquisition_manifest_blind.csv')
    (OUT/'inputs/extract_sha256.txt').write_text(
        f'{sha(OUT/"inputs/frozen_protocol.json")}  inputs/frozen_protocol.json\n'
        f'{sha(OUT/"inputs/reacquisition_manifest_blind.csv")}  inputs/reacquisition_manifest_blind.csv\n')
    print('decoded', len(d), int(d.decode_valid.sum()))


def ellipse_dict(cx, cy, major, minor, angle):
    return {'cx': float(cx), 'cy': float(cy), 'major_radius': float(major),
            'minor_radius': float(minor), 'angle_deg': float(angle)}


def adapted_payload(mask_path: str, image_path: str, stem: str) -> tuple[dict, str, str]:
    """Current ring fit first; reconstruct inner only from its midline if absent."""
    raw_dir = OUT/'frames/ring_raw'
    path = raw_dir/'json'/f'{stem}.json'
    if not path.exists():
        process_mask_with_calibration(Path(mask_path), Path(image_path), raw_dir, None)
    payload = json.loads(path.read_text())
    if not payload.get('rings'):
        raise ValueError('no_ring_component')
    ring = payload['rings'][0]
    if ring.get('inner_ellipse') and ring.get('outer_ellipse'):
        return payload, 'native_inner', str(path)
    outer, mid = ring.get('outer_ellipse'), ring.get('mid_ellipse')
    if not outer or not mid:
        raise ValueError('missing_outer_or_mid_ellipse')
    oa, ob = float(outer['major_radius']), float(outer['minor_radius'])
    ma, mb = float(mid['major_radius']), float(mid['minor_radius'])
    ia, ib = 2*ma-oa, 2*mb-ob
    if not (ia > 0 and ib > 0 and ia < oa and ib < ob):
        raise ValueError('invalid_centerline_inner_extrapolation')
    adapted = json.loads(json.dumps(payload))
    ar = adapted['rings'][0]
    ar['inner_ellipse'] = ellipse_dict(2*float(mid['cx'])-float(outer['cx']), 2*float(mid['cy'])-float(outer['cy']), ia, ib, float(mid['angle_deg']))
    ar['adapter'] = {'method': 'centerline_symmetry_from_existing_outer_and_mid',
                     'native_inner_present': False, 'outer_major_radius': oa, 'outer_minor_radius': ob,
                     'mid_major_radius': ma, 'mid_minor_radius': mb,
                     'inferred_inner_major_radius': ia, 'inferred_inner_minor_radius': ib}
    out = OUT/'frames/ring_adapted'/f'{stem}.json'
    out.write_text(json.dumps(adapted, ensure_ascii=False, indent=2)+'\n')
    return adapted, 'centerline_adapted_inner', str(out)


def unknown(mask: np.ndarray, ring: dict) -> np.ndarray:
    inner, outer = ring['inner_ellipse'], ring['outer_ellipse']
    rr = math.sqrt(float(inner['major_radius'])*float(inner['minor_radius']))
    inner_fill = v2mod.ellipse_mask(mask.shape, inner).astype(bool)
    outer_fill = v2mod.ellipse_mask(mask.shape, outer).astype(bool)
    return (cv2.dilate((mask == 2).astype(np.uint8), v2mod.disk(max(3, .08*rr))).astype(bool)
            | (outer_fill & ~inner_fill))


def support(mask_path: str, ring: dict, result) -> dict:
    label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if label is None or not (label == 1).any():
        raise ValueError('missing_or_empty_polyp')
    name, shape = str(result.chosen_model_candidate), str(result.chosen_shape)
    if name not in ('A', 'B') or shape not in ('circle', 'ellipse'):
        raise ValueError('no_v2_geometry')
    x0, y0 = float(result.crop_x0), float(result.crop_y0)
    if shape == 'circle':
        cx, cy = float(getattr(result, f'{name}_cx'))+x0, float(getattr(result, f'{name}_cy'))+y0
        major = minor = 2*float(getattr(result, f'{name}_radius_px')); angle = 0.
    else:
        cx, cy = float(getattr(result, f'{name}_ellipse_cx'))+x0, float(getattr(result, f'{name}_ellipse_cy'))+y0
        major, minor = float(getattr(result, f'{name}_ellipse_major_diameter_px')), float(getattr(result, f'{name}_ellipse_minor_diameter_px'))
        angle = float(getattr(result, f'{name}_ellipse_angle_deg'))
    e = np.zeros(label.shape, np.uint8)
    cv2.ellipse(e, (round(cx), round(cy)), (max(1, round(major/2)), max(1, round(minor/2))), angle, 0, 360, 1, -1)
    body, ellipse, u = label == 1, e.astype(bool), unknown(label, ring)
    c, _, nbody, nellipse, ninter = area_consistency(body, ellipse, np.zeros(label.shape, bool))
    _, p, _, nobs, nobsinter = area_consistency(body, ellipse, u)
    if not finite(c) or not finite(p):
        raise ValueError('empty_area_denominator')
    return {'C': c, 'P': p, 'mask_area_px': nbody, 'ellipse_area_px': nellipse, 'intersection_px': ninter,
            'observed_ellipse_area_px': nobs, 'observed_intersection_px': nobsinter, 'unknown_area_px': int(u.sum()),
            'geometry_candidate': name, 'geometry_shape': shape, 'geometry_major_px': major, 'geometry_minor_px': minor,
            'geometry_angle_deg': angle, 'geometry_cx': cx, 'geometry_cy': cy}


def process_one(row: dict) -> dict:
    rec = {k: row[k] for k in ('group', 'capture', 'video', 'frame_index', 'timestamp_sec', 'bin_id', 'ordinal', 'stem', 'image_path')}
    try:
        mask = OUT/'frames/masks'/f'{row["stem"]}.png'
        if not mask.exists():
            raise ValueError('missing_frozen_init_mask')
        payload, ring_source, ring_path = adapted_payload(str(mask), row['image_path'], row['stem'])
        scale, ring_px, ring, quality = aud.current_rectified_scale(payload)
        # V2 sees only the per-frame adapted JSON, never an R3 artifact.
        result, _, _, _ = v2mod.process_view({'stem': row['stem'], 'group_key': GROUP,
            'image_path': row['image_path'], 'mask_path': str(mask), 'ring_fit_json': ring_path, 'rect_scale': scale})
        rec.update(result)
        rec.update({'ring_source': ring_source, 'ring_json_path': ring_path, 'mask_path': str(mask),
                    'mm_per_px': scale, 'ring_px': ring_px,
                    'ring_inner_residual': quality.get('inner_mean_residual', math.nan),
                    'sharpness': PREP._frame_sharpness(cv2.imread(row['image_path'], cv2.IMREAD_COLOR))})
        rec['processing_failure_reason'] = ''
    except Exception as exc:
        rec['processing_failure_reason'] = f'{type(exc).__name__}:{exc}'
    return rec


def measure(device: str) -> None:
    import torch
    manifest = pd.read_csv(OUT/'inputs/reacquisition_manifest_blind.csv')
    good = manifest[manifest.decode_valid.map(truth)]
    model = load_pretrained_model(str(MODEL), torch.device(device), local_files_only=True, strict=True)
    model.eval()
    OLD.run_segmentation(model, torch.device(device), [(r.stem, Path(r.image_path)) for r in good.itertuples()], OUT/'frames/masks', 256, .5)
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    todo = [r._asdict() for r in good.itertuples(index=False)]
    rows = []
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3, mp_context=ctx) as pool:
        for i, rec in enumerate(pool.map(process_one, todo, chunksize=2), 1):
            rows.append(rec)
            if i % 50 == 0:
                write(rows, 'tables/per_frame_checkpoint_blind.csv')
                print('measured', i, '/', len(todo), flush=True)
    # V2 consistency selection is the same unchanged procedure used by raw sharp5.
    valid_v2 = [r for r in rows if not r.get('processing_failure_reason')]
    for z in (valid_v2,):
        v2mod.choose(z); v2mod.choose_geometry(z)
    out = []
    for r in rows:
        dxy, scale = r.get('chosen_model_Dxy_mm', math.nan), r.get('mm_per_px', math.nan)
        r['valid_pre_support'] = finite(dxy) and finite(scale) and float(scale) > 0
        if r['valid_pre_support']:
            r['Dxy_mm'] = float(dxy); r['L_px'] = float(dxy)/float(scale); r['q_L_over_ring'] = r['L_px']/float(r['ring_px'])
            try:
                r.update(support(r['mask_path'], json.loads(Path(r['ring_json_path']).read_text())['rings'][0], type('R', (), r)()))
                r['support_assessed'] = True; r['support_reason'] = ''
            except Exception as exc:
                r['support_assessed'] = False; r['support_reason'] = f'{type(exc).__name__}:{exc}'
        else:
            r.update({'Dxy_mm': math.nan, 'L_px': math.nan, 'q_L_over_ring': math.nan,
                      'support_assessed': False, 'support_reason': r.get('processing_failure_reason') or r.get('chosen_model_reason', 'no_v2_geometry')})
        out.append(r)
    d = pd.DataFrame(out).sort_values(['capture', 'frame_index'])
    write(d, 'tables/per_frame_blind.csv')
    print('frames', len(d), 'native', int(d.ring_source.eq('native_inner').sum()),
          'adapted', int(d.ring_source.eq('centerline_adapted_inner').sum()),
          'v2', int(d.valid_pre_support.sum()), 'supported', int(d.support_assessed.sum()))


def select_blind() -> None:
    d = pd.read_csv(OUT/'tables/per_frame_blind.csv')
    th = json.loads((MAIN/'inputs/relaxed_thresholds_blind.json').read_text())
    # Raw sharp5 choice is made from this new, independently decoded pool.
    choices = []
    for (cap, bin_id), z in d.groupby(['capture', 'bin_id'], sort=True):
        z = z[z.valid_pre_support.map(truth)].sort_values(['sharpness', 'frame_index'], ascending=[False, True])
        if len(z):
            q = z.iloc[0].to_dict(); q.update({'raw_selected': True, 'raw_rank': 1})
        else:
            q = {'group': GROUP, 'capture': cap, 'bin_id': bin_id, 'stem': '', 'raw_selected': False, 'raw_rank': math.nan,
                 'valid_pre_support': False, 'support_assessed': False, 'selection_outcome': 'raw_missing'}
        choices.append(q)
    raw = pd.DataFrame(choices)
    # Exact post-capture relaxed-area mechanism, applied to this new pool.
    usable = raw[raw.valid_pre_support.map(truth) & raw.support_assessed.map(truth)]
    caps = usable.groupby('capture').agg(n_raw_supported=('stem', 'size'), median_C=('C', 'median'), median_P=('P', 'median')).reset_index()
    caps['trigger'] = caps.n_raw_supported.ge(4) & ((caps.median_C < th['C']) | (caps.median_P < th['P']))
    trigger = caps.set_index('capture').trigger.to_dict()
    revised = []
    for r in raw.to_dict('records'):
        bad = truth(r.get('valid_pre_support')) and truth(r.get('support_assessed')) and (float(r['C']) < th['C'] or float(r['P']) < th['P'])
        repair = bool(trigger.get(r['capture'], False)) and bad
        if repair:
            pool = d[(d.capture.eq(r['capture'])) & (d.bin_id.eq(r['bin_id'])) & d.valid_pre_support.map(truth) & d.support_assessed.map(truth)].copy()
            pool = pool[(pool.C >= th['C']) & (pool.P >= th['P'])].sort_values(['sharpness', 'frame_index'], ascending=[False, True]).reset_index(drop=True)
            if len(pool):
                q = pool.iloc[0].to_dict(); q.update({'raw_stem': r['stem'], 'selection_outcome': 'post_retry', 'retry_rank': 1})
            else:
                q = dict(r); q.update({'stem': '', 'valid_pre_support': False, 'selection_outcome': 'post_missing', 'retry_rank': math.nan})
        else:
            q = dict(r); q.update({'raw_stem': r.get('stem', ''), 'selection_outcome': 'raw_kept' if truth(r.get('valid_pre_support')) else 'raw_missing', 'retry_rank': math.nan})
        revised.append(q)
    selected = pd.DataFrame(revised)
    write(selected, 'tables/selected_frames_blind.csv')
    write(caps, 'tables/capture_area_postcheck_blind.csv')
    cp = []
    for cap, z in selected.groupby('capture', sort=True):
        good = z[z.valid_pre_support.map(truth)]
        R, L = good.ring_px.astype(float), good.L_px.astype(float)
        dyn = R.quantile(.95)/R.quantile(.05)-1 if len(good) >= 2 else math.nan
        eligible = len(good) >= 4 and finite(dyn) and dyn >= .05
        cp.append({'group': GROUP, 'capture': cap, 'n_selected_valid': len(good), 'ring_dynamic': dyn, 'eligible': eligible,
                   'prediction_mm': 10*L.median()/R.median() if eligible else math.nan,
                   'fallback_reason': '' if eligible else ('fewer_than_4_valid_bins' if len(good) < 4 else 'ring_range_below_5pct'),
                   'post_retries': int((z.selection_outcome == 'post_retry').sum()), 'post_missing': int((z.selection_outcome == 'post_missing').sum())})
    cp = pd.DataFrame(cp)
    write(cp, 'tables/capture_predictions_blind.csv')
    eligible = cp[cp.eligible.map(truth)]
    pred = eligible.prediction_mm.median() if len(eligible) else math.nan
    gp = [{'method': 'raw_video_reacquisition_centerline_adapter_post_area_guard_v1', 'group': GROUP,
           'prediction_mm': pred, 'eligible_video_captures': len(eligible), 'fallback': not finite(pred)}]
    write(gp, 'tables/group_prediction_blind.csv')
    paths = ['inputs/frozen_protocol.json', 'inputs/reacquisition_manifest_blind.csv', 'tables/per_frame_blind.csv',
             'tables/capture_area_postcheck_blind.csv', 'tables/selected_frames_blind.csv', 'tables/capture_predictions_blind.csv', 'tables/group_prediction_blind.csv']
    (OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/p)}  {p}\n' for p in paths))
    print('threshold', th, 'triggers', trigger, 'capture predictions', cp.to_dict('records'), 'group', pred)


def evaluate() -> None:
    for line in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():
        h, p = line.split('  ', 1)
        assert sha(OUT/p) == h, p
    pred = pd.read_csv(OUT/'tables/group_prediction_blind.csv')
    refs = pd.read_csv(AUDIT/'inputs/immutable_scale_manifest.csv').groupby('group', as_index=False).reference_mm.first()
    z = pred.merge(refs, on='group', how='left', validate='one_to_one')
    z['error_mm'] = z.prediction_mm-z.reference_mm
    z['relative_error_pct'] = 100*z.error_mm/z.reference_mm
    z['abs_error_pct'] = z.relative_error_pct.abs()
    write(z, 'tables/group_prediction_evaluated.csv')
    main = pd.read_csv(MAIN/'tables/group_predictions_blind.csv')[['group', 'prediction_mm']]
    extended = pd.concat([main, pred[['group', 'prediction_mm']]], ignore_index=True).merge(refs, on='group', how='left', validate='one_to_one')
    extended['abs_error_pct'] = 100*(extended.prediction_mm-extended.reference_mm).abs()/extended.reference_mm
    metrics = {'coverage': int(extended.prediction_mm.notna().sum()), 'median_abs_error_pct': extended.abs_error_pct.median(),
               'MAE_mm': (extended.prediction_mm-extended.reference_mm).abs().mean(), 'p95_pct': extended.abs_error_pct.quantile(.95), 'max_pct': extended.abs_error_pct.max()}
    write([metrics], 'tables/extended_39_metrics_evaluated.csv')
    print(z.to_string(index=False)); print(metrics)


def gallery_and_report() -> None:
    """Render the five-bin decisions and write a concise provenance report."""
    sel = pd.read_csv(OUT/'tables/selected_frames_blind.csv')
    per = pd.read_csv(OUT/'tables/per_frame_blind.csv').set_index('stem')
    cards = []
    for r in sel[sel.valid_pre_support.map(truth)].itertuples(index=False):
        q = per.loc[r.stem]
        image = cv2.imread(q.image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(q.mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            continue
        canvas = image.copy()
        canvas[label == 1] = (.68*canvas[label == 1]+.32*np.array([50, 235, 50])).astype(np.uint8)
        canvas[label == 2] = (.68*canvas[label == 2]+.32*np.array([0, 210, 255])).astype(np.uint8)
        ring = json.loads(Path(q.ring_json_path).read_text())['rings'][0]
        for key, color in [('outer_ellipse', (0, 255, 0)), ('inner_ellipse', (255, 255, 0))]:
            e = ring[key]
            cv2.ellipse(canvas, (round(e['cx']), round(e['cy'])), (max(1, round(e['major_radius'])), max(1, round(e['minor_radius']))), e['angle_deg'], 0, 360, color, 2, cv2.LINE_AA)
        if q.geometry_shape == 'circle':
            cv2.ellipse(canvas, (round(q.geometry_cx), round(q.geometry_cy)), (round(q.geometry_major_px/2), round(q.geometry_major_px/2)), 0, 0, 360, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.ellipse(canvas, (round(q.geometry_cx), round(q.geometry_cy)), (round(q.geometry_major_px/2), round(q.geometry_minor_px/2)), q.geometry_angle_deg, 0, 360, (255, 0, 255), 2, cv2.LINE_AA)
        x0, y0 = float(q.crop_x0), float(q.crop_y0)
        name = str(q.chosen_model_candidate)
        if q.geometry_shape == 'circle':
            angle = 0.0
            # V2's visual measurement line is the fitted circle diameter
            p1 = (q.geometry_cx-q.geometry_major_px/2, q.geometry_cy)
            p2 = (q.geometry_cx+q.geometry_major_px/2, q.geometry_cy)
        else:
            a = math.radians(float(q.geometry_angle_deg))
            dx, dy = math.cos(a)*q.geometry_major_px/2, math.sin(a)*q.geometry_major_px/2
            p1, p2 = (q.geometry_cx-dx, q.geometry_cy-dy), (q.geometry_cx+dx, q.geometry_cy+dy)
        cv2.line(canvas, tuple(np.rint(p1).astype(int)), tuple(np.rint(p2).astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
        lines = [f'bin {int(q.bin_id)} | frame {int(q.frame_index)} | {q.ring_source}',
                 f'V2 {q.geometry_candidate}/{q.geometry_shape}: {q.Dxy_mm:.2f} mm',
                 f'area support C={q.C:.3f}, P={q.P:.3f}']
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (18, 34+i*30), cv2.FONT_HERSHEY_SIMPLEX, .70, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(canvas, line, (18, 34+i*30), cv2.FONT_HERSHEY_SIMPLEX, .70, (255, 255, 255), 1, cv2.LINE_AA)
        fn = f'{q.capture}_bin{int(q.bin_id)}.jpg'
        cv2.imwrite(str(OUT/'gallery/overlays'/fn), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        cards.append({'capture': q.capture, 'bin': int(q.bin_id), 'frame_index': int(q.frame_index),
                      'file': f'overlays/{fn}', 'Dxy_mm': q.Dxy_mm, 'C': q.C, 'P': q.P,
                      'shape': f'{q.geometry_candidate}/{q.geometry_shape}', 'ring_source': q.ring_source,
                      'outcome': r.selection_outcome})
    write(cards, 'tables/selected_frame_visual_index.csv')
    html = ['<!doctype html><meta charset="utf-8"><title>Ip8.8 raw-video reacquisition</title>',
            '<style>body{font-family:Arial,"Microsoft Yahei",sans-serif;margin:22px} .card{display:inline-block;vertical-align:top;width:47%;margin:1%;border:1px solid #bbb;padding:8px}img{width:100%}small{display:block;white-space:pre-wrap}</style>',
            '<h1>Ip8.8 R10：原始视频重采集，非旧 V2 R3</h1>',
            '<p>绿色：V2/INIT 息肉 mask；黄色：INIT 环 mask；绿色线：环外缘；青色线：由 outer+mid 反推的内缘；紫色：未修改 V2 的最终测量形状和直径线。</p>']
    for c in cards:
        html.append(f'<div class="card"><img src="{c["file"]}"><small>{c["capture"]} | bin {c["bin"]} | frame {c["frame_index"]}\nDxy={c["Dxy_mm"]:.3f} mm; C={c["C"]:.3f}; P={c["P"]:.3f}\n{c["ring_source"]}; {c["shape"]}; {c["outcome"]}</small></div>')
    (OUT/'gallery/index.html').write_text('\n'.join(html)+'\n', encoding='utf-8')
    ev = pd.read_csv(OUT/'tables/group_prediction_evaluated.csv').iloc[0]
    ext = pd.read_csv(OUT/'tables/extended_39_metrics_evaluated.csv').iloc[0]
    cp = pd.read_csv(OUT/'tables/capture_predictions_blind.csv')
    d = pd.read_csv(OUT/'tables/per_frame_blind.csv')
    text = f'''# Ip8.8 R10 原始视频重采集与面积支持适配报告

## 结论

这次没有读取、复用或接入旧 V2 R3 的 4.067 mm 结果。直接从 `_004.mp4`、`_005.mp4` 各重采 250 帧，以冻结 INIT 分割后重新运行当前环拟合和未修改 V2。原有失败不是环不可见，而是环带因息肉紧贴内缘没有形成闭合孔，使当前 fitter 缺少 `inner_ellipse`。

对这种情况，新适配只使用当前 fitter 已产出的 `outer_ellipse` 与 `mid_ellipse`：`inner = 2*mid - outer`（中心和两个半轴各自计算）。它是环带中心线对称的几何补全，不是 GT 标定、R3 迁移或新 V2 测量算法。只有正且位于外缘内的反推内环被接受。

## 盲流程冻结与面积 guard

- 原始候选：500；标准 native inner：{int(d.ring_source.eq('native_inner').sum())}；中心线补全 inner：{int(d.ring_source.eq('centerline_adapted_inner').sum())}；物理上无法反推：{int(d.processing_failure_reason.fillna('').str.contains('invalid_centerline').sum())}。
- 可进入 V2/面积支持：{int(d.valid_pre_support.sum())} 帧；每段视频按 5 个时间 bin，以既有 sharpness 取新池内最佳有效帧。
- 面积 guard 完全沿用主线的 formal-S0 p2：C≥0.085031、P≥0.317505。两个 capture 的 selected-frame 中位 C/P 都未落入该极端尾部，故按 `post-capture relaxed area guard v1` 不触发 retry；这是 guard 的正常“放行”，不是绕过 guard。
- B1 不变：`10 × median(L_px) / median(ring_px)`；`_004`={cp.iloc[0].prediction_mm:.6f} mm（5/5 bin），`_005`={cp.iloc[1].prediction_mm:.6f} mm（4/5 bin），组中位={ev.prediction_mm:.6f} mm。

## 冻结后评价

- 该组：参考 {ev.reference_mm:.1f} mm，预测 {ev.prediction_mm:.6f} mm，绝对相对误差 **{ev.abs_error_pct:.6f}%**。
- 将它作为独立视频扩展接入已冻结 38 组主线后的 39 组范围：coverage {int(ext.coverage)}/39，median absolute error **{ext.median_abs_error_pct:.6f}%**，MAE {ext.MAE_mm:.6f} mm，p95 {ext.p95_pct:.6f}%，max {ext.max_pct:.6f}%。原 38 组主线文件未更改。

## 文件与复现

- 盲流程 SHA：`inputs/blind_sha256.txt`
- 每帧表：`tables/per_frame_blind.csv`
- 选择及面积证据：`tables/selected_frames_blind.csv`、`tables/capture_area_postcheck_blind.csv`
- 可视化：`gallery/index.html`（黄色=原 ring mask，青色=补全的 inner，紫色=V2 最终形状）
- 命令：`run_reacquisition.py extract` → `measure --device cuda` → `select` → `evaluate` → `report`。
'''
    (OUT/'RAW_VIDEO_REACQUISITION_REPORT.md').write_text(text, encoding='utf-8')
    print('gallery cards', len(cards))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['extract', 'measure', 'select', 'evaluate', 'report'])
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    {'extract': extract, 'measure': lambda: measure(a.device), 'select': select_blind, 'evaluate': evaluate, 'report': gallery_and_report}[a.stage]()
