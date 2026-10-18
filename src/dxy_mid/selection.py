"""V2: image sharpness of each target independently of the other target mask.

Contact is permitted for quality assessment, never a geometry/contour claim.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np


@dataclass(frozen=True)
class Config:
    analysis_width: int = 640
    boundary_band: int = 3
    profile_radius: int = 9
    peak_search_radius: int = 3
    samples_per_contour: int = 160
    min_roi_pixels: int = 40
    min_boundary_samples: int = 20
    min_contrast: float = 0.045
    min_continuity: float = 0.20
    min_target_score: float = 0.25
    max_highlight: float = 0.35
    max_clipped_fraction: float = 0.12
    artifact_weight: float = 0.25
    per_segment: int = 2
    min_interval_sec: float = 0.10
    margin_ratio: float = 0.05
    segments: int = 5
    candidates_per_segment: int = 24


def target_metrics(gray, own_mask, target, glare, cfg):
    # The function receives ONLY this target's binary ROI; it cannot inspect
    # the other label or reject adjacency/contact.
    mask = own_mask.astype('uint8')
    n = int(mask.sum())
    prefix = 'ring' if target == 2 else 'polyp'
    out = {prefix+'_status': 'ok', prefix+'_area_fraction': n/mask.size}
    if n < cfg.min_roi_pixels:
        out.update({prefix+'_status': 'localization_failed_or_target_absent',
                    prefix+'_score': np.nan, prefix+'_continuity': np.nan})
        return out
    # Largest component gives a coarse location, never a fitted hidden outline.
    nc, cc, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out[prefix+'_largest_component_fraction'] = float(stats[largest, cv2.CC_STAT_AREA]/n)
    mask = (cc == largest).astype('uint8')
    neighborhood = cv2.dilate(mask, np.ones((7, 7), 'uint8')) > 0
    out[prefix+'_highlight_fraction'] = float(glare[neighborhood].mean())
    kernel = np.ones((2*cfg.boundary_band+1,)*2, 'uint8')
    band = cv2.dilate(mask, kernel) != cv2.erode(mask, kernel)
    band &= ~glare
    if band.any():
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        out[prefix+'_tenengrad'] = float(np.mean((gx*gx+gy*gy)[band]))
        out[prefix+'_laplacian_variance'] = float(cv2.Laplacian(gray, cv2.CV_32F)[band].var())
    # All contours include ring inner and outer edges. Uniform arc-length points
    # avoid weighting ROI area or a coarse staircase boundary as sharpness.
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    points, normals = [], []
    for contour in contours:
        xy = contour[:, 0, :].astype('float32')
        if len(xy) < 24:
            continue
        lengths = np.linalg.norm(np.roll(xy, -1, axis=0)-xy, axis=1)
        arc = np.cumsum(lengths)
        ii = np.searchsorted(arc, np.linspace(0, arc[-1], cfg.samples_per_contour, endpoint=False))
        ii = np.clip(ii, 0, len(xy)-1)
        tangent = xy[(ii+4) % len(xy)] - xy[(ii-4) % len(xy)]
        normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1)
        points.append(xy[ii]); normals.append(normal)
    if not points:
        out.update({prefix+'_status': 'roi_outline_too_small', prefix+'_score': np.nan})
        return out
    pts = np.concatenate(points); normals = np.concatenate(normals)
    h, w = gray.shape
    x = pts[:, 0].astype(int); y = pts[:, 1].astype(int)
    clipped = (x < cfg.profile_radius+1) | (y < cfg.profile_radius+1) | (x >= w-cfg.profile_radius-1) | (y >= h-cfg.profile_radius-1)
    out[prefix+'_clipped_boundary_fraction'] = float(clipped.mean())
    usable = ~clipped
    off = np.arange(-cfg.profile_radius, cfg.profile_radius+1, dtype='float32')
    mx = pts[:, 0, None] + normals[:, 0, None]*off
    my = pts[:, 1, None] + normals[:, 1, None]*off
    prof = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    prof_glare = cv2.remap(glare.astype('uint8'), mx, my, cv2.INTER_NEAREST).any(axis=1)
    # Bright reflections are excluded; adjacency to the other target is not.
    usable &= ~prof_glare
    out[prefix+'_usable_boundary_fraction'] = float(usable.mean())
    out[prefix+'_boundary_samples'] = int(usable.sum())
    if usable.sum() < cfg.min_boundary_samples:
        out.update({prefix+'_status': 'image_edge_samples_unavailable', prefix+'_score': np.nan,
                    prefix+'_continuity': np.nan})
        return out
    p = prof[usable]
    smooth = cv2.GaussianBlur(p, (5, 1), 0.7)
    deriv = np.abs(np.diff(smooth, axis=1))
    # Anchor to this ROI's coarse border. A narrow +/-3px search helps avoid
    # a sharper but unrelated edge farther away (e.g. the other ring rim).
    lo = cfg.profile_radius-cfg.peak_search_radius
    hi = cfg.profile_radius+cfg.peak_search_radius
    win = deriv[:, lo:hi]
    peak = win.max(axis=1)
    contrast = np.quantile(smooth, .9, axis=1)-np.quantile(smooth, .1, axis=1)
    evidence = (contrast >= cfg.min_contrast) & (peak >= 0.0075)
    continuity = float(evidence.mean())
    # Peak/range estimates edge steepness and cancels black/white amplitude.
    # Score is a mean over samples, not a sum over target size.
    concentration = np.clip(peak/(contrast+0.015)/0.40, 0, 1)
    sharp = float(np.median(concentration[evidence])) if evidence.any() else 0.0
    out.update({prefix+'_continuity': continuity, prefix+'_edge_concentration': sharp,
                prefix+'_edge_gradient_median': float(np.median(peak)),
                prefix+'_contrast_median': float(np.median(contrast)),
                prefix+'_score': sharp*np.sqrt(continuity)})
    if continuity < cfg.min_continuity:
        out[prefix+'_status'] = 'low_image_edge_signal'
    elif out[prefix+'_score'] < cfg.min_target_score:
        out[prefix+'_status'] = 'target_image_blur_proxy'
    return out


def evaluate(image, label, cfg=Config(), presence=None):
    """presence may certify absent targets; an empty model mask cannot do so."""
    if image is None:
        return dict(quality_reason='decode_failed', joint_eligible=False, ring_eligible=False,
                    ring_score=np.nan, polyp_score=np.nan, joint_score=np.nan)
    if label is None or label.shape[:2] != image.shape[:2] or not set(np.unique(label)).issubset({0, 1, 2}):
        return dict(quality_reason='roi_localization_invalid', joint_eligible=False, ring_eligible=False,
                    ring_score=np.nan, polyp_score=np.nan, joint_score=np.nan)
    scale = cfg.analysis_width/image.shape[1]
    size = (cfg.analysis_width, round(image.shape[0]*scale))
    im = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float32')/255
    glare = (im.min(axis=2) >= 235) & (im.max(axis=2) >= 248)
    glare = cv2.dilate(glare.astype('uint8'), np.ones((3, 3), 'uint8')) > 0
    union = cv2.dilate((label > 0).astype('uint8'), np.ones((7, 7), 'uint8')) > 0
    highlight = float(glare[union].mean()) if union.any() else np.nan
    out = dict(highlight_fraction=highlight, global_laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_32F).var()))
    for target in [2, 1]:
        out.update(target_metrics(gray, label == target, target, glare, cfg))
        name = 'ring' if target == 2 else 'polyp'
        if presence and presence.get(name) == 'absent_confirmed':
            out[name+'_status'] = 'target_absent_confirmed'
            out[name+'_score'] = np.nan
    reasons = []
    for name in ['ring', 'polyp']:
        if out[name+'_status'] != 'ok':
            reasons.append(name+':'+out[name+'_status'])
        if out.get(name+'_clipped_boundary_fraction', 0) > cfg.max_clipped_fraction:
            reasons.append(name+':target_clipped')
        if out.get(name+'_highlight_fraction', 1) > cfg.max_highlight:
            reasons.append(name+':excessive_highlight')
    r, p = out['ring_score'], out['polyp_score']
    clipped = max(out.get('ring_clipped_boundary_fraction', 0), out.get('polyp_clipped_boundary_fraction', 0))
    artifact = (highlight if np.isfinite(highlight) else 1) + clipped
    out['artifact_score'] = artifact
    ring_artifact = out.get('ring_highlight_fraction', 1)+out.get('ring_clipped_boundary_fraction', 0)
    out['ring_only_score'] = r-cfg.artifact_weight*ring_artifact
    out['joint_equal_score'] = .5*(r+p)-cfg.artifact_weight*artifact
    out['joint_score'] = .5*min(r, p)+.25*(r+p)-cfg.artifact_weight*artifact if np.isfinite(r+p) else np.nan
    out['ring_eligible'] = out['ring_status'] == 'ok' and out.get('ring_clipped_boundary_fraction', 0) <= cfg.max_clipped_fraction and out.get('ring_highlight_fraction', 1) <= cfg.max_highlight
    out['polyp_eligible'] = out['polyp_status'] == 'ok' and out.get('polyp_clipped_boundary_fraction', 0) <= cfg.max_clipped_fraction and out.get('polyp_highlight_fraction', 1) <= cfg.max_highlight
    out['joint_eligible'] = out['ring_eligible'] and out['polyp_eligible']
    out['quality_reason'] = ';'.join(reasons) if reasons else 'both_targets_pass_image_quality'
    return out


def select(rows, method, cfg=Config()):
    """Call once per video segment. No group/video name affects ordering."""
    gate = 'ring_eligible' if method == 'B_ring' else 'joint_eligible'
    score = 'ring_only_score' if method == 'B_ring' else 'joint_score'
    ranked = sorted([r for r in rows if r.get(gate, False) and np.isfinite(r[score])],
                    key=lambda r: (-r[score], r['frame_index']))
    chosen = []
    for row in ranked:
        if all(abs(row['timestamp_sec']-r['timestamp_sec']) >= cfg.min_interval_sec for r in chosen):
            chosen.append(row)
        if len(chosen) == cfg.per_segment:
            break
    return chosen
