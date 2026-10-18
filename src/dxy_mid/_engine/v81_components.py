"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import math as math
import numpy as np
RUN_VERSION = 'V8.1-observed-boundary-evidence-gate-1.0'
RULES = {'normal_probe_px': [3.0, 5.0], 'inside_min': 0.8, 'background_min': 0.8, 'ring_clearance_px': 6.0, 'image_contrast_min': 5.0, 'strong_P_min': 80, 'strong_P_sectors': 12, 'strong_P_per_sector': 4, 'serious_Q_width_multiplier': 1.0, 'serious_Q_pixel_margin': 2.0, 'serious_Q_rr_margin': 0.01, 'serious_Q_arc_fraction_max': 0.2}
VERSION = 'V7-phase2-directional-PQ-1.2'

def existing_ring_check(frame,cached=None):
    """Finite existing closed-inner fitting checked by the unchanged U2 criteria."""
    if cached is not None and cached.valid:return cached,[]
    if cached is None:
        g,e=rc.recover(frame.labels,frame.image,frame.inner_diameter_mm)
        cached=RingResult(g['valid'],g['reason'],{k:g[k] for k in ['inner','outer','centerline','branch','attempts','diagnostics'] if k in g},provenance(frame,'u2_visible_arcs'),e)
    if cached.valid:return cached,[]
    m=rc.preprocess_ring_mask(frame.labels==2);components=rc.select_components(m,1);attempts=[]
    if not components:return cached,[dict(branch='existing_closed_inner',valid=False,reason='ring_component_missing')]
    mask=components[0];native=rc.fit_single_component(mask);inner=native.get('inner_ellipse');outer=native.get('outer_ellipse')
    if inner is None or outer is None:return cached,[dict(branch='existing_closed_inner',valid=False,reason='closed_inner_or_outer_missing')]
    ip,op,xp=rc.ring_points(mask,outer,frame.image);valid,metrics=rc.validate(mask,inner,outer,frame.inner_diameter_mm,ip)
    reasons=[]
    if len(op)<25 or rc.coverage(op,outer)<.35 or rc.stability(op,outer)>.20:valid=False;reasons.append('observable_outer_fit_unstable')
    if len(ip)<25:valid=False;reasons.append('observable_inner_insufficient')
    attempt=dict(branch='existing_closed_inner',valid=bool(valid),**metrics,outer_coverage=float(rc.coverage(op,outer)),outer_stability=float(rc.stability(op,outer)),inner_points=len(ip),outer_points=len(op),inner=inner,outer=outer)
    attempt['reason']=metrics.get('reason','')+(';'+';'.join(reasons) if reasons else '');attempts.append(attempt)
    if not valid:return cached,attempts
    centerline={k:(inner[k]+outer[k])/2 for k in ['cx','cy','major_radius','minor_radius']};centerline['angle_deg']=float(np.rad2deg(np.angle(np.exp(2j*np.deg2rad([inner['angle_deg'],outer['angle_deg']])).mean()))/2)
    geometry=dict(inner=inner,outer=outer,centerline=centerline,branch='existing_closed_inner_validated',attempts=attempts,diagnostics=metrics)
    return RingResult(True,'ok',geometry,dict(provenance(frame,'existing_closed_inner_validated'),implementation_version=RUN_VERSION),dict(inner_points=ip,outer_points=op,rejected_points=xp)),attempts


def evidence_gate(p,data,m):
    why=FROZEN_QUALITY(p,data,m);P=data['P'];rr=data['rr']
    inlier=np.abs(numerical.residual_true(p,P,'circle' if abs(p[2]-p[3])<1e-10 else 'ellipse'))<=max(2.,.04*rr)
    counts=np.bincount(gc.sector(P[inlier],p[:2]),minlength=16)
    strong=len(P)>=RULES['strong_P_min'] and int((counts>=RULES['strong_P_per_sector']).sum())>=RULES['strong_P_sectors'] and m['P_inlier_ratio']>=.60 and m['P_p90_px']<=max(3.,.10*rr)
    m['P_dominant_gate']=bool(strong);m['P_inlier_dense_sectors']=int((counts>=4).sum());m['Q_severe_conflict_arc_weighted']=math.nan
    if strong and data['Q']:
        widths=np.array([q['hi']/data['fraction'] for q in data['Q']])
        residual=numerical.interval_residual(p,data['Q'])
        severe=residual>RULES['serious_Q_width_multiplier']*widths+max(RULES['serious_Q_pixel_margin'],RULES['serious_Q_rr_margin']*rr)
        fraction=float(numerical.weights(data['Q'])@severe);m['Q_severe_conflict_arc_weighted']=fraction;m['Q_severe_conflict_count_fraction']=float(severe.mean())
        why=[r for r in why if r!='Q_interval_support_below_0.80']
        if fraction>RULES['serious_Q_arc_fraction_max']:why.append('P_dominant_serious_Q_conflict')
    return why


def run_B(frame,data,current,seeds,q_cap=None):
    # Existing solver uses its module-global quality hook; process-isolated and
    # restored after each call, never modifies the frozen source file or CONFIG.
    previous=gc.quality_reasons;gc.quality_reasons=evidence_gate
    try:r=run(frame,data,current,method='B',seed=seeds,q_cap=q_cap)
    finally:gc.quality_reasons=previous
    r['gate_version']=RUN_VERSION
    for c in r['candidates']+r['diagnostic_candidates']+([r['chosen']] if r['chosen'] else []):c.provenance.update(Geometry_numerical_version=VERSION,Boundary_Gate_version=RUN_VERSION)
    return r


# Bind relocated dependencies after defining functions (including override hooks).
from . import boundary_v7 as boundary_v7
from . import geometry_competition as gc
from . import ring_component as rc
from . import soft_geometry as numerical
from .contracts import RingResult as RingResult
from .contracts import provenance as provenance
from .geometry_competition import quality_reasons as FROZEN_QUALITY
from .pq_geometry_component import run as run
