"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import math as math
import numpy as np
VERSION84 = 'V8.4.4-conflict-Q-nearest-current-candidate-1.0'
POLICY = {'version': 'V8.4.4-conflict-Q-nearest-current-candidate-1.0', 'Q_anchor_fraction': 0.5, 'Q_total_weight_with_P': 0.35, 'Q_total_weight_zero_P': 1.0, 'anchor_residual': 'same_native_pixel_circle_or_ellipse_residual_as_P', 'diagnostic_only_reasons': ['P_inlier_fraction_below_0.60', 'legacy_arc_coverage_below_0.20', 'weak_evidence_direction_coverage_below_0.50', 'Q_interval_support_below_0.80'], 'unchanged_actual_residual_checks': ['P_p90_residual_too_large', 'legacy_normalized_residual_above_0.08'], 'unchanged_conflict_check': 'P_dominant_serious_Q_conflict', 'unchanged_identifiability': True, 'no_Q_behavior': 'original_current_V2_unchanged', 'original_optimizer_source_sha256': '3ee624682168e5790bf77444c43a09f528dbf3a6fa5b5e11c226c9d9a3c84155', 'adapted_optimizer_source_sha256': '9224356feec8c316ef89782ddcaee0ce9ba03b05b91c8e2e2ea57c17c476f94e'}

def run_policy(frame,data,current,seed,cap=None,mode='anchor'):
    assert mode in ['anchor','gate_only']
    saved=(gc.metrics,gc.optimise,v81.evidence_gate)
    try:
        v81.evidence_gate=evidence_gate43
        if mode=='anchor':gc.metrics=metrics43;gc.optimise=optimise43
        result=v81.run_B(frame,data,current,seed,cap)
    finally:gc.metrics,gc.optimise,v81.evidence_gate=saved
    result['evidence_policy']=dict(POLICY,mode=mode)
    for candidate in result['candidates']+result['diagnostic_candidates']+([result['chosen']] if result['chosen'] else []):
        candidate.provenance.update(Geometry_scoring_version=VERSION84,Gate_policy_version=VERSION84,evidence_policy_mode=mode)
    return result


def evidence_gate43(p,data,m):
    old=ORIGINAL_GATE(p,data,m)
    removed=[r for r in old if r in POLICY['diagnostic_only_reasons']]
    m.update(observation_availability_diagnostics=removed,original_gate_reasons=old,gate_policy_version=VERSION84)
    return [r for r in old if r not in POLICY['diagnostic_only_reasons']]


def metrics43(p,data,shape,plan):
    m=ORIGINAL_METRICS(p,data,shape,plan);Q=data['Q'];sel=plan['score_qi'];alpha=POLICY['Q_anchor_fraction'];rr=data['rr']
    ar=np.abs(anchor_residual(p,Q,shape));selected_loss=float(plan['score_qw']@gc.huber_loss(ar[sel]/rr)) if len(sel) else 0.
    original=m['Q_loss'];mixed=(1-alpha)*original+alpha*selected_loss
    m.update(score=m['P_loss']+plan['mass']*mixed,Q_interval_loss=original,Q_anchor_loss=selected_loss,Q_loss=mixed,Q_anchor_fraction=alpha,Q_anchor_support_arc_weighted=float(numerical.weights(Q)@(ar<=max(1.,.01*rr))) if Q else math.nan,Q_anchor_median_px=float(np.median(ar)) if Q else math.nan,Q_anchor_p90_px=float(np.quantile(ar,.9)) if Q else math.nan,Q_anchor_selected_median_px=float(np.median(ar[sel])) if len(sel) else math.nan,original_interval_score=m['P_loss']+plan['mass']*original)
    return m


def anchor_residual(p,Q,shape):
    return numerical.residual_true(p,np.array([q['q'] for q in Q]).reshape(-1,2),shape)


# Bind relocated dependencies after defining functions (including override hooks).
from . import geometry_competition as gc
from . import soft_geometry as numerical
from . import v81_components as v81
from .geometry_competition import metrics as ORIGINAL_METRICS
from .geometry_competition_slice import optimise as optimise43
from .v81_components import evidence_gate as ORIGINAL_GATE
