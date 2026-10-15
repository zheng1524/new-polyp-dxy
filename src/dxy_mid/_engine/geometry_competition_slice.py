"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.optimize._lsq.least_squares import least_squares as least_squares
import numpy as np
CONFIG = {'sectors': 16, 'P_sufficient_per_sector': 4, 'Q_per_empty_sector': 2, 'Q_per_sparse_sector': 1, 'Q_global_max': 48, 'P_sampling_max': 128, 'Q_weight_with_P': 0.35, 'Q_weight_without_P': 1.0, 'ransac_trials': 64, 'refine_starts_per_shape': 5, 'max_nfev': 70, 'out_fractions': [0.1, 0.2, 0.35], 'default_out_fraction': 0.2, 'P_inlier_min': 0.6, 'P_p90_max_rr': 0.1, 'Q_support_min': 0.8, 'near_score_absolute': 0.0004, 'near_score_relative': 0.1, 'max_UD': 0.2, 'tolerance_sensitivity_warning': 0.1, 'saturation_warning': 0.3, 'ellipse_max_axis_ratio': 3.5}
POLICY = {'version': 'V8.4.4-conflict-Q-nearest-current-candidate-1.0', 'Q_anchor_fraction': 0.5, 'Q_total_weight_with_P': 0.35, 'Q_total_weight_zero_P': 1.0, 'anchor_residual': 'same_native_pixel_circle_or_ellipse_residual_as_P', 'diagnostic_only_reasons': ['P_inlier_fraction_below_0.60', 'legacy_arc_coverage_below_0.20', 'weak_evidence_direction_coverage_below_0.50', 'Q_interval_support_below_0.80'], 'unchanged_actual_residual_checks': ['P_p90_residual_too_large', 'legacy_normalized_residual_above_0.08'], 'unchanged_conflict_check': 'P_dominant_serious_Q_conflict', 'unchanged_identifiability': True, 'no_Q_behavior': 'original_current_V2_unchanged', 'original_optimizer_source_sha256': '3ee624682168e5790bf77444c43a09f528dbf3a6fa5b5e11c226c9d9a3c84155', 'adapted_optimizer_source_sha256': '9224356feec8c316ef89782ddcaee0ce9ba03b05b91c8e2e2ea57c17c476f94e'}

def optimise(record,data,plan):
 shape=record['shape'];P=data['P'];Q=[data['Q'][i] for i in plan['score_qi']];rr=data['rr']
 def unpack(z):return np.array([z[0],z[1],z[2],z[2],0.]) if shape=='circle' else z
 def objective(z):
  p=unpack(z);parts=[]
  if len(P):
   r=numerical.residual_true(p,P,shape)/rr;parts.append(np.sign(r)*np.sqrt(huber_loss(r)/len(P)))
  if Q:
   r=numerical.interval_residual(p,Q)/rr;ar=anchor_residual(p,Q,shape)/rr;alpha=POLICY['Q_anchor_fraction'];parts.append(np.sqrt(huber_loss(r)*plan['mass']*plan['score_qw']*(1-alpha)));parts.append(np.sign(ar)*np.sqrt(huber_loss(ar)*plan['mass']*plan['score_qw']*alpha))
  return np.concatenate(parts) if parts else np.zeros(1)
 lo=np.r_[data['center']-1.25*data['extent'],[np.log(max(.15*rr,.08*data['extent']))]*2,-20.];hi=np.r_[data['center']+1.25*data['extent'],[np.log(max(3*rr,1.6*data['extent']))]*2,20.]
 p=record['p'];ix=[0,1,2] if shape=='circle' else list(range(5))
 try:
  result=least_squares(objective,np.clip(p[ix],lo[ix]+1e-7,hi[ix]-1e-7),bounds=(lo[ix],hi[ix]),max_nfev=CONFIG['max_nfev'])
  return dict(record,p=unpack(result.x),origin=record['origin']+'_refined',converged=bool(result.success),nfev=result.nfev)
 except (ValueError,np.linalg.LinAlgError):return dict(record,converged=False)


# Bind relocated dependencies after defining functions (including override hooks).
from . import soft_geometry as numerical
from .geometry_competition import huber_loss as huber_loss
from .q_evidence_policy import anchor_residual as anchor_residual
