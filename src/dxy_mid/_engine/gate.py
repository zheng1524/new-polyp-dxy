"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import math as math
import numpy as np
METHODS = ('u2_strict', 'legacy', 'g3_original', 'arc_030', 'arc_050', 'stability_040', 'support_040')

def choose(candidates,decisions,method='u2_strict'):
 accepted=[c for c,d in zip(candidates,decisions) if d.accepted]
 if not accepted:return None
 winners=[]
 for name in sorted(set(c.name for c in accepted)):
  opts=[c for c in accepted if c.name==name];ci=next((c for c in opts if c.shape=='circle'),None);el=next((c for c in opts if c.shape=='ellipse'),None)
  c=el if el and (ci is None or (el.quality['axis_ratio']>=1.15 and el.quality['norm_residual_all']<=.75*ci.quality['norm_residual_all'])) else ci
  q=c.quality;score=q['arc_coverage']*q['inlier_ratio']/(1+10*q['norm_residual_all'])
  if method not in ['legacy','g3_original']:score*=q['C']*q['P']/(1+q['sector_instability']/(1+10*q['norm_residual_all']))
  if c.shape=='ellipse':score*=.9
  if c.provenance['method']=='g3_native':score=1/(1+q['g3_score'])
  winners.append((score,c.name.startswith('A'),c))
 return max(winners,key=lambda x:x[:2])[2]


def run(candidate,method='u2_strict'):
 if method not in METHODS:raise ValueError(method)
 q=candidate.quality;reasons=[];checks={}
 # Legacy fit-level constraints stay explicit and cannot be bypassed unknowingly.
 if candidate.provenance['method']=='g3_native':
  checks.update(g3_diameter_spread=dict(value=q['g3_spread'],threshold=.20,passed=q['g3_spread']<=.20),g3_visible_coverage=dict(value=q['arc_coverage'],threshold=.12,passed=q['arc_coverage']>=.12))
 else:
  checks.update(legacy_center=dict(value=q['center_distance_norm'],threshold=1.25,passed=q['center_distance_norm']<=1.25),legacy_radius_min=dict(value=q['fit_minor_radius'],threshold=q['min_radius_bound'],passed=q['fit_minor_radius']>=q['min_radius_bound']),legacy_radius_max=dict(value=q['fit_major_radius'],threshold=q['max_radius_bound'],passed=q['fit_major_radius']<=q['max_radius_bound']),legacy_arc=dict(value=q['arc_coverage'],threshold=.20,passed=q['arc_coverage']>=.20),legacy_residual=dict(value=q['norm_residual'],threshold=.08,passed=q['norm_residual']<=.08))
  if candidate.shape=='ellipse':checks['legacy_axis_ratio']=dict(value=q['axis_ratio'],threshold=3.5,passed=q['axis_ratio']<=3.5)
 for k,check in checks.items():
  if not check['passed']:reasons.append(k)
 if method not in ['legacy','g3_original']:
  thresholds=dict(n_points=25,visible_fraction=.20,arc_coverage=.40,norm_residual_all=.10,inlier_ratio=.60,C=.50,P=.50,sector_instability=.25)
  if method=='arc_030':thresholds['arc_coverage']=.30
  if method=='arc_050':thresholds['arc_coverage']=.50
  if method=='stability_040':thresholds['sector_instability']=.40
  if method=='support_040':thresholds.update(C=.40,P=.40)
  for k,t in thresholds.items():
   v=q.get(k,math.nan);passed=np.isfinite(v) and (v<=t if k in ['norm_residual_all','sector_instability'] else v>=t);checks[k]=dict(value=v,threshold=t,passed=bool(passed))
   if not passed:reasons.append(k)
 prov=dict(candidate.provenance,method=method)
 return GateResult(not reasons,reasons,dict(raw=q,checks=checks),prov)


# Bind relocated dependencies after defining functions (including override hooks).
from .contracts import GateResult as GateResult
