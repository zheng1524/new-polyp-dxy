"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import json as json
import math as math
import numpy as np


def audit_row(row,res):
 res=normalise_audit(res);a=summary(row,res,res['data']['fraction']);records=res['records'];generated=[r for r in records if not r['origin'].startswith('identifiability_')]
 cm=res['chosen'];probe=next((r for r in records if cm is not None and r['shape']==cm.shape and np.allclose(r['p'][:2],cm.center_native) and abs(r['metrics']['diameter_px']-cm.length_native_px)<1e-7),None)
 loss_metrics=probe['metrics'] if probe else min(generated,key=lambda r:r['metrics']['score'])['metrics'] if generated else metrics(numerical.vector(cm),res['data'],cm.shape,res['plan']) if cm else {}
 a.update(P_loss=loss_metrics.get('P_loss',math.nan),Q_loss=loss_metrics.get('Q_loss',math.nan),weighted_Q_loss=res['plan']['mass']*loss_metrics.get('Q_loss',math.nan),Q_loss_share=res['plan']['mass']*loss_metrics.get('Q_loss',0)/loss_metrics['score'] if loss_metrics.get('score',0)>0 else math.nan)
 a.update(generated_count=len(generated) if records else res['common_count'],profile_count=len(records)-len(generated),generated_quality_pass_count=sum(not r['reasons'] for r in generated) if records else res['quality_pass_count'],near_distinct_nonprofile=len({canonical_shape_key(r) for r in res['near'] if not r['origin'].startswith('identifiability_')}))
 if not a['accepted'] and generated:
  diagnostic=min(generated,key=lambda r:r['metrics']['score']);m=diagnostic['metrics'];a.update(diagnostic_best_L_px=m['diameter_px'],diagnostic_best_shape=diagnostic['shape'],diagnostic_reasons=json.dumps(diagnostic['reasons']))
  if not np.isfinite(a['P_median_px']):a['P_median_px']=m['P_median_px'];a['P_p90_px']=m['P_p90_px'];a['Q_support']=m['Q_support_arc_weighted'];a['Q_saturation']=m['Q_saturation_arc_weighted']
 a['B_Q_in_P_sufficient_sectors']=int(res['plan']['selected_Q_sector_counts'][res['plan']['P_sector_counts']>=4].sum()) if res['method']=='B' else math.nan
 return a


# Bind relocated dependencies after defining functions (including override hooks).
from . import soft_geometry as numerical
from .geometry_competition import metrics as metrics
from .geometry_identity import canonical_shape_key as canonical_shape_key
from .pq_geometry_component import normalise_audit as normalise_audit
from .run_competition import summary as summary
