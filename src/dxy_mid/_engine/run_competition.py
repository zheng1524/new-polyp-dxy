"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import json as json
import math as math
import numpy as np


def current_baseline(frame,base,raw,row):
 cs,fail=exact_v2.geometry(frame,None,base,stem=row['stem']);ds=[exact_v2.gate(c) for c in cs];chosen=old_gate.choose(cs,ds,'legacy')
 return cs,chosen,fail


def summary(row,res,fraction):
 data=res['data'];plan=res['plan'];c=res['chosen'];rbest=next((r for r in res['records'] if c is not None and r['shape']==c.shape and np.allclose(r['p'][:2],c.center_native) and abs(r['metrics']['diameter_px']-c.length_native_px)<1e-7),None)
 if rbest is None and res['near']:rbest=min(res['near'],key=lambda r:r['metrics']['score'])
 m=rbest['metrics'] if rbest else metrics(numerical.vector(c),data,c.shape,plan) if c else {}
 return dict(stem=row['stem'],group=row['group'],role=row['role'],source_id=row['source_id'],method=res['method'],body=data['name'],out_fraction=fraction,P_original=data['original_P_count'],Q_original=data['original_Q_count'],P_body=len(data['P']),V2_P_input_count=data['V2_P_input_count'],P_sampling_pool=data['V2_P_input_count'] if res['branch']=='no_Q_exact_V2' else len(plan['pi']),Q_sampling_pool=len(plan['qi']),P_actual_unique=math.nan if res['branch']=='no_Q_exact_V2' else len(res['actual_sampled_P_indices']),P_objective_count=len(data['P']),Q_objective_count=len(plan['score_qi']),Q_actual_unique=len(res['actual_sampled_Q_indices']),Q_shared_initial_unique=len({t['Q_index'] for r in res['records'] if r['origin'].startswith('shared_coherent') for t in r['trace'] if t['kind'].startswith('Q')}),Q_minimal_RANSAC_unique=len({t['Q_index'] for r in res['records'] if 'minimal_RANSAC' in r['origin'] for t in r['trace'] if t['kind'].startswith('Q')}),P_effective_weight=plan['P_effective_weight'],Q_effective_weight=plan['Q_effective_weight'],P_sector_coverage=plan['P_sector_coverage'],Q_sector_coverage=plan['Q_sector_coverage'],selected_Q_sector_coverage=plan['selected_Q_sector_coverage'],P_sector_counts=json.dumps(plan['P_sector_counts'].tolist()),Q_sector_counts=json.dumps(plan['Q_sector_counts'].tolist()),Q_selected_sector_counts=json.dumps(plan['selected_Q_sector_counts'].tolist()),center_x=data['center'][0],center_y=data['center'][1],center_source=data['center_source'],candidate_count=len(res['records']) if res['records'] else res['common_count'],common_count=res['common_count'],extra_count=res['extra_count'],quality_pass_count=res['quality_pass_count'],near_count=len(res['near']),near_nonprofile_count=sum(not r['origin'].startswith('identifiability_') for r in res['near']),UD=res['UD'],D_min_px=res['range_px'][0] if res['range_px'] else math.nan,D_max_px=res['range_px'][1] if res['range_px'] else math.nan,accepted=c is not None,shape=c.shape if c else '',L_px=c.length_native_px if c else math.nan,reason=res['reason'],branch=res['branch'],P_median_px=m.get('P_median_px',math.nan),P_p90_px=m.get('P_p90_px',math.nan),P_inlier_ratio=m.get('P_inlier_ratio',math.nan),Q_support=m.get('Q_support_arc_weighted',math.nan),Q_saturation=m.get('Q_saturation_arc_weighted',math.nan),Q_saturation_count=m.get('Q_saturation_count_fraction',math.nan),Q_arc_shift_std=m.get('Q_arc_shift_fraction_std_mean',math.nan),score=m.get('score',math.nan),history_prediction_fallback=0)


# Bind relocated dependencies after defining functions (including override hooks).
from . import exact_v2 as exact_v2
from . import gate as old_gate
from . import soft_geometry as numerical
from .geometry_competition import metrics as metrics
