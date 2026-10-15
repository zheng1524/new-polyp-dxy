"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import json as json
import math as math
import numpy as np


def variants_audit(row, results, baseline):
    """Exactly the frozen V7 review protocol; no threshold adjustment."""
    selected={v:choose_body(rr) for v,rr in results.items()}
    default=selected['default']; a=audit_row(row,default)
    chosen={f:selected[k] for f,k in [(0.1,'tolerance_10'),(.2,'default'),(.35,'tolerance_35')]}
    lens=[r['chosen'].length_native_px for r in chosen.values() if r['chosen'] is not None]
    a['tolerance_accept_count']=len(lens)
    a['tolerance_L_range_fraction']=float(np.ptp(lens)/np.median(lens)) if lens else math.nan
    for f,r in chosen.items():
        a['L_'+str(f)+'_px']=r['chosen'].length_native_px if r['chosen'] else math.nan
        a['accepted_'+str(f)]=r['chosen'] is not None
    vc={v:selected[v] for v in ['seed_plus_1','Q_pool_cap_24']}
    near=default['near']+[r for item in vc.values() for r in item['near']]
    diam=[r['metrics']['diameter_px'] for r in near]
    a['cross_seed_pool_UD']=float(np.ptp(diam)/np.median(diam)) if diam else math.nan
    a['cross_seed_pool_distinct_solutions']=len({canonical_shape_key(r) for r in near if not r['origin'].startswith('identifiability_')})
    vv=[default['chosen'].length_native_px] if default['chosen'] else []
    for v,r in vc.items():
        a[v+'_accepted']=r['chosen'] is not None
        a[v+'_L_px']=r['chosen'].length_native_px if r['chosen'] else math.nan
        if r['chosen']:vv.append(r['chosen'].length_native_px)
    a['seed_pool_L_range_fraction']=float(np.ptp(vv)/np.median(vv)) if vv else math.nan
    ab=[r['chosen'].length_native_px for r in results['default'] if r['chosen'] is not None]
    a['AB_selected_length_spread']=float(np.ptp(ab)/np.median(ab)) if ab else math.nan
    flags=[]
    if not a['accepted']:flags.append(default['reason'] or 'no_accepted_geometry')
    if default['branch']!='no_Q_exact_V2':
        if a['tolerance_accept_count']!=3:flags.append('not_accepted_at_all_three_tolerances')
        if a['tolerance_L_range_fraction']>.10:flags.append('tolerance_length_sensitivity_above_10pct')
        if any(r['chosen'] is None for r in vc.values()):flags.append('seed_or_Q_pool_variant_rejected')
        if a['cross_seed_pool_UD']>.20:flags.append('cross_seed_pool_diameter_not_identifiable')
        if a['cross_seed_pool_distinct_solutions']<3:flags.append('insufficient_near_optimal_search_not_proven_unique')
        if a['seed_pool_L_range_fraction']>.10:flags.append('seed_or_Q_pool_length_sensitivity_above_10pct')
        if a['Q_saturation']>.30:flags.append('tolerance_boundary_saturation_warning')
        if a['AB_selected_length_spread']>.20:flags.append('AB_candidate_disagreement_above_20pct')
    a['stability_review_pass']=not flags; a['stability_flags']=';'.join(flags)
    generated=sum(r['common_count'] if not r['records'] else sum(not q['origin'].startswith('identifiability_') for q in r['records']) for r in results['default'])
    reasonable=sum(r['quality_pass_count'] if not r['records'] else sum(not q['reasons'] and not q['origin'].startswith('identifiability_') for q in r['records']) for r in results['default'])
    a['candidate_generated']=generated>0; a['candidate_generated_count_AB']=generated
    a['reasonable_fit']=reasonable>0; a['reasonable_candidate_count_AB']=reasonable
    a['geometry_state']='fit_stable' if a['stability_review_pass'] else 'fit_unstable' if reasonable else 'fit_failure'
    a['evidence_kind']='unchanged_current_V2' if default['branch']=='no_Q_exact_V2' else 'occlusion_inference' if len(default['data']['P'])==0 else 'observed_P_plus_soft_Q'
    a['V2_baseline_accepted']=baseline is not None
    a['V2_baseline_L_px']=baseline.length_native_px if baseline else math.nan
    a['V2_baseline_shape']=baseline.shape if baseline else ''
    a['relative_length_change_vs_V2']=a['L_px']/a['V2_baseline_L_px']-1 if a['accepted'] and baseline else math.nan
    a['default_body_reasons']=json.dumps({r['data']['name']:r['reason'] for r in results['default']})
    a['reasonable_L_px']=a['L_px']
    if not a['accepted']:
        # Diagnostic only: never fills L_px or changes frozen selection/Gate.
        rr=[q for r in results['default'] for q in r['records'] if not q['reasons'] and not q['origin'].startswith('identifiability_')]
        if rr:a['reasonable_L_px']=min(rr,key=lambda q:q['metrics']['score'])['metrics']['diameter_px']
    if baseline and default['chosen']:
        bc=default['chosen']; _,u,s,_=ellipse_axes(bc); _,v,t,_=ellipse_axes(baseline)
        a['B_axis_ratio']=float(s[0]/s[1]);a['V2_axis_ratio']=float(t[0]/t[1])
        a['axis_angle_change_deg']=float(np.degrees(np.arccos(np.clip(abs(u[:,0]@v[:,0]),0,1)))) if bc.shape=='ellipse' and baseline.shape=='ellipse' else math.nan
    return a


# Bind relocated dependencies after defining functions (including override hooks).
from .contracts import ellipse_axes as ellipse_axes
from .deliver import audit_row as audit_row
from .geometry_identity import canonical_shape_key as canonical_shape_key
from .pq_review import choose_body as choose_body
