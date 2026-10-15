"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.spatial._ckdtree import cKDTree as cKDTree
import math as math
import numpy as np
DISTANCE_POLICY = {'version': 'V8.4.4-conflict-Q-nearest-current-candidate-1.0', 'trigger': 'both_current_A_B_have_no_accepted_shape_and_Q_available', 'ranking': 'arc_normalized_mean_Euclidean_distance_to_original_Q_anchors_px', 'quarter_ellipse_samples': 257, 'golden_iterations': 32, 'tie_epsilon_px': 1e-09, 'near_distance_margin_min_px': 1.0, 'near_distance_relative': 0.1, 'profiles_eligible_for_prediction': False, 'normal_successes_unchanged': True, 'conflict_recovery_never_called_strictly_stable': True}
VERSION84 = 'V8.4.4-conflict-Q-nearest-current-candidate-1.0'

def choose_distance_body(results):
    recovery=[r for r in results if r.get('Q_distance_recovery_applied') and r['chosen'] is not None]
    if recovery:return min(recovery,key=lambda r:(r['selection_distance_px'],r['chosen'].shape!='circle'))
    return OLD_CHOOSE_BODY(results)


def audit_distance_variants(row,results,baseline):
    namespace=OLD_VARIANTS_AUDIT.__globals__;old=namespace['choose_body'];namespace['choose_body']=choose_distance_body
    try:a=OLD_VARIANTS_AUDIT(row,results,baseline)
    finally:namespace['choose_body']=old
    r=choose_distance_body(results['default']);recovered=r.get('Q_distance_recovery_applied',False) and r['chosen'] is not None
    a.update(Q_distance_recovery=bool(recovered),Q_distance_px=r.get('selection_distance_px',math.nan),original_conflict_reasons=';'.join(r.get('selected_original_rejection_reasons',[])))
    if recovered:
        a.update(stability_review_pass=False,stability_flags=(a['stability_flags']+';' if a['stability_flags'] else '')+'residual_conflict_Q_distance_recovery',geometry_state='fit_unstable',evidence_kind='occlusion_recovery_with_residual_conflict')
    return a


def recover_conflicted_candidates(frame,results):
    # Evidence-based branch, never group/type/GT dependent. Leave every existing
    # accepted frame untouched; use only this run's unaccepted proposal pool.
    if any(r['chosen'] is not None for r in results) or not any(r['data']['Q'] for r in results):
        for r in results:r['Q_distance_recovery_applied']=False
        return results
    for r in results:
        r['pre_recovery_reason']=r['reason'];r['pre_recovery_UD']=r['UD'];r['Q_distance_recovery_applied']=True
        all_bounded=[]
        for rec in r['records']:
            if not numerical.bounded(rec['p'],r['data']):continue
            rec['metrics'].update(distance_metrics(rec,r['data']));all_bounded.append(rec)
        eligible=[rec for rec in all_bounded if not rec['origin'].startswith('identifiability_')]
        if not eligible:r['reason']='no_current_bounded_Q_recovery_candidate';continue
        best=min(rec['metrics']['Q_distance_mean_px'] for rec in eligible)
        tied=[rec for rec in eligible if rec['metrics']['Q_distance_mean_px']<=best+DISTANCE_POLICY['tie_epsilon_px']]
        selected=min(tied,key=lambda rec:(rec['shape']!='circle',rec['metrics']['P_loss'],rec['metrics']['axis_ratio']))
        r['selected_original_rejection_reasons']=list(selected['reasons'])
        # Keep every old quality metric and residual veto for audit. This branch
        # outputs an explicitly inferred recovery candidate, not a clean fit.
        for rec in all_bounded:
            rec['original_rejection_reasons']=list(rec['reasons']);rec['reasons']=[]
            rec['metrics']['recovery_conflict_diagnostics']=rec['original_rejection_reasons']
        candidate=gc.to_candidate(frame,r['data'],selected,'B')
        candidate.fit.update(recovery_only=True,original_quality_reasons=r['selected_original_rejection_reasons'])
        candidate.provenance.update(selection_version=VERSION84,selection_basis=DISTANCE_POLICY['ranking'],evidence_status='occlusion_recovery_with_residual_conflict')
        margin=max(DISTANCE_POLICY['near_distance_margin_min_px'],DISTANCE_POLICY['near_distance_relative']*best)
        near=[rec for rec in all_bounded if rec['metrics']['Q_distance_mean_px']<=best+margin]
        diam=np.array([rec['metrics']['diameter_px'] for rec in near])
        r.update(chosen=candidate,near=near,UD=float(np.ptp(diam)/np.median(diam)),range_px=[float(diam.min()),float(diam.max())],reason='',quality_pass_count=len(eligible),branch='Q_distance_conflict_recovery',selection_distance_px=selected['metrics']['Q_distance_mean_px'],selection_record_origin=selected['origin'],distance_policy=DISTANCE_POLICY)
    return results


def distance_metrics(record,data):
    q=data['Q'];dist=euclidean_Q_distance(record['p'],[z['q'] for z in q],record['shape']);w=numerical.weights(q)
    return dict(Q_distance_mean_px=float(w@dist),Q_distance_median_px=float(np.median(dist)),Q_distance_p90_px=float(np.quantile(dist,.9)),Q_distance_all=dist)


def euclidean_Q_distance(p,points,shape):
    points=np.asarray(points,float).reshape(-1,2)
    if not len(points):return np.empty(0)
    a,b=np.exp(p[2:4]);co,si=np.cos(p[4]),np.sin(p[4]);rotation=np.array([[co,-si],[si,co]])
    local=np.abs((points-p[:2])@rotation)
    if shape=='circle':return np.abs(np.linalg.norm(local,axis=1)-a)
    grid=np.linspace(0.,np.pi/2,DISTANCE_POLICY['quarter_ellipse_samples']);curve=np.c_[a*np.cos(grid),b*np.sin(grid)]
    nearest=cKDTree(curve).query(local)[1];step=grid[1]-grid[0]
    lo=np.maximum(0.,grid[nearest]-2*step);hi=np.minimum(np.pi/2,grid[nearest]+2*step);ratio=(np.sqrt(5)-1)/2
    def squared(t):return (a*np.cos(t)-local[:,0])**2+(b*np.sin(t)-local[:,1])**2
    for _ in range(DISTANCE_POLICY['golden_iterations']):
        left=hi-ratio*(hi-lo);right=lo+ratio*(hi-lo);first=squared(left)<squared(right)
        hi=np.where(first,right,hi);lo=np.where(first,lo,left)
    closest=squared((lo+hi)/2)
    closest=np.minimum(closest,np.minimum(squared(np.zeros(len(local))),squared(np.full(len(local),np.pi/2))))
    return np.sqrt(np.maximum(closest,0.))


# Bind relocated dependencies after defining functions (including override hooks).
from . import geometry_competition as gc
from . import soft_geometry as numerical
from .pq_review import choose_body as OLD_CHOOSE_BODY
from .v8_context import variants_audit as OLD_VARIANTS_AUDIT
