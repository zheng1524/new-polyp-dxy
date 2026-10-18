"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.optimize._lsq.least_squares import least_squares as least_squares
import cv2 as cv2
import math as math
import numpy as np
VERSION = 'V7-phase2-directional-PQ-1.2'
CONFIG = {'sectors': 16, 'P_sufficient_per_sector': 4, 'Q_per_empty_sector': 2, 'Q_per_sparse_sector': 1, 'Q_global_max': 48, 'P_sampling_max': 128, 'Q_weight_with_P': 0.35, 'Q_weight_without_P': 1.0, 'ransac_trials': 64, 'refine_starts_per_shape': 5, 'max_nfev': 70, 'out_fractions': [0.1, 0.2, 0.35], 'default_out_fraction': 0.2, 'P_inlier_min': 0.6, 'P_p90_max_rr': 0.1, 'Q_support_min': 0.8, 'near_score_absolute': 0.0004, 'near_score_relative': 0.1, 'max_UD': 0.2, 'tolerance_sensitivity_warning': 0.1, 'saturation_warning': 0.3, 'ellipse_max_axis_ratio': 3.5}

def make_data(frame,base,raw,frozen,body,fraction):
 ca=next(c for c in base.candidates if c['name']==body)
 visible=set(map(tuple,np.rint(ca['trusted']).astype(int)))
 records=frozen['records'];ps=[r for r in records if r['class_']=='P' and tuple(r['q']) in visible]
 Q=[];arcids={}
 for aid,arc in enumerate(frozen['Q_arcs']):
  for idx in arc['indices']:arcids[(arc['contour_id'],idx)]=aid
 for r in records:
  if r['class_']!='Q':continue
  Q.append(dict(q=np.array(r['q'],float),n=np.array(r['n'],float),lo=-r['delta_in_px'],hi=r['delta_out_px'][str(fraction)],arc_id=arcids[(r['contour_id'],r['contour_index'])],global_index=r['global_index'],contour_id=r['contour_id'],contour_index=r['contour_index']))
 P=np.array([r['q'] for r in ps],float).reshape(-1,2)
 if ca['mask'].any():center=ca['geom_center_local']+ca['crop_offset'];extent=ca['extent'];center_source='unchanged_V2_body_descriptor'
 else:
  pts=np.array([r['q'] for r in records if r['class_'] in ['P','Q']],float).reshape(-1,2)
  center=(pts.min(0)+pts.max(0))/2 if len(pts) else raw['center']+np.array(raw['crop'][:2]);extent=float(np.linalg.norm(np.ptp(pts,axis=0))) if len(pts) else raw['rr'];center_source='frozen_original_contour_PQ_bbox_center_empty_V2_mask'
 return dict(name=body,P=P,Q=Q,rr=raw['rr'],center=np.array(center),extent=max(extent,1.),center_source=center_source,P_global_indices=[r['global_index'] for r in ps],V2_P_input_count=len(ca['trusted']),original_P_count=sum(r['class_']=='P' for r in records),original_Q_count=len(Q),fraction=fraction)


def uniform(ids,n):
 ids=np.asarray(ids,int)
 return ids if len(ids)<=n else ids[np.linspace(0,len(ids)-1,n).astype(int)]


def to_candidate(frame,data,r,method):
 p=r['p'];a,b=np.exp(p[2:4]);t=p[4];basis=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])@np.diag([a,b]);m={k:v for k,v in r['metrics'].items() if not isinstance(v,np.ndarray)};pr=np.abs(numerical.residual_true(p,data['P'],r['shape']));qr=numerical.interval_residual(p,data['Q']);m.update(center_distance_norm=float(np.linalg.norm(p[:2]-data['center'])/data['extent']),fit_minor_radius=float(min(a,b)),fit_major_radius=float(max(a,b)),min_radius_bound=max(.15*data['rr'],.08*data['extent']),max_radius_bound=max(3*data['rr'],1.6*data['extent']),arc_coverage=m['evidence_direction_coverage'],norm_residual=float(np.median(pr)/np.sqrt(a*b)) if len(pr) else float(numerical.weights(data['Q'])@qr/data['rr']),norm_residual_all=float(np.sqrt(m['score'])),inlier_ratio=m['P_inlier_ratio'] if len(pr) else m['Q_support_arc_weighted'],n_points=len(data['P']));prov=dict(provenance(frame,'V7_phase2_'+method),implementation_version=VERSION,length_unit='px_diameter',Q_are_intervals=True,measurement_evidence='occlusion_inference' if not len(data['P']) else 'observed_P_plus_occlusion_Q',candidate_origin=r['origin']);return GeometryCandidate(data['name'],r['shape'],p[:2],basis,float(2*max(a,b)),m,dict(valid=not r['reasons'],reason=';'.join(r['reasons'])),prov)


def metrics(p,data,shape,plan):
 P=data['P'];Q=data['Q'];rr=data['rr'];pr=np.abs(numerical.residual_true(p,P,shape));qr=numerical.interval_residual(p,Q);qw=numerical.weights(Q);sel=plan['score_qi'];t,has=q_intersections(p,Q)
 P_loss=float(huber_loss(pr/rr).mean()) if len(pr) else 0.
 Q_loss=float(plan['score_qw']@huber_loss(qr[sel]/rr)) if len(sel) else 0.
 support=(qr<=max(1.,.01*rr))&has
 saturation=np.array([has[i] and abs(t[i]-q['hi'])<=.1*(q['hi']-q['lo']) for i,q in enumerate(Q)])
 arcs=np.array([q['arc_id'] for q in Q]);coherent=[]
 for aid in np.unique(arcs):
  ii=np.flatnonzero((arcs==aid)&has)
  if len(ii)>1:coherent.append(float(np.std([(t[i]-Q[i]['lo'])/(Q[i]['hi']-Q[i]['lo']) for i in ii])))
 axis=np.exp(p[2:4]);diam=2*max(axis);points=np.vstack([P[pr<=max(2,.04*rr)],np.array([q['q'] for q,ok in zip(Q,support) if ok]).reshape(-1,2)])
 coverage=len(np.unique(sector(points,p[:2])))/16 if len(points) else 0.
 return dict(score=P_loss+plan['mass']*Q_loss,P_loss=P_loss,Q_loss=Q_loss,P_median_px=float(np.median(pr)) if len(pr) else math.nan,P_p90_px=float(np.quantile(pr,.9)) if len(pr) else math.nan,P_inlier_ratio=float((pr<=max(2,.04*rr)).mean()) if len(pr) else math.nan,Q_support_arc_weighted=float(qw@support) if len(Q) else math.nan,Q_support_count_fraction=float(support.mean()) if len(Q) else math.nan,Q_saturation_arc_weighted=float(qw@saturation) if len(Q) else math.nan,Q_saturation_count_fraction=float(saturation.mean()) if len(Q) else math.nan,Q_arc_shift_fraction_std_mean=float(np.mean(coherent)) if coherent else math.nan,evidence_direction_coverage=coverage,diameter_px=float(diam),axis_ratio=float(max(axis)/min(axis)),legacy_norm_residual=float(np.median(pr)/np.sqrt(axis.prod())) if len(pr) else float(qw@qr/rr),normal_intersection_t=t,Q_support_mask=support,Q_saturation_mask=saturation)


def optimise(record,data,plan):
 shape=record['shape'];P=data['P'];Q=[data['Q'][i] for i in plan['score_qi']];rr=data['rr']
 def unpack(z):return np.array([z[0],z[1],z[2],z[2],0.]) if shape=='circle' else z
 def objective(z):
  p=unpack(z);parts=[]
  if len(P):
   r=numerical.residual_true(p,P,shape)/rr;parts.append(np.sign(r)*np.sqrt(huber_loss(r)/len(P)))
  if Q:
   r=numerical.interval_residual(p,Q)/rr;parts.append(np.sqrt(huber_loss(r)*plan['mass']*plan['score_qw']))
  return np.concatenate(parts) if parts else np.zeros(1)
 lo=np.r_[data['center']-1.25*data['extent'],[np.log(max(.15*rr,.08*data['extent']))]*2,-20.];hi=np.r_[data['center']+1.25*data['extent'],[np.log(max(3*rr,1.6*data['extent']))]*2,20.]
 p=record['p'];ix=[0,1,2] if shape=='circle' else list(range(5))
 try:
  result=least_squares(objective,np.clip(p[ix],lo[ix]+1e-7,hi[ix]-1e-7),bounds=(lo[ix],hi[ix]),max_nfev=CONFIG['max_nfev'])
  return dict(record,p=unpack(result.x),origin=record['origin']+'_refined',converged=bool(result.success),nfev=result.nfev)
 except (ValueError,np.linalg.LinAlgError):return dict(record,converged=False)


def huber_loss(r,delta=.04):
 a=np.abs(r);return np.where(a<=delta,r*r,2*delta*a-delta*delta)


def sector(points,center):
 points=np.asarray(points).reshape(-1,2)
 return (np.mod(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]),2*np.pi)/(2*np.pi)*CONFIG['sectors']).astype(int)


def q_intersections(p,Q):
 if not Q:return np.empty(0),np.empty(0,bool)
 q=np.array([z['q'] for z in Q]);n=np.array([z['n'] for z in Q]);lo=np.array([z['lo'] for z in Q]);hi=np.array([z['hi'] for z in Q]);co,si=np.cos(p[4]),np.sin(p[4]);R=np.array([[co,-si],[si,co]]);M=R@np.diag(np.exp(-2*p[2:4]))@R.T;d=q-p[:2];aa=np.einsum('ij,jk,ik->i',n,M,n);bb=2*np.einsum('ij,jk,ik->i',d,M,n);cc=np.einsum('ij,jk,ik->i',d,M,d)-1;disc=bb*bb-4*aa*cc
 roots=np.c_[(-bb-np.sqrt(np.maximum(disc,0)))/(2*aa),(-bb+np.sqrt(np.maximum(disc,0)))/(2*aa)]
 outside=np.maximum(lo[:,None]-roots,0)+np.maximum(roots-hi[:,None],0)
 ix=np.argmin(outside+1e-10*np.abs(roots),axis=1);return roots[np.arange(len(Q)),ix],disc>=0


def quality_reasons(p,data,m):
 why=[]
 if m['legacy_norm_residual']>.08:why.append('legacy_normalized_residual_above_0.08')
 if m['evidence_direction_coverage']<.20:why.append('legacy_arc_coverage_below_0.20')
 if not numerical.bounded(p,data):why.append('V2_center_radius_or_axis_bounds')
 if len(data['P']):
  if m['P_inlier_ratio']<.60:why.append('P_inlier_fraction_below_0.60')
  if m['P_p90_px']>max(3.,.10*data['rr']):why.append('P_p90_residual_too_large')
 if m['Q_support_arc_weighted']<.80:why.append('Q_interval_support_below_0.80')
 if len(data['P'])<25 and m['evidence_direction_coverage']<.50:why.append('weak_evidence_direction_coverage_below_0.50')
 return why


def fit(frame,data,old,method,seed,q_cap=None):
 plan=sample_plan(data,method,q_cap)
 if not data['Q']:
  valid=[c for c in old if exact_v2.gate(c).accepted]
  ci=next((c for c in valid if c.shape=='circle'),None);el=next((c for c in valid if c.shape=='ellipse'),None)
  chosen=el if el and (ci is None or el.quality['axis_ratio']>=1.15 and el.fit['norm_residual_all']<=.75*ci.fit['norm_residual_all']) else ci
  return dict(method=method,data=data,plan=plan,records=[],chosen=chosen,near=[],UD=math.nan,range_px=[],reason='' if chosen else 'original_V2_reject',branch='no_Q_exact_V2',common_count=len(old),extra_count=0,quality_pass_count=len(valid),actual_sampled_P_indices=[],actual_sampled_Q_indices=[])
 pool=shared_pool(data,old);common=len(pool);extra=extra_pool(data,plan,method,seed);pool+=extra
 for r in pool:r['metrics']=metrics(r['p'],data,r['shape'],plan)
 starts=[]
 for shape in ['circle','ellipse']:
  seen=set()
  for r in sorted([r for r in pool if r['shape']==shape],key=lambda r:r['metrics']['score']):
   key=round(r['metrics']['diameter_px']/(.02*data['rr']))
   if key in seen:continue
   seen.add(key);starts.append(r)
   if len(seen)>=CONFIG['refine_starts_per_shape']:break
 for r in starts if method!='C' else []:
  refined=optimise(r,data,plan)
  if numerical.bounded(refined['p'],data):refined['metrics']=metrics(refined['p'],data,refined['shape'],plan);pool.append(refined)
 # Explicit radius/centre profiles retained unrefined: reveal interval plateaus.
 for shape in ['circle','ellipse']:
  options=[r for r in pool if r['shape']==shape]
  if not options:continue
  best=min(options,key=lambda r:r['metrics']['score'])
  for factor in [.85,.90,.95,1.,1.05,1.10,1.15]:
   for axis in ['both'] if shape=='circle' else ['both','major']:
    p=best['p'].copy()
    if axis=='both':p[2:4]+=np.log(factor)
    else:p[2+int(p[3]>p[2])]+=np.log(factor)
    if numerical.bounded(p,data):pool.append(dict(p=p,shape=shape,origin='identifiability_radius_profile_'+axis,trace=[],metrics=metrics(p,data,shape,plan)))
  for ax in [0,1]:
   for delta in [-.05,.05]:
    p=best['p'].copy();p[ax]+=delta*data['rr']
    if numerical.bounded(p,data):pool.append(dict(p=p,shape=shape,origin='identifiability_center_profile',trace=[],metrics=metrics(p,data,shape,plan)))
 for r in pool:r['reasons']=quality_reasons(r['p'],data,r['metrics'])
 passed=[r for r in pool if not r['reasons']];near=[];chosen=None;UD=math.nan;dr=[];why='no_candidate_passed_evidence_checks'
 if passed:
  eligible=[r for r in passed if not r['origin'].startswith('identifiability_')]
  if not eligible:return dict(method=method,data=data,plan=plan,records=pool,chosen=None,near=[],UD=math.nan,range_px=[],reason='only_diagnostic_profile_passed',branch='PQ_intervals',common_count=common,extra_count=len(extra),quality_pass_count=len(passed),actual_sampled_P_indices=sorted({t['P_index'] for r in pool for t in r['trace'] if t['kind']=='P'}),actual_sampled_Q_indices=sorted({t['Q_index'] for r in pool for t in r['trace'] if t['kind'].startswith('Q')}))
  best=min(r['metrics']['score'] for r in eligible);near=[r for r in passed if r['metrics']['score']<=best*(1.10)+.0004];diam=np.array([r['metrics']['diameter_px'] for r in near]);dr=[float(diam.min()),float(diam.max())];UD=float(np.ptp(diam)/np.median(diam))
  distinct=len({tuple(np.round(r['p'],3)) for r in near if not r['origin'].startswith('identifiability_')})
  why='diameter_not_identifiable' if UD>.20 else 'insufficient_distinct_solutions' if len(data['P'])<25 and distinct<3 else ''
  def representative(shape):
   options=[r for r in eligible if r['shape']==shape]
   if not options:return None
   loss=min(r['metrics']['score'] for r in options);ties=[r for r in options if r['metrics']['score']<=loss+1e-12];median=np.median([r['metrics']['diameter_px'] for r in ties]);return min(ties,key=lambda r:abs(r['metrics']['diameter_px']-median))
  ci=representative('circle');el=representative('ellipse')
  selected=el if el and (ci is None or el['metrics']['axis_ratio']>=1.15 and el['metrics']['score']<.75*ci['metrics']['score']-1e-12) else ci
  if selected and not why:chosen=to_candidate(frame,data,selected,method)
 sampledP={t['P_index'] for r in pool for t in r['trace'] if t['kind']=='P'};sampledQ={t['Q_index'] for r in pool for t in r['trace'] if t['kind'].startswith('Q')}
 return dict(method=method,data=data,plan=plan,records=pool,chosen=chosen,near=near,UD=UD,range_px=dr,reason=why,branch='P_main_Q_complement' if len(data['P']) else 'zero_P_Q_intervals',common_count=common,extra_count=len(extra),quality_pass_count=len(passed),actual_sampled_P_indices=sorted(sampledP),actual_sampled_Q_indices=sorted(sampledQ))


def shared_pool(data,old):
 pool=[]
 for c in old:add(pool,numerical.vector(c),c.shape,'shared_current_V2_seed',data)
 # With P present common seeds are P-only V2: no dense-direction Q leaks into B.
 if len(data['P']):return pool
 # With P=0 all methods share sparse coherent Q initialisations.
 ids=uniform(np.arange(len(data['Q'])),32)
 for f in [0.,.25,.5,.75,1.]:
  pts=np.vstack([data['P'][uniform(np.arange(len(data['P'])),128)],np.array([data['Q'][i]['q']+(data['Q'][i]['lo']+f*(data['Q'][i]['hi']-data['Q'][i]['lo']))*data['Q'][i]['n'] for i in ids]).reshape(-1,2)])
  if len(pts)>=3:add(pool,algebraic_circle(pts),'circle','shared_coherent_fraction_'+str(f),data,[dict(kind='Q_coherent_initialisation',Q_index=int(i),global_index=data['Q'][i]['global_index'],t=data['Q'][i]['lo']+f*(data['Q'][i]['hi']-data['Q'][i]['lo']),lo=data['Q'][i]['lo'],hi=data['Q'][i]['hi']) for i in ids])
  if len(pts)>=5:
   try:add(pool,numerical.RAWG3.ellipse_from_cv(pts),'ellipse','shared_coherent_fraction_'+str(f),data,[dict(kind='Q_coherent_initialisation',Q_index=int(i),global_index=data['Q'][i]['global_index'],t=data['Q'][i]['lo']+f*(data['Q'][i]['hi']-data['Q'][i]['lo']),lo=data['Q'][i]['lo'],hi=data['Q'][i]['hi']) for i in ids])
   except cv2.error:pass
 return pool


def sample_plan(data,method,q_cap=None):
 P=data['P'];Q=data['Q'];pc=sector(P,data['center']);qc=sector([q['q'] for q in Q],data['center']);pn=np.bincount(pc,minlength=16);qn=np.bincount(qc,minlength=16)
 pi=uniform(np.arange(len(P)),CONFIG['P_sampling_max'])
 if method=='B':
  qi=[]
  for s in range(16):
   quota=2 if pn[s]==0 else 1 if pn[s]<4 else 0
   qi.extend(uniform(np.flatnonzero(qc==s),quota) if quota else [])
  qi=np.array(qi,int)
 else:qi=uniform(np.arange(len(Q)),q_cap or CONFIG['Q_global_max'])
 if q_cap and len(qi)>q_cap:qi=uniform(qi,q_cap)
 # Density does not change effective mass: arcs normalised separately.
 qw=numerical.weights([Q[i] for i in qi]);mass=CONFIG['Q_weight_with_P'] if len(P) else CONFIG['Q_weight_without_P']
 if not len(qi):mass=0.
 score_qi=np.arange(len(Q)) if method in ['A','C'] else qi
 score_qw=numerical.weights([Q[i] for i in score_qi])
 return dict(pi=pi,qi=qi,qw=qw,score_qi=score_qi,score_qw=score_qw,mass=mass,P_sector_counts=pn,Q_sector_counts=qn,selected_Q_sector_counts=np.bincount(qc[qi],minlength=16),P_sector_coverage=float((pn>0).mean()),Q_sector_coverage=float((qn>0).mean()),selected_Q_sector_coverage=float(len(np.unique(qc[qi]))/16),P_effective_weight=1. if len(P) else 0.,Q_effective_weight=mass)


def extra_pool(data,plan,method,seed):
 rng=np.random.default_rng(seed);pool=[];P=data['P'];Q=data['Q'];pi=plan['pi'];qi=plan['qi']
 if method in ['A','B']:
  if len(P) and len(qi):
   for f in [0.,.25,.5,.75,1.]:
    pts=np.vstack([P[pi],np.array([Q[i]['q']+(Q[i]['lo']+f*(Q[i]['hi']-Q[i]['lo']))*Q[i]['n'] for i in qi])]);trace=[dict(kind='P',P_index=int(i)) for i in pi]+[dict(kind='Q_coherent_initialisation',Q_index=int(i),global_index=Q[i]['global_index'],t=Q[i]['lo']+f*(Q[i]['hi']-Q[i]['lo']),lo=Q[i]['lo'],hi=Q[i]['hi']) for i in qi]
    add(pool,algebraic_circle(pts),'circle',method+'_selected_Q_coherent_profile',data,trace)
    if len(pts)>=5:
     try:add(pool,numerical.RAWG3.ellipse_from_cv(pts),'ellipse',method+'_selected_Q_coherent_profile',data,trace)
     except cv2.error:pass
  for shape,n in [('circle',3),('ellipse',5)]:
   for trial in range(CONFIG['ransac_trials']):
    if len(P)>=n:npick=n-1 if len(qi) and (method=='A' or trial%3==0) else n
    else:npick=min(len(P),n-1 if len(qi) else n)
    nq=n-npick
    if len(pi)<npick or len(qi)<nq:continue
    picked=rng.choice(pi,npick,replace=False) if npick else [];selected=rng.choice(qi,nq,replace=False) if nq else []
    pts=list(P[picked]) if npick else [];trace=[dict(kind='P',P_index=int(i)) for i in picked]
    for j in selected:
     q=Q[j];t=float(rng.uniform(q['lo'],q['hi']));pts.append(q['q']+t*q['n']);trace.append(dict(kind='Q_interval_sample',Q_index=int(j),global_index=q['global_index'],t=t,lo=q['lo'],hi=q['hi']))
    try:p=numerical.circle3(np.array(pts)) if shape=='circle' else numerical.RAWG3.ellipse_from_cv(np.array(pts));add(pool,p,shape,method+'_minimal_RANSAC',data,trace)
    except (cv2.error,np.linalg.LinAlgError,ValueError):pass
 else:
  # Entire original Q arcs share one displacement fraction; no independent Q jitter.
  aids=np.unique([Q[i]['arc_id'] for i in qi])
  for trial in range(19):
   fractions={int(a):(trial/4 if trial<5 else float(rng.uniform(0,1))) for a in aids};pts=list(P[pi]);trace=[dict(kind='P',P_index=int(i)) for i in pi]
   for j in qi:
    q=Q[j];f=fractions[q['arc_id']];t=q['lo']+f*(q['hi']-q['lo']);pts.append(q['q']+t*q['n']);trace.append(dict(kind='Q_arc_shared_displacement',Q_index=int(j),global_index=q['global_index'],arc_id=q['arc_id'],fraction=f,t=t,lo=q['lo'],hi=q['hi']))
   pts=np.array(pts).reshape(-1,2)
   c=legacy.v2.fit_circle(pts,data['center'],data['extent'],data['rr'],str(seed)+'_'+str(trial));e,loc=old_geometry.traced_call(legacy.v2.fit_ellipse,pts,data['center'],data['extent'],data['rr'])
   if 'radius_px' in c:add(pool,np.array([c['cx'],c['cy'],np.log(c['radius_px']),np.log(c['radius_px']),0.]),'circle','C_arc_shift_V2_circle',data,trace)
   if 'major_diameter_px' in e:
    theta=np.deg2rad(e['angle_deg']+(90 if loc['d1']<loc['d2'] else 0));add(pool,np.array([e['cx'],e['cy'],np.log(e['major_diameter_px']/2),np.log(e['minor_diameter_px']/2),theta]),'ellipse','C_arc_shift_V2_ellipse',data,trace)
 return pool


def add(pool,p,shape,origin,data,trace=None):
 if p is not None and numerical.bounded(p,data):pool.append(dict(p=p,shape=shape,origin=origin,trace=trace or []))


def algebraic_circle(pts):
 if len(pts)<3:return None
 s=np.linalg.lstsq(np.c_[2*pts,np.ones(len(pts))],(pts*pts).sum(1),rcond=None)[0];r2=s[2]+s[:2]@s[:2]
 return np.array([*s[:2],.5*np.log(r2),.5*np.log(r2),0.]) if r2>0 else None


# Bind relocated dependencies after defining functions (including override hooks).
from . import components as legacy
from . import exact_v2 as exact_v2
from . import geometry_component as old_geometry
from . import soft_geometry as numerical
from .contracts import GeometryCandidate as GeometryCandidate
from .contracts import provenance as provenance
