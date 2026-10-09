#!/usr/bin/env python3
"""G3: deterministic occlusion-aware amodal ellipse completion experiment.

This is deliberately an experiment-local wrapper around the frozen G2 inputs.
It never writes S7/G0/G2 artefacts.  `blind` does not read reference diameters;
`evaluate` is the only stage that joins them afterwards.
"""
from __future__ import annotations

import argparse, hashlib, html, json, math, shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .layout import EDGE_OUT, FORMAL_MAINLINE as S7, G3_OUT as OUT, MULTIARC_OUT
G0='G0_frozen_S7_V2';G2='G2_adjacency_multiarc_ellipse';G3='G3_occlusion_aware_amodal'

# Fixed before GT evaluation. Distances are normalized by the existing ring
# equivalent radius (rr); no R5/R10/class-specific branch exists.
MIN_TRUSTED_POINTS=20
ARC_PAIR_SEPARATION=0.25
ENDPOINT_BAND_DISTANCE=0.08
EDGE_HUBER=0.04
LAMBDA_TANGENT=0.20
LAMBDA_OCCLUSION=0.15
LAMBDA_BACKGROUND=0.35
BOUNDARY_SAMPLES=72
MULTISTART_SCALE=(0.88,1.00,1.15,1.30)
MULTISTART_CENTER_SHIFT=(-0.14,0.0,0.14)
MAX_STARTS=12
NEAR_OPTIMAL_SCORE_FRACTION=0.08
AMBIGUOUS_DIAMETER_SPREAD=0.20
MIN_VISIBLE_ANGULAR_COVERAGE=0.12
SOURCE=None

def load_source():
 global SOURCE
 if SOURCE is None:
  from . import multiarc
  SOURCE=multiarc
 return SOURCE

def finite(x):
 try:return math.isfinite(float(x))
 except:return False
def truth(x):return str(x).lower()=='true' if isinstance(x,str) else bool(x)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(x,rel):
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False)
def selected():return load_source().selected()
def sample_bilinear(img,points):
 h,w=img.shape[:2];x=np.clip(points[:,0],0,w-1);y=np.clip(points[:,1],0,h-1)
 x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1);wx=x-x0;wy=y-y0
 return (1-wx)*(1-wy)*img[y0,x0]+wx*(1-wy)*img[y0,x1]+(1-wx)*wy*img[y1,x0]+wx*wy*img[y1,x1]
def params_from_row(r):
 if not finite(r.get('ellipse_cx')):return None
 a=float(r.ellipse_major_diameter_px)/2;b=float(r.ellipse_minor_diameter_px)/2
 if a<=1 or b<=1:return None
 return np.array([float(r.ellipse_cx),float(r.ellipse_cy),math.log(a),math.log(b),math.radians(float(r.ellipse_angle_deg))],float)
def canonical(p):return load_source().canonical(p)
def ellipse_residual(points,p):return load_source().ellipse_residual(points,p)
def uniform(points,n):return load_source().uniform_arc_sample(points,n)
def ellipse_from_cv(points):return load_source().ellipse_from_cv(points)
def ellipse_points(p,n=BOUNDARY_SAMPLES):
 cx,cy,la,lb,ang=p;a,b=np.exp(la),np.exp(lb);t=np.linspace(0,2*np.pi,n,endpoint=False);co,si=np.cos(ang),np.sin(ang)
 return np.c_[cx+a*np.cos(t)*co-b*np.sin(t)*si,cy+a*np.cos(t)*si+b*np.sin(t)*co]
def projected_tangent(point,p):
 cx,cy,la,lb,ang=p;a,b=np.exp(la),np.exp(lb);co,si=np.cos(ang),np.sin(ang);q=point-np.array([cx,cy]);x=q[0]*co+q[1]*si;y=-q[0]*si+q[1]*co
 t=math.atan2(y/max(b,1e-9),x/max(a,1e-9));v=np.array([-a*math.sin(t)*co-b*math.cos(t)*si,-a*math.sin(t)*si+b*math.cos(t)*co]);return v/max(np.linalg.norm(v),1e-9)
def contour_tangent(a,at_start):
 p=a['points'];k=min(6,len(p)-1)
 v=p[k]-p[0] if at_start else p[-1]-p[-1-k];return v/max(np.linalg.norm(v),1e-9)
def band_endpoints(arcs,band,rr):
 """Only endpoints of trusted original arcs near unknown U receive tangent terms."""
 dt=cv2.distanceTransform((~band).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);out=[]
 for a in arcs:
  if len(a['points'])<3:continue
  for at_start,pt in ((True,a['points'][0]),(False,a['points'][-1])):
   xi=int(np.clip(round(pt[0]),0,band.shape[1]-1));yi=int(np.clip(round(pt[1]),0,band.shape[0]-1))
   if dt[yi,xi]<=ENDPOINT_BAND_DISTANCE*rr:
    out.append({'point':pt,'tangent':contour_tangent(a,at_start),'arc_id':a['arc_id']})
 return out
def visible_angular_coverage(trusted,p):
 if not len(trusted):return 0.
 e=canonical(p);q=trusted-np.array([e['cx'],e['cy']]);t=math.radians(e['angle_deg']);co,si=math.cos(t),math.sin(t);x=q[:,0]*co+q[:,1]*si;y=-q[:,0]*si+q[:,1]*co
 a=np.mod(np.arctan2(y/max(e['minor_radius_px'],1e-9),x/max(e['major_radius_px'],1e-9)),2*np.pi);return float(np.histogram(a,bins=36,range=(0,2*np.pi))[0].astype(bool).mean())
def support_arcs(arcs,p,rr):
 out=[];tol=max(2.,.05*rr)
 for a in arcs:
  frac=float((np.abs(ellipse_residual(a['points'],p))<=tol).mean()) if len(a['points']) else 0.
  if frac>=.35:out.append(a)
 return out
def separated(arcs,rr):
 return any(np.hypot(a['centroid_x']-b['centroid_x'],a['centroid_y']-b['centroid_y'])>=ARC_PAIR_SEPARATION*rr for i,a in enumerate(arcs) for b in arcs[i+1:])
def objective_residuals(p,trusted,endpoints,bg_dt,rr):
 """Normalized residual vector for Huber/IRLS refinement.

`bg_dt` is non-zero only deep inside unequivocal visible background. Ring U is
zero, so a candidate may cross U but is softly discouraged from expanding into B.
"""
 if not np.all(np.isfinite(p)):return np.ones(max(8,len(trusted)))*1e3
 a,b=np.exp(p[2]),np.exp(p[3])
 if min(a,b)<1 or max(a,b)>8*rr:return np.ones(max(8,len(trusted)))*1e3
 edge=ellipse_residual(trusted,p)/rr
 r=[edge]
 if endpoints:
  tan=[]
  for e in endpoints:
   dot=abs(float(np.dot(projected_tangent(e['point'],p),e['tangent'])));tan.append(math.sqrt(LAMBDA_TANGENT)*math.sqrt(max(0.,1-dot*dot)))
  r.append(np.asarray(tan))
 # Soft background: 72 analytic boundary samples. It does not insist that
 # every non-U pixel is mask, because actual polyps are not perfect ellipses.
 bp=ellipse_points(p);bpen=np.minimum(sample_bilinear(bg_dt,bp)/rr,1.5)
 r.append(math.sqrt(LAMBDA_BACKGROUND)*bpen)
 return np.concatenate(r)
def occlusion_penalty(p,endpoints,band,rr):
 if not endpoints:return 0.
 # An endpoint can be explained only if its predicted ellipse has a nearby
 # portion inside U; no artificial contour point is created.
 bp=ellipse_points(p,144);vals=[]
 for e in endpoints:
  d=np.linalg.norm(bp-e['point'],axis=1);k=np.argsort(d)[:5];vals.append(float((~band[np.clip(np.round(bp[k,1]).astype(int),0,band.shape[0]-1),np.clip(np.round(bp[k,0]).astype(int),0,band.shape[1]-1)]).mean()))
 return float(np.mean(vals))
def make_starts(g2p, trusted, arcs, center, rr):
 starts=[]
 def add(p):
  if p is None or not np.all(np.isfinite(p)):return
  if not any(np.linalg.norm(p[:4]-q[:4])<1e-3 for q in starts):starts.append(p.copy())
 # Full trusted contour and the frozen G2 result seed complementary modes.
 add(g2p);add(ellipse_from_cv(trusted))
 eligible=[a for a in arcs if a.get('eligible_seed_arc',False)]
 for i,a in enumerate(eligible):
  for b in eligible[i+1:]:
   if np.hypot(a['centroid_x']-b['centroid_x'],a['centroid_y']-b['centroid_y'])>=ARC_PAIR_SEPARATION*rr:
    add(ellipse_from_cv(np.vstack([uniform(a['points'],40),uniform(b['points'],40)])))
    if len(starts)>=4:break
  if len(starts)>=4:break
 # Cover distinct modes before adding local variants: the frozen G2 seed, an
 # all-trusted AMS seed, and deterministic separated-arc-pair seeds.  The
 # prior version accidentally exhausted its cap on variants of the first seed.
 expanded=[]
 variants=[(MULTISTART_SCALE[1],MULTISTART_CENTER_SHIFT[1]),(MULTISTART_SCALE[0],MULTISTART_CENTER_SHIFT[1]),(MULTISTART_SCALE[2],MULTISTART_CENTER_SHIFT[1]),(MULTISTART_SCALE[3],MULTISTART_CENTER_SHIFT[1]),(MULTISTART_SCALE[2],MULTISTART_CENTER_SHIFT[0]),(MULTISTART_SCALE[2],MULTISTART_CENTER_SHIFT[2])]
 for p in starts[:3]:
  co,si=np.cos(p[4]),np.sin(p[4])
  for scale,shift in variants:
   q=p.copy();q[2]+=math.log(scale);q[3]+=math.log(scale);q[0]+=shift*rr*co;q[1]+=shift*rr*si;expanded.append(q)
 # Evenly interleave seeds rather than letting the first initialisation own
 # every start slot.  Constants remain the declared fixed values above.
 ordered=[]
 for j in range(6):
  for i in range(min(3,len(starts))):
   k=i*6+j
   if k<len(expanded):ordered.append(expanded[k])
 return ordered[:MAX_STARTS]
def g3_fitter(arcs,trusted,uncertain,band,polyp,rr,g2p):
 if len(trusted)<MIN_TRUSTED_POINTS:return {'valid':False,'reason':'too_few_trusted_original_edges','candidate_count':0}
 # Either two spatially separated trusted arcs, or a single arc with two
 # independently observed contacts to U. This is the minimum observability rule.
 endpoints=band_endpoints(arcs,band,rr)
 eligible=[a for a in arcs if a.get('eligible_seed_arc',False)]
 observable=separated(eligible,rr) or len(endpoints)>=2
 if not observable:return {'valid':False,'reason':'insufficient_multiarc_or_occlusion_endpoints','candidate_count':0}
 visible_background=(~polyp)&(~band)
 bg_dt=cv2.distanceTransform(visible_background.astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
 candidates=[]
 for start in make_starts(g2p,trusted,arcs,np.zeros(2),rr):
  try:res=least_squares(lambda p:objective_residuals(p,trusted,endpoints,bg_dt,rr),start,loss='huber',f_scale=EDGE_HUBER,max_nfev=220)
  except Exception:continue
  if not res.success:continue
  p=res.x;e=canonical(p)
  if min(e['major_radius_px'],e['minor_radius_px'])<=1 or max(e['major_radius_px'],e['minor_radius_px'])>8*rr:continue
  edge=float(np.median(np.abs(ellipse_residual(trusted,p)))/rr)
  tangent=float(np.mean([math.sqrt(max(0.,1-abs(float(np.dot(projected_tangent(x['point'],p),x['tangent'])))**2)) for x in endpoints])) if endpoints else 0.
  occ=occlusion_penalty(p,endpoints,band,rr)
  bp=ellipse_points(p);bg=float(np.mean(np.minimum(sample_bilinear(bg_dt,bp)/rr,1.5)))
  cov=visible_angular_coverage(trusted,p);sup=support_arcs(arcs,p,rr)
  # This explicit score is L_edge + lambda terms, all normalized quantities.
  score=edge+LAMBDA_TANGENT*tangent+LAMBDA_OCCLUSION*occ+LAMBDA_BACKGROUND*bg
  candidates.append({'p':p,'e':e,'score':score,'edge_loss':edge,'tangent_loss':tangent,'occlusion_loss':occ,'background_loss':bg,'coverage':cov,'support':sup})
 if not candidates:return {'valid':False,'reason':'no_optimisation_candidate','candidate_count':0}
 candidates.sort(key=lambda x:x['score']);best=candidates[0];near=[x for x in candidates if x['score']<=best['score']*(1+NEAR_OPTIMAL_SCORE_FRACTION)+1e-9]
 diam=np.array([x['e']['major_diameter_px'] for x in near]);spread=(diam.max()-diam.min())/max(np.median(diam),1e-9)
 if spread>AMBIGUOUS_DIAMETER_SPREAD:return {'valid':False,'reason':'ambiguous_occlusion_completion','candidate_count':len(candidates),'near_optimal_count':len(near),'diameter_spread':float(spread),'best_score':best['score']}
 if best['coverage']<MIN_VISIBLE_ANGULAR_COVERAGE:return {'valid':False,'reason':'insufficient_visible_angular_coverage','candidate_count':len(candidates),'near_optimal_count':len(near),'diameter_spread':float(spread),'best_score':best['score']}
 e=best['e'];sup=best['support'];ufrac=float((np.abs(ellipse_residual(uncertain,best['p']))<=max(2.,.05*rr)).mean()) if len(uncertain) else math.nan
 s=load_source();C,P=s.cp_support(polyp,band,e)
 return {'valid':True,'reason':'ok','shape':'ellipse','L_px':e['major_diameter_px'],'residual':best['edge_loss']*rr,'angular_coverage':best['coverage'],'edge_inlier_fraction':float((np.abs(ellipse_residual(trusted,best['p']))<=max(2.,.05*rr)).mean()),'supported_arc_count':len(sup),'supported_arc_ids':[a['arc_id'] for a in sup],'uncertain_inlier_fraction':ufrac,'C':C,'P':P,'params':e,'candidate_count':len(candidates),'near_optimal_count':len(near),'diameter_spread':float(spread),'best_score':best['score'],'edge_loss':best['edge_loss'],'tangent_loss':best['tangent_loss'],'occlusion_loss':best['occlusion_loss'],'background_loss':best['background_loss'],'occlusion_endpoint_count':len(endpoints)}
def rec(r,z):
 ringmm=10. if '_R10_' in r.group else 5.;ringpx=float(r.ring_px);valid=bool(z.get('valid',False)) and finite(z.get('L_px')) and finite(ringpx)
 q={'geometry':G3,'stem':r.stem,'group':r.group,'capture':r.capture,'bin':int(r.segment_id),'frame_index':r.frame_index,'ring_px':ringpx,'ring_mm':ringmm,'valid':valid,'L_px':z.get('L_px',math.nan),'Dxy_i_mm':ringmm*z.get('L_px',math.nan)/ringpx if valid else math.nan,'fit_reason':z.get('reason',''),'fit_shape':z.get('shape','none'),'residual':z.get('residual',math.nan),'angular_coverage':z.get('angular_coverage',math.nan),'edge_inlier_fraction':z.get('edge_inlier_fraction',math.nan),'supported_arc_count':z.get('supported_arc_count',math.nan),'supported_arc_length_fraction':math.nan,'supported_arc_ids':'|'.join(z.get('supported_arc_ids',[])),'uncertain_inlier_fraction':z.get('uncertain_inlier_fraction',math.nan),'C_diag':z.get('C',math.nan),'P_diag':z.get('P',math.nan),'trusted_arc_count':z.get('trusted_arc_count',math.nan),'trusted_point_count':z.get('trusted_point_count',math.nan),'uncertain_point_count':z.get('uncertain_point_count',math.nan),'cut_point_count':z.get('cut_point_count',math.nan),'small_fit_failure':str(z.get('reason','')).startswith(('too_few','insufficient','no_','ambiguous'))}
 for k in ('candidate_count','near_optimal_count','diameter_spread','best_score','edge_loss','tangent_loss','occlusion_loss','background_loss','occlusion_endpoint_count'):q[k]=z.get(k,math.nan)
 par=z.get('params',{})
 for k in ('cx','cy','major_diameter_px','minor_diameter_px','angle_deg','axis_ratio'):q['ellipse_'+k]=par.get(k,math.nan)
 return q
def frame_g3(r,ep,g2row):
 s=load_source();edge=s.load_edge();d=edge.prepare_original(r);contours=edge.external_contours(d['polyp']);pts=np.concatenate(contours) if contours else np.empty((0,2),np.float32)
 p=ep.set_index('point_index').reindex(np.arange(len(pts)));assert len(p)==len(pts)
 state=p.adjacency_class.to_numpy(object);near=p.ring_adjacent.to_numpy(bool);arcs=s.make_arcs(contours,state,near,d['center'],float(d['rr']));trusted=pts[state=='high_confidence'];uncertain=pts[state=='uncertain'];cut=pts[state=='ring_cut']
 g2p=params_from_row(g2row) if g2row is not None else None
 z=g3_fitter(arcs,trusted,uncertain,d['band'].astype(bool),d['polyp'].astype(bool),float(d['rr']),g2p)
 z.update({'trusted_arc_count':len(arcs),'trusted_point_count':len(trusted),'uncertain_point_count':len(uncertain),'cut_point_count':len(cut)})
 return rec(r,z),{'d':d,'pts':pts,'states':state,'arcs':arcs,'trusted':trusted,'uncertain':uncertain,'g3':z}
def source_rows_blind():
 f=pd.read_csv(MULTIARC_OUT/'tables/frame_geometry_blind.csv');return f[f.geometry.isin([G0,G2])].copy()
def synth_occlusion(sel,points):
 """GT-free sanity check: hide known raw-contour portions with three fixed bands.

The raw complete contour is the reference only for this synthetic exercise;
it is not a clinical reference and never enters video fitting/evaluation.
"""
 s=load_source();edge=s.load_edge();out=[];available=[]
 for r in sel.to_dict('records'):
  ep=points[points.stem.eq(r['stem'])];
  if ep.empty or float((ep.adjacency_class=='ring_cut').mean())>.03:continue
  available.append(r)
 for r in sorted(available,key=lambda x:x['stem'])[:20]:
  rr=type('R',(),r)();d=edge.prepare_original(rr);cs=edge.external_contours(d['polyp']);pts=np.concatenate(cs) if cs else np.empty((0,2));p0=ellipse_from_cv(pts)
  if p0 is None:continue
  ref=canonical(p0)['major_diameter_px'];h,w=d['polyp'].shape;Y,X=np.mgrid[:h,:w];cx,cy=d['center'];
  for ang in (0.,math.pi/4,math.pi/2):
   normal=np.array([math.cos(ang),math.sin(ang)]);dist=np.abs((X-cx)*normal[0]+(Y-cy)*normal[1]);U=dist<=.10*float(d['rr']);keep=np.array([not U[int(np.clip(round(y),0,h-1)),int(np.clip(round(x),0,w-1))] for x,y in pts]);states=np.where(keep,'high_confidence','ring_cut');near=np.array([not x for x in keep]);arcs=s.make_arcs(cs,states,near,d['center'],float(d['rr']));trusted=pts[keep];z2=s.multiarc_fitter(arcs,trusted,np.empty((0,2)),d['center'],float(d['rr']),d['polyp'].astype(bool),U);z3=g3_fitter(arcs,trusted,np.empty((0,2)),U,d['polyp'].astype(bool),float(d['rr']),None)
   for name,z in [(G2,z2),(G3,z3)]:
    val=z.get('L_px',math.nan);out.append({'stem':r['stem'],'occlusion_angle_deg':round(math.degrees(ang)),'geometry':name,'valid':bool(z.get('valid',False)),'full_contour_major_px':ref,'estimated_major_px':val,'relative_major_error_pct':100*(val-ref)/ref if finite(val) else math.nan,'fit_reason':z.get('reason','')})
 return pd.DataFrame(out)
def blind():
 for d in ('inputs','tables','visualizations','scripts'):(OUT/d).mkdir(parents=True,exist_ok=True)
 protocol={'baseline':'S7-FMD39 and frozen G2 inputs','GT_used_before_evaluation':False,'g3':'amodal contour completion: trusted original edges + ring U tangent/occlusion/background soft constraints','fixed_parameters':{k:globals()[k] for k in ('MIN_TRUSTED_POINTS','ARC_PAIR_SEPARATION','ENDPOINT_BAND_DISTANCE','EDGE_HUBER','LAMBDA_TANGENT','LAMBDA_OCCLUSION','LAMBDA_BACKGROUND','BOUNDARY_SAMPLES','MULTISTART_SCALE','MULTISTART_CENTER_SHIFT','MAX_STARTS','NEAR_OPTIMAL_SCORE_FRACTION','AMBIGUOUS_DIAMETER_SPREAD','MIN_VISIBLE_ANGULAR_COVERAGE')},'statistics':'frozen per-frame Dxy; >=4 valid frames capture median; frozen formal group fusion','original_edge_only':True,'ring_edges_not_fit_points':True}
 (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 sel=selected();points=pd.read_csv(EDGE_OUT/'tables/original_contour_point_ring_adjacency.csv');src=source_rows_blind();g2rows=src[src.geometry.eq(G2)].set_index('stem');rows=[]
 for i,r in enumerate(sel.to_dict('records'),1):
  rr=type('R',(),r)();g2row=g2rows.loc[r['stem']] if r['stem'] in g2rows.index else None;q,_=frame_g3(rr,points[points.stem.eq(r['stem'])],g2row);rows.append(q)
  if i%25==0:print(f'G3 {i}/395',flush=True)
 g3=pd.DataFrame(rows);frames=pd.concat([src,g3],ignore_index=True,sort=False);caps=load_source().capture_rows(frames);groups=load_source().fuse(caps)
 write(frames,'tables/frame_geometry_blind.csv');write(caps,'tables/capture_predictions_blind.csv');write(groups,'tables/group_predictions_blind.csv');syn=synth_occlusion(sel,points);write(syn,'tables/synthetic_occlusion_recovery_blind.csv')
 files=['inputs/frozen_protocol.json','tables/frame_geometry_blind.csv','tables/capture_predictions_blind.csv','tables/group_predictions_blind.csv','tables/synthetic_occlusion_recovery_blind.csv'];(OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/f)}  {f}\n' for f in files));print('sealed')
def metrics(x):return load_source().metrics(x)
def evaluate():
 for line in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():h,r=line.split('  ',1);assert sha(OUT/r)==h
 refs=pd.read_csv(S7/'tables/group_errors_descending.csv')[['group','reference_mm']];frames=pd.read_csv(OUT/'tables/frame_geometry_blind.csv').merge(refs,on='group',how='left');frames['signed_error_pct']=100*(frames.Dxy_i_mm-frames.reference_mm)/frames.reference_mm;frames['abs_error_pct']=frames.signed_error_pct.abs();caps=pd.read_csv(OUT/'tables/capture_predictions_blind.csv').merge(refs,on='group',how='left');caps['capture_abs_error_pct']=100*(caps.capture_prediction_mm-caps.reference_mm).abs()/caps.reference_mm;groups=pd.read_csv(OUT/'tables/group_predictions_blind.csv').merge(refs,on='group',how='left');groups['signed_error_pct']=100*(groups.prediction_mm-groups.reference_mm)/groups.reference_mm;groups['abs_error_pct']=groups.signed_error_pct.abs()
 write(frames,'tables/frame_geometry_evaluated.csv');write(caps,'tables/capture_predictions_evaluated.csv');write(groups,'tables/group_predictions_evaluated.csv');m=pd.DataFrame([{'geometry':geo,**metrics(x)} for geo,x in groups.groupby('geometry')]);write(m,'tables/metrics_summary.csv')
 # Paired geometry audit: a positive value means the amodal candidate enlarged
 # the G2 long axis. This is descriptive post-evaluation, never a rule input.
 wide=frames[frames.geometry.isin([G2,G3])].pivot(index=['stem','group','capture','bin'],columns='geometry',values=['L_px','Dxy_i_mm','valid','diameter_spread','candidate_count']).reset_index()
 wide.columns=['_'.join(str(v) for v in x if v) for x in wide.columns]
 wide['both_valid']=wide[f'valid_{G2}'].map(truth)&wide[f'valid_{G3}'].map(truth)
 wide['G3_minus_G2_L_px']=wide[f'L_px_{G3}']-wide[f'L_px_{G2}']
 wide['G3_minus_G2_L_relative_pct']=100*wide['G3_minus_G2_L_px']/wide[f'L_px_{G2}']
 write(wide,'tables/g3_vs_g2_long_axis.csv')
 reasons=frames[frames.geometry.eq(G3)].groupby('fit_reason').size().rename('frames').reset_index();write(reasons,'tables/g3_fit_reason_counts.csv')
 fs=[]
 for geo,x in frames.groupby('geometry'):
  a=x.abs_error_pct;cs=caps[caps.geometry.eq(geo)];fs.append({'geometry':geo,'valid_frames':int(a.notna().sum()),'frame_median_abs_error_pct':a.median(),'catastrophic_gt20':int((a>20).sum()),'catastrophic_gt50':int((a>50).sum()),'eligible_captures':int(cs.eligible.sum()),'fallback_captures':int((~cs.eligible).sum()),'median_capture_MAD_mm':cs.capture_MAD_mm.median(),'median_capture_range_mm':cs.capture_range_mm.median()})
 write(fs,'tables/stability_summary.csv');excel(groups,caps,frames,m);visualize(frames,groups,caps);reports(frames,groups,caps,m,fs)
 print(m.to_string(index=False))
def excel(groups,caps,frames,m):
 wide=groups.pivot(index=['group','reference_mm'],columns='geometry',values=['prediction_mm','signed_error_pct','abs_error_pct']).reset_index();wide.columns=['_'.join(str(y) for y in x if y) for x in wide.columns];wide=wide.sort_values(f'abs_error_pct_{G3}',ascending=False);dwide=frames.pivot(index=['geometry','group','capture'],columns='bin',values='Dxy_i_mm').rename(columns=lambda x:f'Dxy_i_bin{int(x):02d}_mm').reset_index();detail=caps.merge(dwide,on=['geometry','group','capture'],how='left')
 with pd.ExcelWriter(OUT/'G3_results.xlsx',engine='openpyxl') as w:wide.to_excel(w,sheet_name='Group Ranking',index=False,na_rep='NA');detail.to_excel(w,sheet_name='Capture Detail',index=False,na_rep='NA');frames.to_excel(w,sheet_name='Frame Detail',index=False,na_rep='NA');m.to_excel(w,sheet_name='Summary',index=False)
 wb=load_workbook(OUT/'G3_results.xlsx');fill=PatternFill('solid',fgColor='1F4E78')
 for ws in wb.worksheets:
  ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
  for c in ws[1]:c.font=Font(color='FFFFFF',bold=True);c.fill=fill
  for i,col in enumerate(ws.columns,1):ws.column_dimensions[get_column_letter(i)].width=min(38,max(11,max(len(str(v.value or '')) for v in list(col)[:100])+2))
 wb.save(OUT/'G3_results.xlsx')
def draw_e(ax,row,color,label):
 if not finite(row.ellipse_cx):return
 ax.add_patch(Ellipse((row.ellipse_cx,row.ellipse_cy),row.ellipse_major_diameter_px,row.ellipse_minor_diameter_px,angle=row.ellipse_angle_deg,fill=False,edgecolor=color,lw=2.4,label=label))
def visualization_frame(stem,frames,label):
 s=load_source();edge=s.load_edge();sel=selected();r=sel[sel.stem.eq(stem)].iloc[0];d=edge.prepare_original(type('R',(),r.to_dict())());x0,y0,x1,y1=d['crop'];im=cv2.cvtColor(cv2.imread(str(r.image_path)),cv2.COLOR_BGR2RGB)[y0:y1,x0:x1];z=frames[frames.stem.eq(stem)].set_index('geometry');pts=pd.read_csv(EDGE_OUT/'tables/original_contour_point_ring_adjacency.csv');p=pts[pts.stem.eq(stem)]
 fig,axs=plt.subplots(1,3,figsize=(17,5));overlay=im.copy()
 for mask,color,alpha in ((d['polyp'],(30,210,70),.16),(d['ring'],(255,205,0),.48),(d['band'],(185,40,220),.16)):overlay[mask.astype(bool)]=(overlay[mask.astype(bool)]*(1-alpha)+np.asarray(color)*alpha).astype(np.uint8)
 axs[0].imshow(overlay);axs[0].set_title('image + polyp / ring U');axs[1].imshow(im);axs[1].set_title('trusted / uncertain / cut original edges');axs[2].imshow(im);axs[2].set_title('frozen G2 vs G3')
 for c,col in [('high_confidence','lime'),('uncertain','orange'),('ring_cut','red')]:q=p[p.adjacency_class.eq(c)];axs[1].scatter(q.x,q.y,s=1,c=col,label=c)
 axs[1].legend(fontsize=7,loc='lower right')
 for geo,col in [(G2,'cyan'),(G3,'lime')]:
  row=z.loc[geo];draw_e(axs[2],row,col,geo);val='NA' if not finite(row.Dxy_i_mm) else f'{row.Dxy_i_mm:.2f} mm'
  extra=''
  if geo==G3 and finite(row.get('candidate_count',math.nan)):
   extra=f"; candidates={int(row.candidate_count)}, near={int(row.near_optimal_count) if finite(row.near_optimal_count) else 'NA'}, spread={row.diameter_spread:.2%}" if finite(row.diameter_spread) else f"; candidates={int(row.candidate_count)}"
  axs[2].text(7,20+18*(geo==G3),f'{geo}: {val}; {row.fit_reason}{extra}',color=col,fontsize=8,bbox={'facecolor':'black','alpha':.6})
 axs[2].legend(fontsize=7,loc='lower right')
 for ax in axs:ax.axis('off')
 fig.suptitle(label+' — '+stem);fig.tight_layout();fig.savefig(OUT/'visualizations'/f'{label}_{stem}.png',dpi=145);plt.close(fig)
def visualize(frames,groups,caps):
 q=frames.pivot(index='stem',columns='geometry',values=['Dxy_i_mm','abs_error_pct','valid']).reset_index();q.columns=['_'.join(str(y) for y in x if y) for x in q.columns];pick=[]
 # Evaluation-only choices for inspection, never feedback.
 both=q[q[f'valid_{G2}'].map(truth)&q[f'valid_{G3}'].map(truth)].copy();both['change']=both[f'Dxy_i_mm_{G3}']-both[f'Dxy_i_mm_{G2}'];both['improve']=both[f'abs_error_pct_{G2}']-both[f'abs_error_pct_{G3}'];both['worse']=-both['improve']
 for label,sub,col,asc in [('g2_small_to_g3_larger',both,'change',False),('g3_frame_improvement',both,'improve',False),('g3_frame_worsening',both,'worse',False)]:
  if len(sub):pick.append((label,sub.sort_values(col,ascending=asc).stem.iloc[0]))
 inv=frames[(frames.geometry.eq(G3))&(~frames.valid.map(truth))]
 for reason,label in [('ambiguous_occlusion_completion','candidate_uncertainty'),('insufficient_multiarc_or_occlusion_endpoints','severe_occlusion_insufficient_edges')]:
  x=inv[inv.fit_reason.eq(reason)]
  if len(x):pick.append((label,x.stem.iloc[0]))
 # Even when the fixed ambiguity gate does not reject a frame, show the most
 # dispersed near-optimal candidate set for human review.
 uncertainty=frames[(frames.geometry.eq(G3))&frames.diameter_spread.map(finite)].sort_values('diameter_spread',ascending=False)
 if len(uncertainty):pick.append(('largest_candidate_uncertainty',uncertainty.stem.iloc[0]))
 # one group-level largest deterioration with a valid G3 frame
 gw=groups.pivot(index='group',columns='geometry',values='abs_error_pct');gw['d']=gw[G3]-gw[G2]
 for group in gw.sort_values('d',ascending=False).index:
  x=frames[(frames.group.eq(group))&(frames.geometry.eq(G3))&frames.valid.map(truth)]
  if len(x):pick.append(('largest_group_g3_degradation',x.stem.iloc[0]));break
 seen=set()
 for label,stem in pick:
  if stem not in seen:visualization_frame(stem,frames,label);seen.add(stem)
def reports(frames,groups,caps,m,fs):
 mm=m.set_index('geometry');ss=pd.DataFrame(fs).set_index('geometry');g2=groups[groups.geometry.eq(G2)].set_index('group');g3=groups[groups.geometry.eq(G3)].set_index('group');delta=(g3.abs_error_pct-g2.abs_error_pct);changed=int((delta!=0).sum());imp=int((delta<0).sum());worse=int((delta>0).sum());syn=pd.read_csv(OUT/'tables/synthetic_occlusion_recovery_blind.csv');synsum=syn.groupby('geometry').agg(valid=('valid','sum'),n=('valid','size'),median_abs_relative_major_error_pct=('relative_major_error_pct',lambda x:np.nanmedian(np.abs(x)))).reset_index();write(synsum,'tables/synthetic_occlusion_summary.csv')
 la=pd.read_csv(OUT/'tables/g3_vs_g2_long_axis.csv');both=la[la.both_valid.map(truth)];longer=int((both.G3_minus_G2_L_px>0).sum());shorter=int((both.G3_minus_G2_L_px<0).sum());ambiguous=int((frames[(frames.geometry.eq(G3))].fit_reason=='ambiguous_occlusion_completion').sum())
 human=f'''# G3 遮挡感知椭圆结论

G3 以 ring-adjacency 的原始可信边缘为主，允许椭圆在 ring unknown 区穿过；用端点切线连续性和可见背景软惩罚选择有限个确定性候选，未把 ring 或补全边缘作为数据点。完整轮廓人工遮挡模拟结果见 `tables/synthetic_occlusion_summary.csv`。在真实 39 组上，G3 median abs error 为 {mm.loc[G3,'median_abs_error_pct']:.3f}%（G2 {mm.loc[G2,'median_abs_error_pct']:.3f}%，G0 {mm.loc[G0,'median_abs_error_pct']:.3f}%），MAE {mm.loc[G3,'MAE_mm']:.3f} mm、p95 {mm.loc[G3,'p95_pct']:.3f}%、max {mm.loc[G3,'max_pct']:.3f}%，coverage {int(mm.loc[G3,'coverage'])}/39。G3 有效帧 {int(ss.loc[G3,'valid_frames'])}、eligible capture {int(ss.loc[G3,'eligible_captures'])}，fallback capture {int(ss.loc[G3,'fallback_captures'])}。相对 G2，{imp} 组改善、{worse} 组恶化、{39-changed} 组未变。是否构成真实几何改善须以 PNG 人工审查为准：G3 改善可能来自可见边缘与遮挡连续性，也可能仅是 fallback 改变；本实验不晋升主线。'''
 (OUT/'HUMAN_REPORT.md').write_text(human,encoding='utf8')
 tech=f'''# G3 Occlusion-aware amodal ellipse — Technical report

## Frozen scope

G0/G2 blind inputs、selected stems、ring scale、单帧 `Dxy_i`、capture median（>=4）及 formal group fusion 全部从既有冻结文件复用。`blind` 运行前不读取 reference/GT；其 SHA 位于 `inputs/blind_sha256.txt`。GT 仅在 `evaluate` 后用于本报告和评价表。

## Algorithm

拟合点只有 ring-adjacency `high_confidence` 的原始 polyp contour；`ring_cut` 永不参与，`uncertain` 只诊断。ring band 是 unknown U。对 G2 seed、全可信点 AMS seed、以及前若干空间分离 arc-pair seed，生成至多 {MAX_STARTS} 个确定性宽范围 starts（轴长 scale 0.88/1/1.15/1.30，中心可沿主轴移 0.14 rr），并最小化归一化残差：

`L = L_edge(Huber) + {LAMBDA_TANGENT} L_tangent + {LAMBDA_OCCLUSION} L_occlusion + {LAMBDA_BACKGROUND} L_background`。

其中 edge 是可信点几何距离；tangent 仅作用于靠 U 的可信 arc 端点；occlusion 检查预测边界是否能进入该端点的 U；background 是 ellipse 边界落入明确可见背景的软距离惩罚。没有可靠端点则没有 tangent/occlusion 项。候选若近最优集合长轴跨度 >{AMBIGUOUS_DIAMETER_SPREAD:.0%} 则标为 `ambiguous_occlusion_completion`，不强制输出。

## Results

|geometry|coverage|median abs %|MAE mm|p95 %|max %|valid frames|eligible captures|fallback captures|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
 for geo in (G0,G2,G3):tech+=f"|{geo}|{int(mm.loc[geo,'coverage'])}|{mm.loc[geo,'median_abs_error_pct']:.3f}|{mm.loc[geo,'MAE_mm']:.3f}|{mm.loc[geo,'p95_pct']:.3f}|{mm.loc[geo,'max_pct']:.3f}|{int(ss.loc[geo,'valid_frames'])}|{int(ss.loc[geo,'eligible_captures'])}|{int(ss.loc[geo,'fallback_captures'])}|\n"
 synlines=['|geometry|valid|n|median abs relative major error %|','|---|---:|---:|---:|']
 for x in synsum.itertuples(index=False):synlines.append(f'|{x.geometry}|{int(x.valid)}|{int(x.n)}|{x.median_abs_relative_major_error_pct:.3f}|')
 tech+=f'''\nG3 vs G2 group 变化：改善 {imp}、恶化 {worse}、未变 {39-changed}。在二者均有效的 {len(both)} 帧中，G3-G2 长轴相对变化中位数为 {both.G3_minus_G2_L_relative_pct.median():.3f}%，变长/变短为 {longer}/{shorter}；完整逐帧方向和 candidate diameter spread 在 `tables/g3_vs_g2_long_axis.csv`。不确定性 gate 以 `ambiguous_occlusion_completion` 拒绝 {ambiguous} 帧；所有 fit reason 计数在 `tables/g3_fit_reason_counts.csv`。frame/capture 明细在 `tables/` 与 `G3_results.xlsx`；PNG 复核在 `visualizations/`。这里区分三类：有 G3 椭圆并改变 Dxy 的为候选几何改变；G3 invalid 导致 capture 无替换的是 fallback 改变；只看图形更顺滑而无 paired 值支持的仅为视觉改善。\n\nSynthetic occlusion summary:\n\n{chr(10).join(synlines)}\n'''
 (OUT/'TECHNICAL_REPORT.md').write_text(tech,encoding='utf8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['blind','evaluate']);a=p.parse_args();{'blind':blind,'evaluate':evaluate}[a.stage]()
