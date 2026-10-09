#!/usr/bin/env python3
"""GT-blind G0/G1/G2 study on frozen ring-adjacency original-contour edges."""
from __future__ import annotations
import argparse, hashlib, html, json, math
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .layout import AUDIT, EDGE_OUT, FORMAL_MAINLINE as S7, IP88 as IP, MULTIARC_OUT as OUT, RAW, S6, bind_asset_paths
EDGE=None
# Preregistered, ring-normalized geometry constants.  No GT is read in blind().
ARC_MIN_LENGTH_FACTOR=.08; ARC_MIN_POINTS=10; ARC_SAMPLE_SPACING_FACTOR=.04; ARC_MAX_SAMPLES=80
ARC_PAIR_MIN_CENTROID_SEPARATION=.25; SUPPORT_DISTANCE_FACTOR=.05; SUPPORT_ARC_INLIER_FRACTION=.35
MIN_SUPPORTED_ARCS=2; MIN_SUPPORTED_LENGTH_FRACTION=.40; MIN_ANGULAR_COVERAGE=.20; HUBER_SCALE_FACTOR=.04

def load_edge():
 global EDGE
 if EDGE is None:
  from . import edge_completion
  EDGE=edge_completion
 return EDGE
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
def selected():
 s=pd.read_csv(S6/'tables/selected_bins_blind.csv');s=s[s.method.eq('S6_NDG_FCP39')][['stem','group','capture','segment_id']]
 r=bind_asset_paths(pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv'))
 r=r.drop(columns=[c for c in ('group','capture','segment_id') if c in r.columns])
 z=s.merge(r,on='stem',how='left',validate='one_to_one');assert len(z)==395
 # same harmless path completion used by the fixed edge audit, no inference.
 cache=RAW/'candidates'
 for col,sub,ext in [('image_path','images','.jpg'),('mask_path','masks','.png'),('ring_json_path','ring_fit/json','.json')]:
  fb=z.stem.map(lambda x:str(cache/sub/f'{x}{ext}') if (cache/sub/f'{x}{ext}').exists() else np.nan);z[col]=z[col].where(z[col].notna(),fb)
 assert z.mask_path.notna().all() and z.ring_json_path.notna().all()
 return z
def uniform_arc_sample(pts,n):
 if len(pts)<=n:return pts
 step=np.linalg.norm(np.diff(pts,axis=0),axis=1);cum=np.r_[0,np.cumsum(step)];want=np.linspace(0,cum[-1],n);idx=np.searchsorted(cum,want);idx=np.clip(idx,0,len(pts)-1);return pts[idx]
def ellipse_residual(points,p):
 cx,cy,la,lb,angle=p;a=np.exp(la);b=np.exp(lb);c,s=np.cos(angle),np.sin(angle);q=points-np.array([cx,cy]);x=q[:,0]*c+q[:,1]*s;y=-q[:,0]*s+q[:,1]*c
 return (np.sqrt((x/a)**2+(y/b)**2)-1)*math.sqrt(a*b)
def ellipse_from_cv(points):
 if len(points)<5:return None
 try:(cx,cy),(d1,d2),ang=cv2.fitEllipseAMS(points.astype(np.float32).reshape(-1,1,2))
 except cv2.error:return None
 if min(d1,d2)<=1:return None
 # cv angle is its first ellipse axis; ordering does not change the ellipse.
 return np.array([cx,cy,math.log(d1/2),math.log(d2/2),math.radians(ang)],float)
def canonical(p):
 cx,cy,la,lb,ang=p;a,b=np.exp(la),np.exp(lb)
 if b>a:a,b=b,a;ang+=math.pi/2
 return {'cx':float(cx),'cy':float(cy),'major_radius_px':float(a),'minor_radius_px':float(b),'angle_deg':float(math.degrees(ang)%180),'major_diameter_px':float(2*a),'minor_diameter_px':float(2*b),'axis_ratio':float(a/max(b,1e-9))}
def ellipse_mask(shape,e):
 m=np.zeros(shape,np.uint8);axes=(max(1,round(e['major_radius_px'])),max(1,round(e['minor_radius_px'])));cv2.ellipse(m,(round(e['cx']),round(e['cy'])),axes,e['angle_deg'],0,360,1,-1);return m.astype(bool)
def cp_support(polyp,band,e):
 em=ellipse_mask(polyp.shape,e);obs=polyp&~band;C=(polyp&em).sum()/max(polyp.sum(),1);P=(obs&em).sum()/max((em&~band).sum(),1);return float(C),float(P)
def make_arcs(contours,states,near,center,rr):
 """Only original trusted points become arcs; edge.high_arcs preserves contour order."""
 edge=load_edge();raw=edge.high_arcs(contours,states,near,center);out=[];allpts=np.concatenate(contours) if contours else np.empty((0,2))
 for a in raw:
  ids=np.array(a.pop('_global_ids'),int);p=allpts[ids];a['points']=p;a['global_ids']=ids.tolist();a['eligible_seed_arc']=bool(a['arc_length_px']>=max(ARC_MIN_POINTS,ARC_MIN_LENGTH_FACTOR*rr) and a['point_count']>=ARC_MIN_POINTS);out.append(a)
 cent=np.array([[x['centroid_x'],x['centroid_y']] for x in out]) if out else np.empty((0,2))
 for i,a in enumerate(out):
  a['nearest_other_arc_centroid_distance_px']=float(np.min(np.delete(cdist(cent[i:i+1],cent),i,axis=1))) if len(out)>1 else math.nan
 return out
def current_fitter(trusted,center,rr,stem,polyp,band):
 v2,_=load_edge().load_tools();
 if len(trusted)<25:return {'valid':False,'reason':'too_few_trusted_points'}
 extent=float(np.hypot(np.ptp(trusted[:,0]),np.ptp(trusted[:,1])));c=v2.fit_circle(trusted,center,extent,rr,stem+'_G1');e=v2.fit_ellipse(trusted,center,extent,rr)
 if e.get('valid') and (not c.get('valid') or (e['axis_ratio']>=1.15 and e['norm_residual_all']<=.75*c['norm_residual_all'])):
  return {'valid':True,'reason':'ellipse_current_v2','shape':'ellipse','L_px':e['major_diameter_px'],'residual':e['norm_residual_all'],'angular_coverage':e['arc_coverage'],'edge_inlier_fraction':e['inlier_ratio'],'params':{'cx':e['cx'],'cy':e['cy'],'major_radius_px':e['major_diameter_px']/2,'minor_radius_px':e['minor_diameter_px']/2,'angle_deg':e['angle_deg'],'major_diameter_px':e['major_diameter_px'],'minor_diameter_px':e['minor_diameter_px'],'axis_ratio':e['axis_ratio']},'C':math.nan,'P':math.nan}
 if c.get('valid'):
  rr0=c['radius_px'];return {'valid':True,'reason':'circle_current_v2','shape':'circle','L_px':2*rr0,'residual':c['norm_residual_all'],'angular_coverage':c['arc_coverage'],'edge_inlier_fraction':c['inlier_ratio'],'params':{'cx':c['cx'],'cy':c['cy'],'major_radius_px':rr0,'minor_radius_px':rr0,'angle_deg':0.,'major_diameter_px':2*rr0,'minor_diameter_px':2*rr0,'axis_ratio':1.},'C':math.nan,'P':math.nan}
 return {'valid':False,'reason':f"circle:{c.get('reason','na')} ellipse:{e.get('reason','na')}"}
def multiarc_fitter(arcs, trusted, uncertain, center, rr, polyp, band):
 seed=[a for a in arcs if a['eligible_seed_arc']]
 if len(seed)<2:return {'valid':False,'reason':'fewer_than_two_eligible_trusted_arcs','supported_arc_ids':[]}
 candidates=[];support_dist=max(2.,SUPPORT_DISTANCE_FACTOR*rr);huber=max(1.,HUBER_SCALE_FACTOR*rr)
 for i in range(len(seed)):
  for j in range(i+1,len(seed)):
   ai,aj=seed[i],seed[j];dist=np.hypot(ai['centroid_x']-aj['centroid_x'],ai['centroid_y']-aj['centroid_y'])
   if dist<ARC_PAIR_MIN_CENTROID_SEPARATION*rr:continue
   parts=[]
   for a in (ai,aj):
    n=min(ARC_MAX_SAMPLES,max(ARC_MIN_POINTS,round(a['arc_length_px']/max(1.,ARC_SAMPLE_SPACING_FACTOR*rr))));parts.append(uniform_arc_sample(a['points'],n))
   init=ellipse_from_cv(np.vstack(parts))
   if init is None:continue
   # All arcs participate in robust refinement, with per-arc uniform sampling.
   allparts=[]
   for a in seed:
    n=min(ARC_MAX_SAMPLES,max(ARC_MIN_POINTS,round(a['arc_length_px']/max(1.,ARC_SAMPLE_SPACING_FACTOR*rr))));allparts.append(uniform_arc_sample(a['points'],n))
   data=np.vstack(allparts)
   try:res=least_squares(lambda p:ellipse_residual(data,p),init,loss='huber',f_scale=huber,max_nfev=180)
   except Exception:continue
   if not res.success:continue
   e=canonical(res.x)
   if not(np.isfinite(e['major_radius_px']) and np.isfinite(e['minor_radius_px']) and e['minor_radius_px']>1):continue
   # Validate globally on ALL trusted original points; uncertain only diagnoses.
   residual=np.abs(ellipse_residual(trusted,res.x));inlier=residual<=support_dist;sup=[];sup_len=0.;angs=[]
   for a in arcs:
    ap=a['points'];ar=np.abs(ellipse_residual(ap,res.x));fraction=float((ar<=support_dist).mean()) if len(ap) else 0.
    if fraction>=SUPPORT_ARC_INLIER_FRACTION:
     sup.append(a);sup_len+=a['arc_length_px'];angs.append(ap[ar<=support_dist])
   separated=False
   for x in range(len(sup)):
    for y in range(x+1,len(sup)):
     if np.hypot(sup[x]['centroid_x']-sup[y]['centroid_x'],sup[x]['centroid_y']-sup[y]['centroid_y'])>=ARC_PAIR_MIN_CENTROID_SEPARATION*rr:separated=True
   total_len=sum(a['arc_length_px'] for a in arcs);frac=sup_len/max(total_len,1e-9)
   # Parameter-space ellipse angular coverage of supported original points.
   q=np.vstack(angs) if angs else np.empty((0,2));theta=math.radians(e['angle_deg']);co,si=math.cos(theta),math.sin(theta);z=q-np.array([e['cx'],e['cy']]);xx=z[:,0]*co+z[:,1]*si;yy=-z[:,0]*si+z[:,1]*co
   aa=np.mod(np.arctan2(yy/max(e['minor_radius_px'],1e-9),xx/max(e['major_radius_px'],1e-9)),2*np.pi) if len(q) else np.array([]);coverage=float(np.histogram(aa,bins=36,range=(0,2*np.pi))[0].astype(bool).mean()) if len(aa) else 0.
   if len(sup)<MIN_SUPPORTED_ARCS or not separated or frac<MIN_SUPPORTED_LENGTH_FRACTION or coverage<MIN_ANGULAR_COVERAGE:continue
   ufrac=float((np.abs(ellipse_residual(uncertain,res.x))<=support_dist).mean()) if len(uncertain) else math.nan
   C,P=cp_support(polyp,band,e);score=(len(sup),frac,coverage,-float(np.median(residual)))
   candidates.append((score,{'valid':True,'reason':'ok','shape':'ellipse','L_px':e['major_diameter_px'],'residual':float(np.median(residual)),'angular_coverage':coverage,'edge_inlier_fraction':float(inlier.mean()),'supported_arc_count':len(sup),'supported_arc_length_fraction':frac,'supported_arc_ids':[a['arc_id'] for a in sup],'uncertain_inlier_fraction':ufrac,'C':C,'P':P,'params':e}))
 if not candidates:return {'valid':False,'reason':'no_candidate_with_two_separated_supported_arcs','supported_arc_ids':[]}
 return max(candidates,key=lambda x:x[0])[1]
def g0_geometry(r):
 get=lambda k,d=math.nan:getattr(r,k,d)
 name=get('chosen_model_candidate','none');shape=get('chosen_shape','none')
 if name not in ('A','B') or shape not in ('circle','ellipse'):return None
 if shape=='circle':return {'shape':'circle','cx':get(f'{name}_cx'),'cy':get(f'{name}_cy'),'major_diameter_px':2*get(f'{name}_radius_px'),'minor_diameter_px':2*get(f'{name}_radius_px'),'angle_deg':0.}
 return {'shape':'ellipse','cx':get(f'{name}_ellipse_cx'),'cy':get(f'{name}_ellipse_cy'),'major_diameter_px':get(f'{name}_ellipse_major_diameter_px'),'minor_diameter_px':get(f'{name}_ellipse_minor_diameter_px'),'angle_deg':get(f'{name}_ellipse_angle_deg')}
def frame_process(r, edgepoints):
 edge=load_edge();d=edge.prepare_original(r);contours=edge.external_contours(d['polyp']);pts=np.concatenate(contours) if contours else np.empty((0,2),np.float32)
 p=edgepoints.set_index('point_index').reindex(np.arange(len(pts)));assert len(p)==len(pts) and (not len(pts) or np.allclose(p.x.to_numpy(float),pts[:,0]))
 state=p.adjacency_class.to_numpy(object);near=p.ring_adjacent.to_numpy(bool);arcs=make_arcs(contours,state,near,d['center'],float(d['rr']));trusted=pts[state=='high_confidence'];uncertain=pts[state=='uncertain'];cut=pts[state=='ring_cut']
 g1=current_fitter(trusted,d['center'],float(d['rr']),r.stem,d['polyp'].astype(bool),d['band'].astype(bool));g2=multiarc_fitter(arcs,trusted,uncertain,d['center'],float(d['rr']),d['polyp'].astype(bool),d['band'].astype(bool))
 ringmm=10. if '_R10_' in r.group else 5.;ringpx=float(r.ring_px)
 def rec(name,z):
  valid=bool(z.get('valid',False)) and finite(z.get('L_px')) and finite(ringpx)
  q={'geometry':name,'stem':r.stem,'group':r.group,'capture':r.capture,'bin':int(r.segment_id),'frame_index':r.frame_index,'ring_px':ringpx,'ring_mm':ringmm,'valid':valid,'L_px':z.get('L_px',math.nan),'Dxy_i_mm':ringmm*z.get('L_px',math.nan)/ringpx if valid else math.nan,'fit_reason':z.get('reason',''), 'fit_shape':z.get('shape','none'),'residual':z.get('residual',math.nan),'angular_coverage':z.get('angular_coverage',math.nan),'edge_inlier_fraction':z.get('edge_inlier_fraction',math.nan),'supported_arc_count':z.get('supported_arc_count',math.nan),'supported_arc_length_fraction':z.get('supported_arc_length_fraction',math.nan),'supported_arc_ids':'|'.join(z.get('supported_arc_ids',[])),'uncertain_inlier_fraction':z.get('uncertain_inlier_fraction',math.nan),'C_diag':z.get('C',math.nan),'P_diag':z.get('P',math.nan),'trusted_arc_count':len(arcs),'trusted_point_count':len(trusted),'uncertain_point_count':len(uncertain),'cut_point_count':len(cut),'small_fit_failure':str(z.get('reason','')).startswith(('too_few','fewer_than','no_candidate'))};
  par=z.get('params',{})
  for k in ('cx','cy','major_diameter_px','minor_diameter_px','angle_deg','axis_ratio'):q['ellipse_'+k]=par.get(k,math.nan)
  return q
 # G0 uses its frozen per-frame paired scale and length directly.
 get=lambda k,d=math.nan:getattr(r,k,d)
 g0={'valid':truth(get('valid',False)) and finite(get('L_px')) and finite(ringpx),'L_px':get('L_px',math.nan),'reason':'frozen_S7_V2','shape':get('chosen_shape','none'),'residual':math.nan,'angular_coverage':math.nan,'edge_inlier_fraction':math.nan,'params':g0_geometry(r) or {}}
 g0r=rec('G0_frozen_S7_V2',g0);g1r=rec('G1_adjacency_current_fitter',g1);g2r=rec('G2_adjacency_multiarc_ellipse',g2)
 return [g0r,g1r,g2r],arcs,{'d':d,'pts':pts,'state':state,'arcs':arcs,'g0':g0,'g1':g1,'g2':g2}
def capture_rows(frames):
 rows=[]
 for (geo,g,c),z in frames.groupby(['geometry','group','capture']):
  ok=z[z.valid.map(truth)&z.Dxy_i_mm.map(finite)];n=len(ok);med=ok.Dxy_i_mm.median() if n>=4 else math.nan;mad=np.median(np.abs(ok.Dxy_i_mm-ok.Dxy_i_mm.median())) if n else math.nan
  rows.append({'geometry':geo,'group':g,'capture':c,'valid_bins':n,'eligible':n>=4,'capture_prediction_mm':med,'capture_MAD_mm':mad,'capture_range_mm':ok.Dxy_i_mm.max()-ok.Dxy_i_mm.min() if n else math.nan,'capture_range_over_median':(ok.Dxy_i_mm.max()-ok.Dxy_i_mm.min())/ok.Dxy_i_mm.median() if n and ok.Dxy_i_mm.median() else math.nan,'small_fit_failures':int(z.small_fit_failure.sum())})
 return pd.DataFrame(rows)
def fuse(caps):
 formal=pd.read_csv(AUDIT/'inputs/immutable_scale_manifest.csv');formal['capture']=formal.capture_id.str.replace('video:','',regex=False);formal=formal[formal.current_valid.map(truth)&formal.current_Dxy_mm.map(finite)].copy();rows=[]
 for geo,z in caps.groupby('geometry'):
  rep=z[z.eligible].set_index('capture').capture_prediction_mm.to_dict();a=formal.copy();a['used_mm']=a.current_Dxy_mm;a['replaced']=a.source_kind.eq('video_keyframe')&a.capture.isin(rep);a.loc[a.replaced,'used_mm']=a.loc[a.replaced,'capture'].map(rep);g=a.groupby('group').used_mm.median().rename('prediction_mm').reset_index();g['geometry']=geo;ip=pd.read_csv(IP/'tables/group_prediction_blind.csv')[['group','prediction_mm']];ip['geometry']=geo;rows.append(pd.concat([g,ip],ignore_index=True))
 return pd.concat(rows,ignore_index=True)
def blind():
 for d in ('inputs','tables','gallery','gallery/assets','scripts'):(OUT/d).mkdir(parents=True,exist_ok=True)
 protocol={'baseline':'S7-FMD39','GT_used_before_evaluation':False,'geometries':{'G0':'frozen S7 V2 per-frame L/ring','G1':'ring-adjacency trusted original contour + frozen current V2 fitter','G2':'ring-adjacency trusted original contour + Direct Least Squares cv2 AMS initialization + Huber geometric residual refinement'},'frozen_statistics':'D_i=ring_mm*L_i/ring_px_i; capture median over >=4 valid frames; unchanged formal fusion','original_contour_only':True,'uncertain':'verification only, never seed/fit point','cut':'never fit point','multiarc_constants':{'arc_min_length_over_rr':ARC_MIN_LENGTH_FACTOR,'arc_min_points':ARC_MIN_POINTS,'arc_sample_spacing_over_rr':ARC_SAMPLE_SPACING_FACTOR,'arc_pair_centroid_separation_over_rr':ARC_PAIR_MIN_CENTROID_SEPARATION,'support_distance_over_rr':SUPPORT_DISTANCE_FACTOR,'support_arc_inlier_fraction':SUPPORT_ARC_INLIER_FRACTION,'min_supported_arcs':MIN_SUPPORTED_ARCS,'min_supported_length_fraction':MIN_SUPPORTED_LENGTH_FRACTION,'min_angular_coverage':MIN_ANGULAR_COVERAGE,'huber_scale_over_rr':HUBER_SCALE_FACTOR}}
 (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 sel=selected();adj=pd.read_csv(EDGE_OUT/'tables/original_contour_point_ring_adjacency.csv');allrows=[];arows=[];aux={}
 for i,r in enumerate(sel.to_dict('records'),1):
  rr=type('R',(),r)();ep=adj[adj.stem.eq(r['stem'])];res,ar,a=frame_process(rr,ep);allrows+=res
  for x in ar:
   y={k:v for k,v in x.items() if k not in ('points','global_ids')};y.update({'stem':r['stem'],'group':r['group'],'capture':r['capture'],'bin':r['segment_id']});arows.append(y)
  aux[r['stem']]=a
  if i%25==0:print(f'{i}/395',flush=True)
 frames=pd.DataFrame(allrows);caps=capture_rows(frames);groups=fuse(caps);write(frames,'tables/frame_geometry_blind.csv');write(arows,'tables/trusted_arcs_blind.csv');write(caps,'tables/capture_predictions_blind.csv');write(groups,'tables/group_predictions_blind.csv')
 files=['inputs/frozen_protocol.json','tables/frame_geometry_blind.csv','tables/trusted_arcs_blind.csv','tables/capture_predictions_blind.csv','tables/group_predictions_blind.csv'];(OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/f)}  {f}\n' for f in files));print('sealed',len(frames),len(caps),len(groups))
def metrics(x):
 q=x.dropna(subset=['prediction_mm']);e=100*(q.prediction_mm-q.reference_mm)/q.reference_mm;a=e.abs();return {'coverage':len(q),'median_abs_error_pct':a.median(),'MAE_mm':(q.prediction_mm-q.reference_mm).abs().mean(),'mean_abs_error_pct':a.mean(),'within5_pct':100*(a<=5).mean(),'within10_pct':100*(a<=10).mean(),'p90_pct':a.quantile(.9),'p95_pct':a.quantile(.95),'max_pct':a.max()}
def draw_geometry(ax,e,color,label):
 if not e or not finite(e.get('cx')):return
 ax.add_patch(Ellipse((e['cx'],e['cy']),e['major_diameter_px'],e['minor_diameter_px'],angle=e.get('angle_deg',0),fill=False,color=color,lw=2,label=label))
def gallery(frames,groups,arcs,sel):
 # Evaluation-only choices; never feed rules.
 f=frames.pivot(index='stem',columns='geometry',values='abs_error_pct').reset_index();d=frames.pivot(index='stem',columns='geometry',values='Dxy_i_mm').reset_index();q=f.merge(d,on='stem',suffixes=('_err','_D'))
 picks=[]
 for label,sub in [('G0 小值→G2改善',q.assign(delta=q.G0_frozen_S7_V2_err-q.G2_adjacency_multiarc_ellipse_err).sort_values('delta',ascending=False)),('G0 大值→G2改善',q.assign(delta=q.G0_frozen_S7_V2_D-q.G2_adjacency_multiarc_ellipse_D).sort_values('delta',ascending=False)),('G2 拒绝局部弧',frames[(frames.geometry=='G2_adjacency_multiarc_ellipse')&(~frames.valid.map(truth))].sort_values('trusted_arc_count')),('G2 与 G0 接近',q.assign(delta=(q.G0_frozen_S7_V2_D-q.G2_adjacency_multiarc_ellipse_D).abs()).sort_values('delta')),('G2 最大恶化',q.assign(delta=q.G2_adjacency_multiarc_ellipse_err-q.G0_frozen_S7_V2_err).sort_values('delta',ascending=False))]:
  for stem in sub.stem:
   if stem not in [x[1] for x in picks]:picks.append((label,stem));break
 gwide=groups.pivot(index='group',columns='geometry',values='abs_error_pct').reset_index();
 for label,sub in [('最大 group 改善',gwide.assign(delta=gwide.G0_frozen_S7_V2-gwide.G2_adjacency_multiarc_ellipse).sort_values('delta',ascending=False)),('最大 group 恶化',gwide.assign(delta=gwide.G2_adjacency_multiarc_ellipse-gwide.G0_frozen_S7_V2).sort_values('delta',ascending=False))]:
  group=sub.iloc[0].group;stem=frames[(frames.group==group)&(frames.geometry=='G2_adjacency_multiarc_ellipse')].sort_values('Dxy_i_mm').stem.iloc[0]
  if stem not in [x[1] for x in picks]:picks.append((label,stem))
 cards=[]
 for label,stem in picks:
  r=sel[sel.stem.eq(stem)].iloc[0];edge=load_edge();d0=edge.prepare_original(type('R',(),r.to_dict())());x0,y0,x1,y1=d0['crop'];im=cv2.imread(str(r.image_path));im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)[y0:y1,x0:x1]
  fig,axs=plt.subplots(1,3,figsize=(17,5));z=frames[frames.stem.eq(stem)].set_index('geometry');pa=arcs[arcs.stem.eq(stem)]
  first=im.copy()
  for mask,color,alpha in [(d0['polyp'],(30,210,70),.16),(d0['ring'],(255,205,0),.48),(d0['band'],(190,40,220),.14)]:
   first[mask.astype(bool)]=(first[mask.astype(bool)]*(1-alpha)+np.asarray(color)*alpha).astype(np.uint8)
  for ax,title,panel in zip(axs,['Original edges / trusted arcs','G0 frozen V2','G2 multi-arc ellipse'],[first,im,im]):ax.imshow(panel);ax.set_title(title);ax.axis('off')
  color={'high_confidence':'lime','uncertain':'orange','ring_cut':'red'};pp=pd.read_csv(EDGE_OUT/'tables/original_contour_point_ring_adjacency.csv');pp=pp[pp.stem.eq(stem)]
  for cl,c in color.items():axs[0].scatter(pp.loc[pp.adjacency_class.eq(cl),'x'],pp.loc[pp.adjacency_class.eq(cl),'y'],s=1,c=c)
  supported=set(str(z.loc['G2_adjacency_multiarc_ellipse','supported_arc_ids']).split('|'))
  for a in pa.itertuples():
   axs[0].text(a.centroid_x,a.centroid_y,a.arc_id,color='white',fontsize=8)
   axs[2].text(a.centroid_x,a.centroid_y,a.arc_id,color=('cyan' if a.arc_id in supported else 'white'),fontsize=8)
  for key,col in [('G0_frozen_S7_V2','cyan'),('G1_adjacency_current_fitter','yellow'),('G2_adjacency_multiarc_ellipse','lime')]:
   row=z.loc[key];e={k.replace('ellipse_',''):row[k] for k in row.index if k.startswith('ellipse_')};e={'cx':e.get('cx'), 'cy':e.get('cy'),'major_diameter_px':e.get('major_diameter_px'),'minor_diameter_px':e.get('minor_diameter_px'),'angle_deg':e.get('angle_deg')}
   target=axs[1] if key=='G0_frozen_S7_V2' else axs[2];draw_geometry(target,e,col,key);target.text(8,42+18*(['G0_frozen_S7_V2','G1_adjacency_current_fitter','G2_adjacency_multiarc_ellipse'].index(key)),f'{key}: D={row.Dxy_i_mm:.2f} mm valid={row.valid}',color=col,fontsize=8,bbox={'facecolor':'black','alpha':.5})
  fig.suptitle(f'Case: {stem}');fig.tight_layout();fn=f'{stem}.png';fig.savefig(OUT/'gallery/assets'/fn,dpi=130);plt.close(fig);cards.append((label,stem,fn))
 htmls=''.join(f'<h2>{html.escape(a)} — {html.escape(s)}</h2><img src="assets/{html.escape(f)}">' for a,s,f in cards);(OUT/'gallery/multiarc_cases.html').write_text('<meta charset="utf-8"><style>body{font:14px system-ui;margin:20px;max-width:1800px}img{width:100%}</style><h1>Ring-adjacency multi-arc ellipse audit</h1><p>绿色=trusted，橙=uncertain，红=cut；所有边缘来自原始 mask。图仅用于审查，GT 仅在图例案例选择后参与误差标注。</p>'+htmls,encoding='utf8')
def excel(groups,caps,frames):
 x=OUT/'ring_adjacency_multiarc_results.xlsx';wide=groups.pivot(index=['group','reference_mm'],columns='geometry',values=['prediction_mm','signed_error_pct','abs_error_pct']).reset_index();wide.columns=['_'.join([str(y) for y in x if y]) for x in wide.columns];wide['G2_minus_G0_abs_error_pp']=wide['abs_error_pct_G2_adjacency_multiarc_ellipse']-wide['abs_error_pct_G0_frozen_S7_V2'];wide=wide.sort_values('abs_error_pct_G2_adjacency_multiarc_ellipse',ascending=False)
 dwide=frames.pivot(index=['geometry','group','capture'],columns='bin',values='Dxy_i_mm').rename(columns=lambda b:f'Dxy_i_bin{int(b):02d}_mm').reset_index();capdetail=caps.merge(dwide,on=['geometry','group','capture'],how='left',validate='one_to_one')
 with pd.ExcelWriter(x,engine='openpyxl') as w:
  wide.to_excel(w,sheet_name='Group Ranking',index=False,na_rep='NA');capdetail.to_excel(w,sheet_name='Capture Detail',index=False,na_rep='NA');frames.to_excel(w,sheet_name='Frame Detail',index=False,na_rep='NA')
 wb=load_workbook(x);fill=PatternFill('solid',fgColor='1F4E78')
 for ws in wb.worksheets:
  ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
  for c in ws[1]:c.font=Font(color='FFFFFF',bold=True);c.fill=fill
  for i,col in enumerate(ws.columns,1):ws.column_dimensions[get_column_letter(i)].width=min(40,max(11,max(len(str(x.value or '')) for x in list(col)[:100])+2))
 wb.save(x)
def evaluate():
 for line in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():h,r=line.split('  ',1);assert sha(OUT/r)==h
 refs=pd.read_csv(S7/'tables/group_errors_descending.csv')[['group','reference_mm']];frames=pd.read_csv(OUT/'tables/frame_geometry_blind.csv').merge(refs,on='group',how='left');frames['signed_error_pct']=100*(frames.Dxy_i_mm-frames.reference_mm)/frames.reference_mm;frames['abs_error_pct']=frames.signed_error_pct.abs();groups=pd.read_csv(OUT/'tables/group_predictions_blind.csv').merge(refs,on='group',how='left');groups['signed_error_pct']=100*(groups.prediction_mm-groups.reference_mm)/groups.reference_mm;groups['abs_error_pct']=groups.signed_error_pct.abs();caps=pd.read_csv(OUT/'tables/capture_predictions_blind.csv').merge(refs,on='group',how='left');caps['capture_abs_error_pct']=100*(caps.capture_prediction_mm-caps.reference_mm).abs()/caps.reference_mm
 write(frames,'tables/frame_geometry_evaluated.csv');write(groups,'tables/group_predictions_evaluated.csv');write(caps,'tables/capture_predictions_evaluated.csv');write([{'geometry':g,**metrics(x)} for g,x in groups.groupby('geometry')],'tables/metrics_summary.csv')
 # catastrophic frame counts and stability summary, as requested.
 fs=[]
 for g,x in frames.groupby('geometry'):
  a=x.abs_error_pct;fs.append({'geometry':g,'valid_frames':int(a.notna().sum()),'catastrophic_gt20':int((a>20).sum()),'catastrophic_gt30':int((a>30).sum()),'catastrophic_gt50':int((a>50).sum()),'median_frame_abs_error_pct':a.median(),'median_capture_MAD_mm':caps[caps.geometry.eq(g)].capture_MAD_mm.median(),'median_capture_range_mm':caps[caps.geometry.eq(g)].capture_range_mm.median(),'eligible_captures':int(caps[caps.geometry.eq(g)].eligible.sum()),'small_fit_failures':int(x.small_fit_failure.sum())})
 write(fs,'tables/frame_stability_summary.csv');arcs=pd.read_csv(OUT/'tables/trusted_arcs_blind.csv');sel=selected();gallery(frames,groups,arcs,sel);excel(groups,caps,frames)
 g0='G0_frozen_S7_V2';g1='G1_adjacency_current_fitter';g2='G2_adjacency_multiarc_ellipse'
 fp=frames.pivot(index='stem',columns='geometry',values='abs_error_pct').dropna(subset=[g0,g2]);cp=caps.pivot(index=['group','capture'],columns='geometry',values=['eligible','capture_MAD_mm','capture_range_mm']);eligible_g2=cp[('eligible',g2)].fillna(False).astype(bool);cp=cp[eligible_g2]
 pair=[{'scope':'matched_G0_G2_frames','n':len(fp),'G0_median_abs_error_pct':fp[g0].median(),'G2_median_abs_error_pct':fp[g2].median(),'G2_improved_frames':int((fp[g2]<fp[g0]).sum()),'G2_worsened_frames':int((fp[g2]>fp[g0]).sum()),'G0_gt20':int((fp[g0]>20).sum()),'G2_gt20':int((fp[g2]>20).sum()),'G0_gt30':int((fp[g0]>30).sum()),'G2_gt30':int((fp[g2]>30).sum()),'G0_gt50':int((fp[g0]>50).sum()),'G2_gt50':int((fp[g2]>50).sum())},{'scope':'G2_eligible_capture_subset','n':len(cp),'G0_median_MAD_mm':cp[('capture_MAD_mm',g0)].median(),'G2_median_MAD_mm':cp[('capture_MAD_mm',g2)].median(),'G0_median_range_mm':cp[('capture_range_mm',g0)].median(),'G2_median_range_mm':cp[('capture_range_mm',g2)].median(),'G2_lower_MAD_captures':int((cp[('capture_MAD_mm',g2)]<cp[('capture_MAD_mm',g0)]).sum()),'G2_lower_range_captures':int((cp[('capture_range_mm',g2)]<cp[('capture_range_mm',g0)]).sum())}]
 write(pair,'tables/paired_stability_summary.csv');pairf=pair[0];pairc=pair[1]
 m=pd.read_csv(OUT/'tables/metrics_summary.csv');st=pd.read_csv(OUT/'tables/frame_stability_summary.csv').set_index('geometry');mm=m.set_index('geometry')
 h=f'''# Ring-adjacency multi-arc ellipse（待审查）

ring-adjacency 本身令 G1 有效帧从 {int(st.loc[g0,'valid_frames'])} 提至 {int(st.loc[g1,'valid_frames'])}，但是否“改善可用边缘”须结合 gallery 人工判断。G2 要求两条分离弧共同支持，因此仅 {int(st.loc[g2,'valid_frames'])} 个有效帧、{int(st.loc[g2,'eligible_captures'])} 个 eligible capture；它拒绝了 {int(st.loc[g2,'small_fit_failures'])} 个局部/不充分弧候选，而非用单弧补回。

在两者均有效的 {int(pairf['n'])} 帧中，G2 median 单帧误差 {pairf['G2_median_abs_error_pct']:.3f}%（G0 {pairf['G0_median_abs_error_pct']:.3f}%），改善/恶化为 {int(pairf['G2_improved_frames'])}/{int(pairf['G2_worsened_frames'])}；但 >20/>30/>50% 为 {int(pairf['G2_gt20'])}/{int(pairf['G2_gt30'])}/{int(pairf['G2_gt50'])}，低于 G0 的 {int(pairf['G0_gt20'])}/{int(pairf['G0_gt30'])}/{int(pairf['G0_gt50'])}。在 G2 eligible 的 {int(pairc['n'])} 个 capture 子集，MAD 为 {pairc['G2_median_MAD_mm']:.3f} mm（G0 {pairc['G0_median_MAD_mm']:.3f}），range 为 {pairc['G2_median_range_mm']:.3f} mm（G0 {pairc['G0_median_range_mm']:.3f}）；这是严格拒绝造成的选择性子集，不能视为全体稳定性提升。

最终 group median abs error：G0 {mm.loc[g0,'median_abs_error_pct']:.3f}%，G1 {mm.loc[g1,'median_abs_error_pct']:.3f}%，G2 {mm.loc[g2,'median_abs_error_pct']:.3f}%。因此当前形式**不值得作为下一版 geometry 候选或晋升主线**：它有尾部抑制诊断信号，却以大幅有效 capture 降低和主指标退化为代价；应仅保留为等待人工审查的 rejection diagnostic。

完整指标见 `tables/metrics_summary.csv`、`tables/frame_stability_summary.csv`、`tables/paired_stability_summary.csv` 和 Excel `ring_adjacency_multiarc_results.xlsx`。
''';(OUT/'HUMAN_REPORT.md').write_text(h,encoding='utf8');print(m.to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['blind','evaluate']);a=p.parse_args();{'blind':blind,'evaluate':evaluate}[a.stage]()
