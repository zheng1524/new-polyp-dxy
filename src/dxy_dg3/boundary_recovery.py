#!/usr/bin/env python3
"""GT-blind ring-occlusion contour reclassification on frozen S6 stems."""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np, pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .layout import AUDIT, FORMAL_MAINLINE as MAIN, IP88, OUTPUT_ROOT, RAW, S6, bind_asset_paths
OUT = OUTPUT_ROOT / 'boundary_recovery'
SCALES=(.05,.10,.15,.20,.25); CONSENSUS=4; BOUNDARY_TOL=1.5; INTERIOR_TOL=2.; METHOD='V1_ring_recovered_original_contour'
V2=None
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
def load_v2():
 from . import frozen_v2
 return frozen_v2
def selected():
 s=pd.read_csv(S6/'tables/selected_bins_blind.csv');s=s[s.method.eq('S6_NDG_FCP39')][['stem','group','capture','segment_id','selection_outcome']]
 r=bind_asset_paths(pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv'));cols=['stem','frame_index','sharpness','ring_px','mm_per_px','image_path','mask_path','ring_json_path']
 z=s.merge(r[cols],on='stem',how='left',validate='one_to_one');assert len(z)==395 and z.image_path.notna().all();return z
def point_values(im,pts):
 x=np.clip(np.rint(pts[:,0]).astype(int),0,im.shape[1]-1);y=np.clip(np.rint(pts[:,1]).astype(int),0,im.shape[0]-1);return im[y,x]
def completion(M,U,rr):
 """Constrained closing: only accepted bridges in U joining two original components."""
 n,lab,stats,_=cv2.connectedComponentsWithStats(M.astype(np.uint8),8); substantial={i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>=max(12,int(.003*rr*rr))}
 comps=[]
 for fac in SCALES:
  rad=max(1,round(fac*rr));closed=cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rad+1,2*rad+1))).astype(bool)
  cand=closed&~M&U;num,cl,_,_=cv2.connectedComponentsWithStats(cand.astype(np.uint8),8);accepted=np.zeros_like(M,bool);data=[]
  for k in range(1,num):
   c=cl==k;touch=cv2.dilate(c.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&M; ids=set(np.unique(lab[touch]).tolist())&substantial
   ok=len(ids)>=2
   if ok:accepted|=c
   data.append({'scale_over_rr':fac,'radius_px':rad,'candidate_added_px':int(c.sum()),'accepted':ok,'touch_components':len(ids),'inside_unknown_fraction':float((c&U).sum()/max(c.sum(),1))})
  comps.append({'fac':fac,'completed':M|accepted,'bridge':accepted,'bridge_px':int(accepted.sum()),'raw_closed_added_px':int((closed&~M).sum()),'outside_U_added_px':int((closed&~M&~U).sum()),'components':data})
 return comps
def classify(M,U,rr,pts):
 runs=completion(M,U,rr);cut=np.zeros(len(pts),int);outer=np.zeros(len(pts),int)
 for x in runs:
  # distance-to-background: original boundary becomes interior only when a
  # valid bridge surrounds it. No completed contour point is returned.
  dt=cv2.distanceTransform(x['completed'].astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE);d=point_values(dt,pts)
  cut+=d>INTERIOR_TOL;outer+=d<=BOUNDARY_TOL
 guard=cv2.dilate(U.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*max(2,round(.06*rr))+1,)*2)).astype(bool)
 in_guard=point_values(guard,pts)>0
 cls=np.full(len(pts),'retained',object);cls[in_guard]='ambiguous';cls[in_guard&(cut>=CONSENSUS)]='artificial_cut';cls[in_guard&(outer>=CONSENSUS)&~(cut>=CONSENSUS)]='recovered_outer'
 return cls,guard,runs
def support(M,U,par):
 e=np.zeros(M.shape,np.uint8);cv2.ellipse(e,(round(par[0]),round(par[1])),(max(1,round(par[2])),max(1,round(par[3]))),par[4],0,360,1,-1);e=e.astype(bool);obs=M&~U
 cf=(M&e).sum()/max(M.sum(),1);co=(obs&e).sum()/max(obs.sum(),1);po=(obs&e).sum()/max((e&~U).sum(),1);return {'C_full':cf,'C_obs':co,'P_obs':po,'F_CP':2*co*po/max(co+po,1e-12),'G_CP':math.sqrt(co*po)}
def process(r):
 global V2
 if V2 is None:V2=load_v2()
 try:
  _,_,_,p=V2.process_view({'stem':r.stem,'group_key':r.group,'image_path':r.image_path,'mask_path':r.mask_path,'ring_fit_json':r.ring_json_path,'rect_scale':float(r.mm_per_px)})
  M=p['polyp'].astype(bool);U=p['band'].astype(bool);pts=V2.contour(M);cls,guard,runs=classify(M,U,float(p['rr']),pts)
  base={'stem':r.stem,'group':r.group,'capture':r.capture,'segment_id':r.segment_id,'frame_index':r.frame_index,'sharpness':r.sharpness,'ring_px':r.ring_px,'mm_per_px':r.mm_per_px,'image_path':r.image_path,'mask_path':r.mask_path,'ring_json_path':r.ring_json_path,'original_contour_n':len(pts),'v2_guard_n':int((point_values(guard,pts)>0).sum()),'recovered_n':int((cls=='recovered_outer').sum()),'cut_n':int((cls=='artificial_cut').sum()),'ambiguous_n':int((cls=='ambiguous').sum()),'scales_with_bridge':sum(x['bridge_px']>0 for x in runs),'bridge_px_median':float(np.median([x['bridge_px'] for x in runs]),),'raw_close_outside_U_px_max':max(x['outside_U_added_px'] for x in runs)}
  out=[]
  for ab in ('A','B'):
   # Restrict to original segmentation contour points residing in frozen A/B;
   # no morphology-created B or completion boundary enters fitting.
   member=point_values(p[ab].astype(bool),pts)>0; keep=member&((cls=='retained')|(cls=='recovered_outer'));vis=pts[keep].astype(np.float32)
   ys,xs=np.nonzero(p[ab]);geom=np.array([xs.mean(),ys.mean()]) if len(xs) else p['center'];extent=float(np.hypot(xs.max()-xs.min(),ys.max()-ys.min())) if len(xs) else float(p['rr']);rr=float(p['rr'])
   c=V2.fit_circle(vis,geom,extent,rr,r.stem+'_'+ab+'_recovered');e=V2.fit_ellipse(vis,geom,extent,rr)
   q=base.copy();q.update({'AB':ab,'candidate_original_n':int(member.sum()),'candidate_current_retained_n':int((member&(point_values(guard,pts)==0)).sum()),'candidate_new_visible_n':len(vis),'candidate_recovered_n':int((member&(cls=='recovered_outer')).sum()),'candidate_cut_n':int((member&(cls=='artificial_cut')).sum()),'candidate_ambiguous_n':int((member&(cls=='ambiguous')).sum())})
   for k,v in c.items():q['circle_'+k]=v
   for k,v in e.items():q['ellipse_'+k]=v
   q['circle_Dxy_mm']=2*c['radius_px']*float(r.mm_per_px) if c.get('valid') else math.nan;q['ellipse_Dxy_mm']=e['major_diameter_px']*float(r.mm_per_px) if e.get('valid') else math.nan
   out.append(q)
  return base,out,{'p':p,'pts':pts,'cls':cls,'guard':guard,'runs':runs}
 except Exception as e:return {'stem':r.stem,'group':r.group,'capture':r.capture,'segment_id':r.segment_id,'frame_index':r.frame_index,'sharpness':r.sharpness,'ring_px':r.ring_px,'mm_per_px':r.mm_per_px,'image_path':r.image_path,'mask_path':r.mask_path,'ring_json_path':r.ring_json_path,'error':f'{type(e).__name__}:{e}'},[],None
def choose(rows,bases):
    # Reuse frozen V2 score/complexity formula with only the input contour altered.
    by=defaultdict(list)
    for r in rows:
        by[r['group']].append(r)
    chosen=[]
    for _,z in by.items():
        proposals=[]
        for r in z:
            for shape in ('circle','ellipse'):
                if not truth(r.get(f'{shape}_valid',False)):
                    continue
                if shape=='ellipse':
                    if r['ellipse_axis_ratio']<1.15:
                        continue
                    cval=r.get('circle_norm_residual_all',math.inf);eval_=r.get('ellipse_norm_residual_all',math.inf)
                    if truth(r.get('circle_valid',False)) and not (eval_<=.75*cval):
                        continue
                    val=r['ellipse_Dxy_mm'];arc=r['ellipse_arc_coverage'];inl=r['ellipse_inlier_ratio'];res=eval_;pen=.9
                else:
                    val=r['circle_Dxy_mm'];arc=r['circle_arc_coverage'];inl=r['circle_inlier_ratio'];res=r['circle_norm_residual_all'];pen=1.
                proposals.append((r,shape,val,arc,inl,res,pen))
        med=np.median([x[2] for x in proposals]) if proposals else math.nan
        for stem in {r['stem'] for r in z}:
            opts=[]
            for q in proposals:
                if q[0]['stem']!=stem:
                    continue
                _,shape,val,arc,inl,res,pen=q
                score=pen*(arc*inl)/(1+10*res+2*abs(math.log(val/med)))
                opts.append((score,shape,val,q[0]))
            if opts:
                score,shape,val,source=max(opts,key=lambda x:(x[0],x[1]=='circle',x[3]['AB']=='A'))
                a=source.copy();a.update({'valid':True,'chosen_shape':shape,'chosen_Dxy_mm':val,'L_px':val/float(a['mm_per_px']),'chosen_score':score})
            else:
                a=next(r for r in z if r['stem']==stem).copy()
                a.update({'valid':False,'chosen_shape':'none','chosen_Dxy_mm':math.nan,'L_px':math.nan,'chosen_score':math.nan})
            chosen.append(a)
    # Retain missing/failed frozen stems so their bin cannot silently disappear.
    seen={r['stem'] for r in chosen}
    for b in bases:
        if b['stem'] not in seen:
            a=b.copy();a.update({'AB':'none','valid':False,'chosen_shape':'none','chosen_Dxy_mm':math.nan,'L_px':math.nan,'chosen_score':math.nan});chosen.append(a)
    return pd.DataFrame(chosen)
def b1_fuse(sel):
 caps=[]
 for cap,z in sel.groupby('capture'):
  ok=z[z.valid.map(truth)&z.L_px.map(finite)];elig=len(ok)>=4;ring=10. if '_R10_' in z.group.iloc[0] else 5.;caps.append({'method':METHOD,'group':z.group.iloc[0],'capture':cap,'valid_bins':len(ok),'eligible':elig,'prediction_mm':ring*np.median(ok.L_px)/np.median(ok.ring_px) if elig else math.nan,'fallback_reason':'' if elig else 'fewer_than_4_valid_bins'})
 caps=pd.DataFrame(caps);formal=pd.read_csv(AUDIT/'inputs/immutable_scale_manifest.csv');formal['capture']=formal.capture_id.str.replace('video:','',regex=False);formal=formal[formal.current_valid.map(truth)&formal.current_Dxy_mm.map(finite)].copy();rep=caps[caps.eligible].set_index('capture').prediction_mm.to_dict();formal['prediction_value_mm']=formal.current_Dxy_mm;formal['replaced']=formal.source_kind.eq('video_keyframe')&formal.capture.isin(rep);formal.loc[formal.replaced,'prediction_value_mm']=formal.loc[formal.replaced,'capture'].map(rep);g=formal.groupby('group').prediction_value_mm.median().rename('prediction_mm').reset_index();g['method']=METHOD;ip=pd.read_csv(IP88/'tables/group_prediction_blind.csv')[['group','prediction_mm']];ip['method']=METHOD;g=pd.concat([g,ip],ignore_index=True);assert g.group.nunique()==39;return caps,g,formal
def gallery(records,chosen):
 # highest recovered original-edge fraction only; selection never uses GT.
 pick=chosen[chosen.valid.map(truth)].assign(fr=lambda z:z.candidate_recovered_n/z.candidate_original_n).sort_values('fr',ascending=False).drop_duplicates('stem').head(10).stem.tolist();by={x['stem']:(x,a) for x,a,aux in records if aux is not None}
 cards=[]
 for stem in pick:
  aux=next(c for a,b,c in records if a['stem']==stem and c is not None);p=aux['p'];x0,y0,x1,y1=p['crop'];im=cv2.imread(str(chosen[chosen.stem.eq(stem)].image_path.iloc[0]));im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)[y0:y1,x0:x1];pts=aux['pts'];cl=aux['cls'];guard=aux['guard'];comp=aux['runs'][2]['completed'];bridge=aux['runs'][2]['bridge'];fig,ax=plt.subplots(3,4,figsize=(16,12));ax=ax.ravel();mask=np.zeros((*p['polyp'].shape,3),np.uint8);mask[p['polyp'].astype(bool)]=(30,210,70);mask[guard]=(255,165,0)
  panels=[('image',im,'original image'),('mask',mask,'original mask green / guard orange'),('pts',None,'original contour'),('pts',None,'V2 removed by guard'),('pts',None,'V2 retained'),('mask',comp,'diagnostic constrained completion'),('mask',bridge,'accepted bridge (diagnostic only)'),('pts',None,'artificial-cut original edge'),('pts',None,'recovered original outer edge'),('fit','v2','frozen V2 geometry'),('fit','new','recovered-edge V1 geometry')]
  raw=bind_asset_paths(pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv')).set_index('stem').loc[stem]
  new=chosen[chosen.stem.eq(stem)].iloc[0]
  for a,(typ,data,title) in zip(ax,panels):
   a.imshow(im if typ!='mask' else data,cmap='gray' if typ=='mask' else None)
   if typ=='pts':
    a.imshow(im);sel={'original contour':np.ones(len(pts),bool),'V2 removed by guard':point_values(guard,pts)>0,'V2 retained':point_values(guard,pts)==0,'artificial-cut original edge':cl=='artificial_cut','recovered original outer edge':cl=='recovered_outer'}[title];a.scatter(pts[sel,0],pts[sel,1],s=2,c='cyan')
   if typ=='fit':
    a.imshow(im)
    if data=='v2':
     nm,sh=raw.chosen_model_candidate,raw.chosen_shape
     if nm in ('A','B'):
      if sh=='circle':cx,cy=raw[f'{nm}_cx'],raw[f'{nm}_cy'];w=h=2*raw[f'{nm}_radius_px'];ang=0
      else:cx,cy=raw[f'{nm}_ellipse_cx'],raw[f'{nm}_ellipse_cy'];w=raw[f'{nm}_ellipse_major_diameter_px'];h=raw[f'{nm}_ellipse_minor_diameter_px'];ang=raw[f'{nm}_ellipse_angle_deg']
      a.add_patch(Ellipse((cx,cy),w,h,angle=ang,fill=False,color='cyan',lw=2))
    elif new.valid:
     ab=new.AB;sh=new.chosen_shape;pref=sh;a.add_patch(Ellipse((new[f'{pref}_cx'] if sh=='circle' else new[f'{pref}_cx'],new[f'{pref}_cy'] if sh=='circle' else new[f'{pref}_cy']),2*new[f'{pref}_radius_px'] if sh=='circle' else new[f'{pref}_major_diameter_px'],2*new[f'{pref}_radius_px'] if sh=='circle' else new[f'{pref}_minor_diameter_px'],angle=0 if sh=='circle' else new[f'{pref}_angle_deg'],fill=False,color='cyan',lw=2))
   a.set_title(title);a.axis('off')
  ax[-1].axis('off');fig.suptitle(stem);fig.tight_layout();fn=f'{stem}.jpg';fig.savefig(OUT/'gallery'/fn,dpi=120);plt.close(fig);cards.append(fn)
 (OUT/'gallery/index.html').write_text('<meta charset="utf-8"><h1>Ring occlusion boundary recovery (GT-free examples)</h1><p>Completion is diagnostic. Cyan recovered points are original mask contour pixels; no morphology-created contour is fitted.</p>'+''.join(f'<img style="width:100%" src="{x}">' for x in cards))
def blind():
 for d in ('inputs','tables','gallery'):(OUT/d).mkdir(parents=True,exist_ok=True)
 protocol={'baseline':'S6-NDG-FCP39','GT_used_before_evaluation':False,'scales_over_rr':SCALES,'multi_scale_consensus':CONSENSUS,'completion':'closing candidate bridge restricted to U and retained only if it contacts >=2 substantial original components','fit_points':'original INIT polyp contour only; morphology-created edges forbidden','unchanged':'V2 circle/ellipse fitter and selection semantics, S6 selected stems, B1 no min dynamic gate, formal fusion, Ip8 extension'};(OUT/'inputs/frozen_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 sel=selected();write(sel,'tables/frozen_s6_selected_stems_blind.csv');records=[];rows=[];audit=[]
 for i,r in enumerate(sel.itertuples(index=False),1):
  base,ab,aux=process(r);records.append((base,ab,aux));rows+=ab
  if aux:
   for run in aux['runs']:
    audit.append({'stem':r.stem,'group':r.group,'capture':r.capture,'scale_over_rr':run['fac'],'bridge_px':run['bridge_px'],'raw_closed_added_px':run['raw_closed_added_px'],'outside_U_added_px':run['outside_U_added_px'],'accepted_bridge_components':sum(x['accepted'] for x in run['components'])})
  if i%50==0:print(i,flush=True)
 cand=pd.DataFrame(rows);chosen=choose(rows,[x[0] for x in records]);write(cand,'tables/candidate_boundary_audit_blind.csv');write(chosen,'tables/recovered_geometry_selected_blind.csv');write(audit,'tables/completion_multiscale_audit_blind.csv')
 caps,g,integ=b1_fuse(chosen);write(caps,'tables/capture_predictions_blind.csv');write(g,'tables/group_predictions_blind.csv');write(integ,'tables/integrated_rows_blind.csv')
 stats=[]
 for _,z in chosen.groupby('AB'):
  pass
 ok=chosen[chosen.valid.map(truth)];stats.append({'method':METHOD,'frames':len(chosen),'valid_frames':len(ok),'median_original_contour_n':ok.original_contour_n.median(),'median_v2_retained_fraction':(ok.candidate_current_retained_n/ok.candidate_original_n).median(),'median_recovered_fraction':(ok.candidate_recovered_n/ok.candidate_original_n).median(),'median_cut_fraction':(ok.candidate_cut_n/ok.candidate_original_n).median(),'median_ambiguous_fraction':(ok.candidate_ambiguous_n/ok.candidate_original_n).median(),'eligible_captures':int(caps.eligible.sum()),'fallback_captures':int((~caps.eligible).sum())});write(stats,'tables/boundary_summary_blind.csv')
 files=['inputs/frozen_protocol.json','tables/frozen_s6_selected_stems_blind.csv','tables/candidate_boundary_audit_blind.csv','tables/recovered_geometry_selected_blind.csv','tables/completion_multiscale_audit_blind.csv','tables/capture_predictions_blind.csv','tables/group_predictions_blind.csv','tables/boundary_summary_blind.csv'];(OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/x)}  {x}\n' for x in files));gallery(records,chosen);print('sealed')
def metric(z,k):
 z=z.dropna(subset=['prediction_mm']);e=100*(z.prediction_mm-z.reference_mm)/z.reference_mm;a=e.abs();return {'method':k,'coverage':len(z),'median_abs_error_pct':a.median(),'MAE_mm':(z.prediction_mm-z.reference_mm).abs().mean(),'within5_pct':100*(a<=5).mean(),'within10_pct':100*(a<=10).mean(),'p95_pct':a.quantile(.95),'max_pct':a.max()}
def evaluate():
 for l in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():h,r=l.split('  ',1);assert sha(OUT/r)==h
 refs=pd.read_csv(MAIN/'tables/group_errors_descending.csv')[['group','reference_mm']];new=pd.read_csv(OUT/'tables/group_predictions_blind.csv');base=pd.read_csv(S6/'tables/group_predictions_blind.csv');base=base[base.method.eq('S6_NDG_FCP39')];base['method']='S6_frozen';x=pd.concat([base,new],ignore_index=True).merge(refs,on='group',how='left');x['abs_error_pct']=100*(x.prediction_mm-x.reference_mm).abs()/x.reference_mm;write(x,'tables/group_evaluated.csv');m=pd.DataFrame([metric(z,k) for k,z in x.groupby('method')]);write(m,'tables/metrics_summary.csv');b=x[x.method.eq('S6_frozen')].set_index('group');pairs=[]
 for r in x[x.method.eq(METHOD)].itertuples(index=False):q=b.loc[r.group];pairs.append({'group':r.group,'baseline_prediction_mm':q.prediction_mm,'candidate_prediction_mm':r.prediction_mm,'baseline_abs_error_pct':q.abs_error_pct,'candidate_abs_error_pct':r.abs_error_pct,'delta_abs_error_pp':r.abs_error_pct-q.abs_error_pct})
 write(pairs,'tables/group_paired.csv');print(m.to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['blind','evaluate']);a=p.parse_args();{'blind':blind,'evaluate':evaluate}[a.stage]()
