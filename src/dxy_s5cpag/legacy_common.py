"""Frozen 20260805 INIT inputs and GT-blind Dxy ablation helpers."""
from pathlib import Path
from collections import defaultdict
import csv,math,statistics,json
import numpy as np
HERE=Path(__file__).resolve().parents[1]
# `frozen()` is retained only for the historical V2 command-line entrypoint.
# Supply its data root explicitly through DXY_DATA_ROOT when that entrypoint is
# used; S5-CPAG runtime imports only n/good/med/write.
ROOT=Path(__import__('os').environ.get('DXY_DATA_ROOT', HERE.parents[2]))
PREV=ROOT/'data'/'dxy_remeasurement_20260917'
def read(path):
 with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:r.get(k,'') for k in fields} for r in rows])
def n(x):
 try:return float(x)
 except:return math.nan
def good(x):return math.isfinite(n(x)) and n(x)>0
def med(xs):return statistics.median(xs) if xs else math.nan
def mean(xs):return statistics.mean(xs) if xs else math.nan
def std(xs):return statistics.pstdev(xs) if len(xs)>1 else (0.0 if xs else math.nan)
def weighted_median(values,weights):
 pairs=sorted((v,w) for v,w in zip(values,weights) if good(v) and good(w))
 if not pairs:return math.nan
 total=sum(w for _,w in pairs);cum=0
 for v,w in pairs:
  cum+=w
  if cum>=total/2:return v
 return pairs[-1][0]
def build_frozen():
 views=read(PREV/'03_ring_detection_and_scale/expanded_full_new_per_view.csv')
 rect={r['stem']:r for r in read(PREV/'verification/old_rectified_scale_same_image_per_view.csv') if r['arm']=='new'}
 multi={r['stem']:r for r in read(PREV/'verification/ring_occlusion_per_view_ablation.csv') if r['arm']=='new'}
 manifest={r['stem']:r for r in read(PREV/'01_inputs/expanded_manifest.csv')}
 routes={r['group_key']:r for r in read(PREV/'04_classifier_router/expanded_full_new_group_routes.csv')}
 gt={r['group_key']:r for r in read(PREV/'01_inputs/blender_gt_mapping.csv')}
 assert len(views)==len(manifest)==515 and len(routes)==len(gt)==39
 out=[]
 for v in views:
  stem=v['stem'];m=manifest[stem];rr=rect.get(stem,{});mm=multi.get(stem,{});route=routes[v['group_key']];g=gt[v['group_key']]
  source=m['source_kind'];cap='photo:'+stem if source=='photo' else 'video:'+Path(m['source_video']).stem
  rec={**v,'source_kind':source,'source_video':m['source_video'],'capture_id':cap,'source_frame_index':m['source_frame_index'],'rect_scale':rr.get('old_rectified_mm_per_px_same_ring',''),'rect_ring_diameter_px':rr.get('old_rectified_diameter_px',''),'multi_accepted':mm.get('accepted','False'),'multi_feret_px':mm.get('chosen_feret_px','') if mm.get('accepted')=='True' else v.get('feret_px',''),'route':route['routed_branch'],'mean_ip_probability':route['mean_ip_probability'],'nominal_mm':g['nominal_diameter_mm'],'gt_mm':g['Dxy_gt_mm'],'gt_source':g['gt_source'],'image_width':m['width'],'image_height':m['height'],'image_sha256':m['image_sha256']}
  out.append(rec)
 return out
ROWS=None
def frozen():
 global ROWS
 if ROWS is None:
  p=HERE/'baseline/frozen_per_view.csv'
  ROWS=read(p) if p.exists() else build_frozen()
 return ROWS
def eligible_base(r):return r['status']=='ok' and good(r['rect_scale'])
def eligible_pre(r):return (r['status']=='ok' or r['invalid_reason']=='scale_outlier_vs_group_median') and good(r['rect_scale'])
def gate(rows,variant):
 by=defaultdict(list)
 for r in rows:by[r['group_key']].append(r)
 keep=set()
 for key,group in by.items():
  pre=[r for r in group if eligible_pre(r)]
  if variant=='B0':
   keep.update(r['stem'] for r in group if eligible_base(r));continue
  if variant=='B1':
   center=med([n(r['rect_scale']) for r in pre])
   keep.update(r['stem'] for r in pre if good(center) and abs(n(r['rect_scale'])-center)/center<=.25)
  elif variant=='B2':
   caps=defaultdict(list)
   for r in pre:caps[r['capture_id']].append(r)
   for capture,items in caps.items():
    if capture.startswith('photo:'):keep.update(r['stem'] for r in items)
    else:
     center=med([n(r['rect_scale']) for r in items])
     keep.update(r['stem'] for r in items if good(center) and abs(n(r['rect_scale'])-center)/center<=.25)
  elif variant=='B3_MAD':
   vals=[n(r['rect_scale']) for r in pre];center=med(vals)
   if not good(center):continue
   mad=med([abs(x-center) for x in vals]);sigma=max(1.4826*mad,.05*center)
   keep.update(r['stem'] for r in pre if abs(n(r['rect_scale'])-center)<=3.5*sigma)
  elif variant=='B3_IQR':
   vals=np.asarray([n(r['rect_scale']) for r in pre],float)
   if len(vals)==0:continue
   q1,q3=np.quantile(vals,[.25,.75]);iqr=max(q3-q1,.05*float(np.median(vals)))
   lo,hi=q1-1.5*iqr,q3+1.5*iqr
   keep.update(r['stem'] for r in pre if lo<=n(r['rect_scale'])<=hi)
  else:raise ValueError(variant)
 return keep
def candidate_px(r,branch,geometry=None):
 if branch=='Ip':
  col=geometry or 'ring_parallel_px'
  return n(r.get(col))
 if branch=='nonIp':
  col=geometry or 'multi_feret_px'
  return n(r.get(col))
 return math.nan
def candidate_mm(r,branch,geometry=None):
 px=candidate_px(r,branch,geometry);scale=n(r['rect_scale'])
 return px*scale if good(px) and good(scale) else math.nan
def quality_weight(r,group):
 q=n(r['ring_fit_quality']);q=max(q,0) if math.isfinite(q) else 10
 ecc=n(r['inner_aspect_ratio']);ecc=max(ecc,1) if math.isfinite(ecc) else 3
 scales=[n(x['rect_scale']) for x in group if good(x['rect_scale'])]
 center=med(scales);dev=abs(n(r['rect_scale'])-center)/center if good(center) else 1
 areas=[n(x['polyp_pixels']) for x in group if good(x['polyp_pixels'])]
 area=med(areas);adev=abs(math.log(n(r['polyp_pixels'])/area)) if good(area) and good(r['polyp_pixels']) else 1
 return 1/(.5+q)/(1+2*(ecc-1))/(1+4*dev)/(1+adev)
def fuse(view_rows,branch,method='C0',geometry=None):
 vals=[(r,candidate_mm(r,branch,geometry)) for r in view_rows]
 vals=[(r,x) for r,x in vals if good(x)]
 if not vals:return math.nan,'',0,math.nan,math.nan
 rr=[r for r,_ in vals];xx=[x for _,x in vals]
 if method=='C0':
  if branch=='Ip':return med(xx),'',len(xx),std(xx),float(np.subtract(*np.percentile(xx,[75,25])))
  chosen=min(vals,key=lambda pair:n(pair[0]['ring_fit_quality']) if math.isfinite(n(pair[0]['ring_fit_quality'])) else 9999)
  return chosen[1],chosen[0]['stem'],len(xx),std(xx),float(np.subtract(*np.percentile(xx,[75,25])))
 if method=='C1':y=med(xx)
 elif method=='C2':
  sortedx=sorted(xx);trim=max(1,int(.1*len(sortedx))) if len(sortedx)>=5 else 0
  y=mean(sortedx[trim:len(sortedx)-trim])
 elif method=='C3':y=weighted_median(xx,[quality_weight(r,rr) for r in rr])
 elif method=='C4':
  captures=defaultdict(list)
  for r,x in vals:captures[r['capture_id']].append(x)
  y=med([med(z) for z in captures.values()])
 else:raise ValueError(method)
 return y,'',len(xx),std(xx),float(np.subtract(*np.percentile(xx,[75,25])))
def predict(rows,variant='B0',fusion='C0',policy='type_locked',ip_geometry=None,nonip_geometry=None):
 selected=gate(rows,variant);by=defaultdict(list)
 for r in rows:by[r['group_key']].append(r)
 out=[]
 for key,group in sorted(by.items()):
  typ=group[0]['paris_type'];branch=('Ip' if typ=='Ip' else 'nonIp') if policy=='type_locked' else group[0]['route']
  used=[r for r in group if r['stem'] in selected]
  if branch=='ambiguous':pred,stem,nvalid,sd,iqr=math.nan,'',0,math.nan,math.nan
  else:pred,stem,nvalid,sd,iqr=fuse(used,branch,fusion,ip_geometry if branch=='Ip' else nonip_geometry)
  nom=n(group[0]['nominal_mm']);gt=n(group[0]['gt_mm']);source=group[0]['gt_source']
  out.append({'group_key':key,'paris_type':typ,'policy':policy,'routed_branch':group[0]['route'],'used_branch':branch,'gate':variant,'fusion':fusion,'ip_geometry':ip_geometry or 'ring_parallel_px','nonip_geometry':nonip_geometry or 'multi_feret_px','n_views_total':len(group),'n_views_gate':len(used),'n_candidate_views':nvalid,'selected_stem':stem,'within_group_sd_mm':sd if math.isfinite(sd) else '', 'within_group_iqr_mm':iqr if math.isfinite(iqr) else '', 'pred_mm':pred if good(pred) else '', 'nominal_mm':nom,'gt_mm':gt if good(gt) else '', 'gt_source':source,'abs_pct_nominal':abs(pred-nom)/nom*100 if good(pred) else '', 'abs_pct_same_name_gt':abs(pred-gt)/gt*100 if good(pred) and good(gt) and source=='exact_prior_group_Dxy_gt' else ''})
 return out
def metrics(group_rows,method,scope='nominal_all'):
 sub=[r for r in group_rows if scope=='nominal_all' or r['gt_source']=='exact_prior_group_Dxy_gt']
 field='abs_pct_nominal' if scope=='nominal_all' else 'abs_pct_same_name_gt'
 errs=[n(r[field]) for r in sub if math.isfinite(n(r[field]))]
 aes=[abs(n(r['pred_mm'])-n(r['nominal_mm'] if scope=='nominal_all' else r['gt_mm'])) for r in sub if good(r['pred_mm']) and good(r['nominal_mm'] if scope=='nominal_all' else r['gt_mm'])]
 rec={'method':method,'scope':scope,'n_total':len(sub),'n_valid':len(errs),'median_relative_error':med(errs),'mean_relative_error':mean(errs),'MAE_mm':mean(aes),'within5_pct':sum(e<=5 for e in errs)/len(errs)*100 if errs else '', 'within10_pct':sum(e<=10 for e in errs)/len(errs)*100 if errs else ''}
 for typ in ('Ip','Is','IIa'):
  z=[n(r[field]) for r in sub if r['paris_type']==typ and math.isfinite(n(r[field]))]
  rec[f'{typ}_n_valid']=len(z);rec[f'{typ}_median']=med(z)
 return rec
def paired(base,cand):
 b={r['group_key']:r for r in base};c={r['group_key']:r for r in cand};out=[]
 for key in sorted(b):
  a=b[key];z=c[key];e0=n(a['abs_pct_nominal']);e1=n(z['abs_pct_nominal'])
  out.append({'group_key':key,'paris_type':a['paris_type'],'baseline_pred_mm':a['pred_mm'],'candidate_pred_mm':z['pred_mm'],'baseline_error_pct':e0 if math.isfinite(e0) else '', 'candidate_error_pct':e1 if math.isfinite(e1) else '', 'delta_error_pct':e1-e0 if math.isfinite(e0) and math.isfinite(e1) else '', 'baseline_valid_views':a['n_candidate_views'],'candidate_valid_views':z['n_candidate_views'],'baseline_sd_mm':a['within_group_sd_mm'],'candidate_sd_mm':z['within_group_sd_mm']})
 return out
