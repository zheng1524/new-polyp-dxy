"""Ring-censored, multi-component circle fitting. No GT or Paris input in geometry."""
from pathlib import Path
from collections import defaultdict
import math, json, hashlib, argparse
import cv2, numpy as np
from skimage.morphology import skeletonize
from .legacy_common import frozen,gate,write,n,good,med
HERE=Path(__file__).resolve().parents[1]

def disk(r):
 r=max(1,int(round(r)))
 return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def contour(mask):
 cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 return np.concatenate([c.reshape(-1,2) for c in cs]).astype(np.float32) if cs else np.empty((0,2),np.float32)
def ellipse_mask(shape,e):
 out=np.zeros(shape,np.uint8)
 cv2.ellipse(out,(round(e['cx']),round(e['cy'])),(max(1,round(e['major_radius'])),max(1,round(e['minor_radius']))),e['angle_deg'],0,360,1,-1)
 return out
def angular_bins(points,center,nbins=72):
 if len(points)==0:return set()
 angles=np.mod(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]),2*np.pi)
 return set(np.floor(angles/(2*np.pi)*nbins).astype(int).tolist())
def span_degrees(bins,nbins=72):
 if not bins:return 0.
 a=sorted(bins);gaps=[(a[(i+1)%len(a)]-a[i])%nbins for i in range(len(a))]
 return 360*(1-max(gaps)/nbins)
def bin_overlap(a,b,margin=3,nbins=72):
 if not a or not b:return 0
 aa=set((x+j)%nbins for x in a for j in range(-margin,margin+1))
 return len(aa&b)
def circle_three(p):
 a,b,c=p.astype(float);A=2*np.vstack([b-a,c-a]);B=np.array([b@b-a@a,c@c-a@a])
 try:center=np.linalg.solve(A,B)
 except np.linalg.LinAlgError:return None
 return center,float(np.linalg.norm(center-a))
def fit_circle(points,geom_center,extent,rr,seed):
 """Stage-specific reason plus raw quality for auditing; no reference diameter."""
 if len(points)==0:return {'valid':False,'reason':'no_visible_contour','n_visible':0}
 if len(points)<20:return {'valid':False,'reason':'too_few_visible_points','n_visible':len(points)}
 pts=points.astype(float);rng=np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8],16));th=max(2.,.04*rr)
 count={'triplets':0,'center_pass':0,'radius_pass':0,'inlier_pass':0};best=None
 minr=max(.15*rr,.08*extent);maxr=max(3*rr,1.6*extent)
 for _ in range(180):
  m=circle_three(pts[rng.choice(len(pts),3,replace=False)])
  if m is None:continue
  count['triplets']+=1;center,R=m
  if np.linalg.norm(center-geom_center)>1.25*extent:continue
  count['center_pass']+=1
  if not minr<=R<=maxr:continue
  count['radius_pass']+=1
  res=np.abs(np.linalg.norm(pts-center,axis=1)-R);inside=res<=th
  if inside.sum()<max(10,int(.30*len(pts))):continue
  count['inlier_pass']+=1
  score=(int(inside.sum()),-float(np.median(res[inside])))
  if best is None or score>best[0]:best=(score,inside)
 if best is None:
  reason='circle_center_constraint' if count['center_pass']==0 else ('circle_radius_constraint' if count['radius_pass']==0 else 'ransac_inlier_ratio_fail')
  return {'valid':False,'reason':reason,'n_visible':len(pts),**count}
 sub=pts[best[1]];A=np.column_stack([2*sub[:,0],2*sub[:,1],np.ones(len(sub))]);b=(sub**2).sum(axis=1)
 par,*_=np.linalg.lstsq(A,b,rcond=None);center=par[:2];R=math.sqrt(max(0,par[2]+center@center))
 res=np.abs(np.linalg.norm(sub-center,axis=1)-R)
 res_all=np.abs(np.linalg.norm(pts-center,axis=1)-R)
 ang=np.mod(np.arctan2(sub[:,1]-center[1],sub[:,0]-center[0]),2*np.pi)
 coverage=np.histogram(ang,bins=36,range=(0,2*np.pi))[0].astype(bool).mean();norm=float(np.median(res)/R) if R else math.inf
 reason='ok'
 if np.linalg.norm(center-geom_center)>1.25*extent:reason='circle_center_constraint'
 elif not minr<=R<=maxr:reason='circle_radius_constraint'
 elif coverage<.20:reason='insufficient_arc_coverage'
 elif norm>.08:reason='circle_residual_fail'
 return {'valid':reason=='ok','reason':reason,'n_visible':len(pts),'cx':float(center[0]),'cy':float(center[1]),'radius_px':float(R),'norm_residual':norm,'norm_residual_all':float(np.median(res_all)/R) if R else math.inf,'arc_coverage':float(coverage),'inlier_ratio':float(len(sub)/len(pts)),**count}

def fit_ellipse(points,geom_center,extent,rr):
 if len(points)==0:return {'valid':False,'reason':'no_visible_contour'}
 if len(points)<25:return {'valid':False,'reason':'too_few_visible_points'}
 pts=points.astype(np.float32);current=pts
 try:
  for _ in range(4):
   (cx,cy),(d1,d2),angle=cv2.fitEllipseAMS(current.reshape(-1,1,2))
   if d1<=0 or d2<=0:return {'valid':False,'reason':'ellipse_invalid_axes'}
   theta=math.radians(angle);c,s=math.cos(theta),math.sin(theta);q=pts-np.array([cx,cy]);xx=q[:,0]*c+q[:,1]*s;yy=-q[:,0]*s+q[:,1]*c
   residual=np.abs(np.sqrt((xx/(d1/2))**2+(yy/(d2/2))**2)-1)
   current=pts[residual<=np.quantile(residual,.8)]
   if len(current)<20:return {'valid':False,'reason':'too_few_robust_points'}
  (cx,cy),(d1,d2),angle=cv2.fitEllipseAMS(current.reshape(-1,1,2))
 except cv2.error:return {'valid':False,'reason':'ellipse_numeric_fail'}
 major,minor=max(d1,d2),min(d1,d2);ratio=major/max(minor,1e-6)
 theta=math.radians(angle);c,s=math.cos(theta),math.sin(theta)
 def coords(x):
  q=x-np.array([cx,cy]);return q[:,0]*c+q[:,1]*s,-q[:,0]*s+q[:,1]*c
 xx,yy=coords(current);allx,ally=coords(pts)
 residual=np.abs(np.sqrt((xx/(d1/2))**2+(yy/(d2/2))**2)-1)
 residual_all=np.abs(np.sqrt((allx/(d1/2))**2+(ally/(d2/2))**2)-1)
 angles=np.mod(np.arctan2(yy/(d2/2),xx/(d1/2)),2*np.pi)
 arc=np.histogram(angles,bins=36,range=(0,2*np.pi))[0].astype(bool).mean()
 reason='ok'
 if np.linalg.norm(np.array([cx,cy])-geom_center)>1.25*extent:reason='ellipse_center_constraint'
 elif minor/2<max(.15*rr,.08*extent) or major/2>max(3*rr,1.6*extent):reason='ellipse_radius_constraint'
 elif ratio>3.5:reason='ellipse_axis_ratio_fail'
 elif arc<.20:reason='insufficient_arc_coverage'
 elif float(np.median(residual))>.08:reason='ellipse_residual_fail'
 return {'valid':reason=='ok','reason':reason,'cx':float(cx),'cy':float(cy),'major_diameter_px':float(major),'minor_diameter_px':float(minor),'angle_deg':float(angle),'axis_ratio':float(ratio),'norm_residual':float(np.median(residual)),'norm_residual_all':float(np.median(residual_all)),'arc_coverage':float(arc),'inlier_ratio':len(current)/len(pts)}

def component_records(polyp,inside,outside,band,center,rr):
 evidence=(polyp&(~band.astype(bool))).astype(np.uint8)
 num,labels,stats,cents=cv2.connectedComponentsWithStats(evidence,8)
 data=[];graph=[]
 for i in range(1,num):
  area=int(stats[i,cv2.CC_STAT_AREA]);cm=(labels==i).astype(np.uint8)
  if area<max(12,int(.003*rr*rr)):continue
  pts=contour(cm);region='inside' if (cm&inside).sum()>(cm&outside).sum() else 'outside'
  dt=cv2.distanceTransform(cm,cv2.DIST_L2,cv2.DIST_MASK_PRECISE);sk=skeletonize(cm>0)
  widths=2*dt[sk];thick=widths[widths>0]
  hull=cv2.convexHull(pts).reshape(-1,2) if len(pts)>=3 else pts
  hullarea=cv2.contourArea(hull.reshape(-1,1,2)) if len(hull)>=3 else area
  rect=cv2.minAreaRect(pts.reshape(-1,1,2)) if len(pts)>=3 else ((0,0),(0,0),0)
  dims=sorted(rect[1]);elong=dims[1]/max(dims[0],1)
  near=(cv2.dilate(cm,disk(max(2,.06*rr)))&band).astype(np.uint8)
  ys,xs=np.nonzero(near);bins=angular_bins(np.column_stack([xs,ys]),center)
  contact_guard=cv2.dilate(band,disk(max(2,.06*rr)))
  contact_pts=pts[contact_guard[np.clip(pts[:,1].astype(int),0,band.shape[0]-1),np.clip(pts[:,0].astype(int),0,band.shape[1]-1)]>0]
  visible=pts[contact_guard[np.clip(pts[:,1].astype(int),0,band.shape[0]-1),np.clip(pts[:,0].astype(int),0,band.shape[1]-1)]==0]
  descriptor={'component_id':i,'region':region,'area_px':area,'normalized_area':area/(rr*rr),'centroid_x':float(cents[i,0]),'centroid_y':float(cents[i,1]),'max_thickness_px':float(thick.max()) if len(thick) else 0.,'median_skeleton_thickness_px':float(np.median(thick)) if len(thick) else 0.,'normalized_thickness':float(np.median(thick)/rr) if len(thick) else 0.,'skeleton_length_px':int(sk.sum()),'solidity':area/max(hullarea,1),'elongation':elong,'ring_contact_length_px':len(contact_pts),'ring_contact_area_px':int(near.sum()),'ring_contact_angular_span_deg':span_degrees(bins),'ring_contact_angle_center_deg':float(np.degrees(np.angle(np.mean(np.exp(1j*np.array(list(bins))*2*np.pi/72)))))%360 if bins else math.nan,'visible_contour_points':len(visible),'broad_descriptor':bool((float(np.median(thick)/rr) if len(thick) else 0)>=.30 and area/(rr*rr)>=.08)}
  data.append((descriptor,cm,bins,pts,visible))
 for a in range(len(data)):
  for b in range(a+1,len(data)):
   da,db=data[a][0],data[b][0]
   if da['region']==db['region']:continue
   ov=bin_overlap(data[a][2],data[b][2]);dist=float(np.linalg.norm(np.array([da['centroid_x'],da['centroid_y']])-np.array([db['centroid_x'],db['centroid_y']]))/rr)
   if ov>0 and dist<=4.0:graph.append({'component_a':da['component_id'],'component_b':db['component_id'],'angular_overlap_bins':ov,'centroid_distance_over_ring_radius':dist})
 return data,graph

def candidate_fit(mask,band,center,rr,stem,name):
 # The candidate was cut at the unknown band boundary. A second guard keeps
 # that artificial cut edge out of the physical contour fit.
 guard=cv2.dilate(band,disk(max(2,.06*rr)))
 pts=contour(mask);visible=pts[~guard[np.clip(pts[:,1].astype(int),0,guard.shape[0]-1),np.clip(pts[:,0].astype(int),0,guard.shape[1]-1)].astype(bool)] if len(pts) else pts
 removed=len(pts)-len(visible);ys,xs=np.nonzero(mask)
 if len(xs):geom=np.array([float(xs.mean()),float(ys.mean())]);extent=float(np.hypot(xs.max()-xs.min(),ys.max()-ys.min()))
 else:geom=center;extent=rr
 if len(visible)>2500:visible=visible[np.linspace(0,len(visible)-1,2500).astype(int)]
 fit=fit_circle(visible,geom,extent,rr,stem+'_'+name)
 ellipse=fit_ellipse(visible,geom,extent,rr)
 return fit,ellipse,visible,removed

def process_view(r):
 original=cv2.imread(r['mask_path'],cv2.IMREAD_UNCHANGED)
 if original is None:raise FileNotFoundError(r['mask_path'])
 with open(r['ring_fit_json']) as f:j=json.load(f)
 rings=j.get('rings',[])
 if not rings:return {'stem':r['stem'],'group_key':r['group_key'],'status':'missing_ring_fit'},[],[],None
 ringinfo=rings[0];inner=ringinfo['inner_ellipse'];outer=ringinfo['outer_ellipse']
 yy,xx=np.nonzero((original==1)|(original==2));pad=40
 x0=max(0,int(xx.min())-pad);x1=min(original.shape[1],int(xx.max())+pad+1);y0=max(0,int(yy.min())-pad);y1=min(original.shape[0],int(yy.max())+pad+1)
 polyp=(original[y0:y1,x0:x1]==1).astype(np.uint8);ring=(original[y0:y1,x0:x1]==2).astype(np.uint8)
 def local(e):return {**e,'cx':e['cx']-x0,'cy':e['cy']-y0}
 inner=local(inner);outer=local(outer);center=np.array([inner['cx'],inner['cy']]);rr=math.sqrt(inner['major_radius']*inner['minor_radius'])
 innerfill=ellipse_mask(polyp.shape,inner);outerfill=ellipse_mask(polyp.shape,outer)
 inside=innerfill.astype(np.uint8);outside=(1-outerfill).astype(np.uint8)
 band=cv2.dilate(ring,disk(max(3,.08*rr))).astype(np.uint8)
 band=np.maximum(band,((1-inside)&(1-outside)).astype(np.uint8))
 comps,edges=component_records(polyp,inside,outside,band,center,rr)
 descriptors=[{'stem':r['stem'],'group_key':r['group_key'],**d} for d,*_ in comps]
 graph=[{'stem':r['stem'],'group_key':r['group_key'],**edge} for edge in edges]
 # A keeps broad outside body evidence when present. Otherwise it retains
 # all substantial evidence, including inside pieces, without a seed choice.
 broad_out=[(d,cm) for d,cm,*_ in comps if d['region']=='outside' and d['broad_descriptor'] and d['visible_contour_points']>=12]
 all_substantial=[(d,cm) for d,cm,*_ in comps if d['visible_contour_points']>=12]
 A=np.zeros_like(polyp)
 for d,cm in (broad_out if broad_out else all_substantial):A=np.maximum(A,cm)
 allmask=np.zeros_like(polyp)
 for d,cm in all_substantial:allmask=np.maximum(allmask,cm)
 # B removes thin branches at a ring-normalized, fixed physical image scale;
 # all surviving components remain, without a seed-component selection.
 B=cv2.morphologyEx(allmask,cv2.MORPH_OPEN,disk(max(2,.12*rr)))
 fitA,ellA,visA,removedA=candidate_fit(A,band,center,rr,r['stem'],'A')
 fitB,ellB,visB,removedB=candidate_fit(B,band,center,rr,r['stem'],'B')
 scale=n(r['rect_scale'])
 result={'stem':r['stem'],'group_key':r['group_key'],'status':'ok','image_path':r['image_path'],'mask_path':r['mask_path'],'ring_fit_json':r['ring_fit_json'],'crop_x0':x0,'crop_y0':y0,'crop_x1':x1,'crop_y1':y1,'ring_cx':center[0],'ring_cy':center[1],'ring_radius_equiv_px':rr,'rectified_mm_per_px':scale,'n_components':len(comps),'n_graph_edges':len(edges),'n_inside_components':sum(d['region']=='inside' for d,*_ in comps),'n_outside_components':sum(d['region']=='outside' for d,*_ in comps),'n_broad_outside':sum(d['region']=='outside' and d['broad_descriptor'] for d,*_ in comps),'candidate_A_source':'broad_outside' if broad_out else 'all_substantial','candidate_A_area_px':int(A.sum()),'candidate_B_area_px':int(B.sum()),'candidate_A_visible_points':len(visA),'candidate_B_visible_points':len(visB),'candidate_A_ring_contour_removed':removedA,'candidate_B_ring_contour_removed':removedB}
 for name,fit in [('A',fitA),('B',fitB)]:
  for k,v in fit.items():result[f'{name}_{k}']=v
  result[f'{name}_Dxy_mm']=2*fit['radius_px']*scale if fit['valid'] else math.nan
 for name,ell in [('A',ellA),('B',ellB)]:
  for k,v in ell.items():result[f'{name}_ellipse_{k}']=v
  result[f'{name}_ellipse_Dxy_mm']=ell['major_diameter_px']*scale if ell['valid'] else math.nan
 payload={'polyp':polyp,'ring':ring,'inside':inside,'outside':outside,'band':band,'components':comps,'edges':edges,'A':A,'B':B,'visA':visA,'visB':visB,'fitA':fitA,'fitB':fitB,'ellA':ellA,'ellB':ellB,'crop':(x0,y0,x1,y1),'rr':rr,'center':center}
 return result,descriptors,graph,payload

def choose(group):
 # Candidate choice relies only on fit quality and consistency of candidate
 # radii within this capture group. No labels or reference sizes are read.
 allmm=[n(r[k]) for r in group for k in ('A_Dxy_mm','B_Dxy_mm') if good(r[k])]
 center=med(allmm)
 for r in group:
  scored=[]
  for name in ('A','B'):
   if not r[f'{name}_valid']:continue
   cov=n(r[f'{name}_arc_coverage']);inlier=n(r[f'{name}_inlier_ratio']);res=n(r[f'{name}_norm_residual']);val=n(r[f'{name}_Dxy_mm'])
   consistency=abs(math.log(val/center)) if good(center) else 0
   score=(cov*inlier)/(1+10*res+2*consistency)
   scored.append((score,name))
  if not scored:r.update({'chosen_candidate':'none','chosen_Dxy_mm':math.nan,'chosen_reason':'both_fits_failed'});continue
  score,name=max(scored,key=lambda x:(x[0],x[1]=='A'))
  r.update({'chosen_candidate':name,'chosen_Dxy_mm':r[f'{name}_Dxy_mm'],'chosen_score':score,'chosen_reason':'quality_and_group_consistency'})

def choose_geometry(group):
 # A circle is preferred unless ellipse fit reduces whole-visible-contour
 # normalized residual by >=25% and has axis ratio >=1.15. This fixed
 # complexity hurdle uses only geometry; it applies to every Paris type.
 proposals=[]
 for r in group:
  for name in ('A','B'):
   c=r[f'{name}_valid'];e=r[f'{name}_ellipse_valid']
   er=n(r[f'{name}_ellipse_norm_residual_all']) if e else math.inf
   cr=n(r[f'{name}_norm_residual_all']) if c else math.inf
   if e and (not c or (n(r[f'{name}_ellipse_axis_ratio'])>=1.15 and er<=.75*cr)):
    shape='ellipse';value=n(r[f'{name}_ellipse_Dxy_mm']);res=er;arc=n(r[f'{name}_ellipse_arc_coverage']);inlier=n(r[f'{name}_ellipse_inlier_ratio'])
   elif c:shape='circle';value=n(r[f'{name}_Dxy_mm']);res=cr;arc=n(r[f'{name}_arc_coverage']);inlier=n(r[f'{name}_inlier_ratio'])
   else:continue
   proposals.append((r,name,shape,value,res,arc,inlier))
 center=med([p[3] for p in proposals if good(p[3])])
 for r in group:
  choices=[]
  for p in proposals:
   if p[0] is not r:continue
   _,name,shape,value,res,arc,inlier=p
   consistency=abs(math.log(value/center)) if good(center) else 0
   score=(arc*inlier)/(1+10*res+2*consistency)
   if shape=='ellipse':score*=.9
   choices.append((score,name,shape,value))
  if not choices:r.update({'chosen_model_candidate':'none','chosen_shape':'none','chosen_model_Dxy_mm':math.nan,'chosen_model_reason':'all_geometries_failed'});continue
  score,name,shape,value=max(choices,key=lambda x:(x[0],x[1]=='A'))
  r.update({'chosen_model_candidate':name,'chosen_shape':shape,'chosen_model_Dxy_mm':value,'chosen_model_score':score,'chosen_model_reason':'fit_quality_and_group_consistency'})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int);args=p.parse_args()
 rows=frozen();keep=gate(rows,'B3_MAD');selected=[r for r in rows if r['stem'] in keep]
 if args.limit:selected=selected[:args.limit]
 out=[];desc=[];edges=[]
 for i,r in enumerate(selected,1):
  z,d,e,_=process_view(r);out.append(z);desc.extend(d);edges.extend(e)
  if i%25==0:print(i,'/',len(selected),flush=True)
 by=defaultdict(list)
 for r in out:by[r['group_key']].append(r)
 for group in by.values():choose(group);choose_geometry(group)
 write(HERE/'frames/per_view.csv',out);write(HERE/'components/descriptors.csv',desc);write(HERE/'components/graph_edges.csv',edges)
 print('done',len(out),'A circles',sum(r.get('A_valid',False) for r in out),'B circles',sum(r.get('B_valid',False) for r in out),'A ellipses',sum(r.get('A_ellipse_valid',False) for r in out),'B ellipses',sum(r.get('B_ellipse_valid',False) for r in out),'circle chosen',sum(good(r.get('chosen_Dxy_mm')) for r in out),'model chosen',sum(good(r.get('chosen_model_Dxy_mm')) for r in out))
