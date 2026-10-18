"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from skimage.morphology._skeletonize import skeletonize as skeletonize
import cv2 as cv2
import hashlib as hashlib
import math as math
import numpy as np


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


def circle_three(p):
 a,b,c=p.astype(float);A=2*np.vstack([b-a,c-a]);B=np.array([b@b-a@a,c@c-a@a])
 try:center=np.linalg.solve(A,B)
 except np.linalg.LinAlgError:return None
 return center,float(np.linalg.norm(center-a))


def disk(r):
 r=max(1,int(round(r)))
 return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))


def contour(mask):
 cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
 return np.concatenate([c.reshape(-1,2) for c in cs]).astype(np.float32) if cs else np.empty((0,2),np.float32)


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


def ellipse_mask(shape,e):
 out=np.zeros(shape,np.uint8)
 cv2.ellipse(out,(round(e['cx']),round(e['cy'])),(max(1,round(e['major_radius'])),max(1,round(e['minor_radius']))),e['angle_deg'],0,360,1,-1)
 return out


def angular_bins(points,center,nbins=72):
 if len(points)==0:return set()
 angles=np.mod(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]),2*np.pi)
 return set(np.floor(angles/(2*np.pi)*nbins).astype(int).tolist())


def bin_overlap(a,b,margin=3,nbins=72):
 if not a or not b:return 0
 aa=set((x+j)%nbins for x in a for j in range(-margin,margin+1))
 return len(aa&b)


def span_degrees(bins,nbins=72):
 if not bins:return 0.
 a=sorted(bins);gaps=[(a[(i+1)%len(a)]-a[i])%nbins for i in range(len(a))]
 return 360*(1-max(gaps)/nbins)


# Bind relocated dependencies after defining functions (including override hooks).
