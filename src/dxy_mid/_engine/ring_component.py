"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from skimage.morphology._skeletonize import medial_axis as medial_axis
import cv2 as cv2
import math as math
import numpy as np
VERSION = 'RING_VISIBLE_ARC_CENTERLINE_v2.0'

def ring_points(mask,e,image):
 cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE);parts=[];tangent=[]
 for c in cs:
  p=c.reshape(-1,2).astype(float)
  if len(p)<10:continue
  k=min(5,(len(p)-1)//2);v=np.roll(p,-k,axis=0)-np.roll(p,k,axis=0);parts.append(p);tangent.append(v/np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-8))
 if not parts:return np.empty((0,2)),np.empty((0,2)),np.empty((0,2))
 p=np.concatenate(parts);t=np.concatenate(tangent);q=coords(p,e);angle=np.deg2rad(e['angle_deg']);grad=q/[e['major_radius']**2,e['minor_radius']**2];v=np.column_stack([grad[:,0]*np.cos(angle)-grad[:,1]*np.sin(angle),grad[:,0]*np.sin(angle)+grad[:,1]*np.cos(angle)]);v/=np.maximum(np.linalg.norm(v,axis=1)[:,None],1e-8)
 # Ring-side orientation distinguishes concave inner arcs from convex outer arcs,
 # even when neither contour is topologically closed. Cut ends are near-radial.
 alignment=np.abs((t*np.column_stack([-v[:,1],v[:,0]])).sum(1));rho=np.sqrt((q[:,0]/e['major_radius'])**2+(q[:,1]/e['minor_radius'])**2)
 hop=max(3.,max(mask.shape)/256);h,w=mask.shape
 def sample(m,points):
  x=np.clip(np.rint(points[:,0]).astype(int),0,w-1);y=np.clip(np.rint(points[:,1]).astype(int),0,h-1);return m[y,x]
 plus=sample(mask,p+hop*v);minus=sample(mask,p-hop*v)
 gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float) if image.ndim==3 else image.astype(float)
 # Light support on either non-material side; blurred, weak-contrast samples
 # are rejected rather than treating every segmentation transition as an edge.
 contrast_in=sample(gray,p-hop*v)-sample(gray,p+hop*v);contrast_out=sample(gray,p+hop*v)-sample(gray,p-hop*v)
 inner=(alignment>=.60)&plus&~minus&(rho>.30)&(rho<.96)&(contrast_in>=5)
 outer=(alignment>=.60)&minus&~plus&(rho>.78)&(rho<1.18)&(contrast_out>=5)
 return p[inner],p[outer],p[~(inner|outer)]


def validate(mask,inner,outer,D,innerpoints=None):
 diag={};reasons=[]
 if inner is None or outer is None:return False,{'reason':'inner_or_outer_fit_missing'}
 ai,bi,ao,bo=[x for e in (inner,outer) for x in (e['major_radius'],e['minor_radius'])]
 if min(ai,bi,ao,bo)<=0 or ai>=ao or bi>=bo:return False,{'reason':'inner_outer_axes_invalid'}
 amid=(ai+ao)/2;bmid=(bi+bo)/2;pxmm=2*amid/(D+1.2);gap=math.hypot(inner['cx']-outer['cx'],inner['cy']-outer['cy'])/amid
 majorwidth=(ao-ai)/pxmm;minorwidth=(bo-bi)/pxmm
 angle=abs((inner['angle_deg']-outer['angle_deg']+90)%180-90)
 diag.update(center_gap_norm=gap,wire_major_mm=majorwidth,wire_minor_mm=minorwidth,outer_inner_major_ratio=ao/ai,angle_gap_deg=angle,centerline_frontality=bmid/amid)
 if gap>.12:reasons.append('inner_outer_center_inconsistent')
 if not .6<=majorwidth<=1.8 or not .42<=minorwidth<=1.98:reasons.append('wire_thickness_inconsistent_1p2mm')
 if angle>18 and min(ai/bi,ao/bo)>1.15:reasons.append('inner_outer_orientation_inconsistent')
 if bmid/amid<.35:reasons.append('ring_too_oblique')
 ann=fill(mask.shape,outer)&~fill(mask.shape,inner);rec=(ann&mask).sum()/max(mask.sum(),1);prec=(ann&mask).sum()/max(ann.sum(),1);diag.update(mask_band_recall=float(rec),mask_band_precision=float(prec))
 if rec<.55 or prec<.50:reasons.append('annular_mask_support_insufficient')
 if innerpoints is not None:
  cov=float(coverage(innerpoints,inner));res=float(np.median(ellipse_residuals(ellipse_from_dict(inner),innerpoints.astype(np.float32)))/inner['minor_radius']);stab=stability(innerpoints,inner)
  diag.update(inner_arc_coverage=cov,inner_normalized_residual=res,inner_sector_instability=stab)
  if cov<.30:reasons.append('visible_inner_arc_insufficient')
  if res>.08:reasons.append('inner_arc_residual_high')
  if stab>.20:reasons.append('inner_fit_unstable')
 diag['reason']=';'.join(reasons) if reasons else 'ok';return not reasons,diag


def recover(labels,image,D):
 h,w=labels.shape;region=(labels==2);ys,xs=np.nonzero(region)
 base={'valid':False,'version':VERSION,'coordinate_system':'native_image_px','radius_unit':'native_px_radius','diameter_unit':'native_px_diameter','wire_diameter_mm':1.2,'reference_inner_diameter_mm':float(D),'history_prediction_used':False,'attempts':[]}
 payload=dict(inner_points=np.empty((0,2)),outer_points=np.empty((0,2)),rejected_points=np.empty((0,2)))
 if len(xs)<300:base['reason']='ring_mask_insufficient';return base,payload
 # Crop preserves native coordinates; do not downsample the mask before ring fitting.
 pad=15;x0=max(0,int(xs.min())-pad);x1=min(w,int(xs.max())+pad+1);y0=max(0,int(ys.min())-pad);y1=min(h,int(ys.max())+pad+1)
 m=preprocess_ring_mask(region[y0:y1,x0:x1]);im=image[y0:y1,x0:x1];comps=select_components(m,1)
 if not comps:base['reason']='ring_component_missing';return base,payload
 m=comps[0];native=fit_single_component(m);outer=native.get('outer_ellipse');payload['native_ring']=native;payload['crop']=(x0,y0,x1,y1)
 if outer is None:base['reason']='outer_fit_missing';return base,payload
 innerpts,outerpts,rejected=ring_points(m,outer,im)
 payload.update(inner_points=innerpts+[x0,y0],outer_points=outerpts+[x0,y0],rejected_points=rejected+[x0,y0])
 candidateouter=fit(outerpts)
 if candidateouter is None or len(outerpts)<25:
  base['reason']='observable_outer_edge_insufficient';return base,payload
 if coverage(outerpts,candidateouter)<.35 or stability(outerpts,candidateouter)>.20:
  base['reason']='outer_edge_fit_unstable';return base,payload
 if candidateouter is not None:
  if stability(outerpts,candidateouter)<=.20 and coverage(outerpts,candidateouter)>=.35:outer=candidateouter
 inner=fit(innerpts);ok,diag=validate(m,inner,outer,D,innerpts if inner is not None else None);base['attempts'].append(dict(branch='visible_inner_arc_RANSAC',valid=ok,**diag))
 branch='visible_inner_arc_RANSAC'
 if not ok:
  # Same centerline symmetry as historical Ip8.8 recovery. Evidence-triggered,
  # no group input and no old fitted JSON or old measurement value.
  sk=medial_axis(m,rng=42);yy,xx=np.nonzero(sk);p=np.column_stack([xx,yy]).astype(np.float32)
  # Remove skeleton branches of local endpoints/occluders before fitting.
  q=coords(p,outer);rho=np.sqrt((q[:,0]/outer['major_radius'])**2+(q[:,1]/outer['minor_radius'])**2);p=p[(rho>.45)&(rho<.97)]
  mid=fit(p);reconstructed=None
  if mid is not None:
   reconstructed={k:float(2*mid[k]-outer[k]) for k in ['cx','cy','major_radius','minor_radius']};reconstructed['angle_deg']=mid['angle_deg']
  ok,diag=validate(m,reconstructed,outer,D)
  diag.update(mid_points=len(p),mid_arc_coverage=float(coverage(p,mid)) if mid else 0,mid_sector_instability=float(stability(p,mid)) if mid else math.inf)
  if not mid or diag['mid_arc_coverage']<.50 or diag['mid_sector_instability']>.15:ok=False;diag['reason']=diag.get('reason','')+';centerline_evidence_unstable'
  if ok and len(innerpts)>=25:
   residual=float(np.median(ellipse_residuals(ellipse_from_dict(reconstructed),innerpts.astype(np.float32)))/max(reconstructed['minor_radius'],1));diag['recovered_inner_visible_residual']=residual
   if residual>.10:ok=False;diag['reason']='recovered_inner_disagrees_visible_arc'
  base['attempts'].append(dict(branch='centerline_symmetry_2mid_minus_outer',valid=ok,**diag));branch='centerline_symmetry_2mid_minus_outer';inner=reconstructed;payload['mid_points']=p+[x0,y0]
 if not ok:base.update(reason='ring_recovery_unreliable',diagnostics=diag);return base,payload
 def global_e(e):return dict(e,cx=float(e['cx']+x0),cy=float(e['cy']+y0))
 inner=global_e(inner);outer=global_e(outer)
 # A round wire has concentric projected centerline; averaging inner/outer
 # radii yields the ring plane, unlike inner-only or area-weighted transforms.
 centerline={k:(inner[k]+outer[k])/2 for k in ['cx','cy','major_radius','minor_radius']};z=np.exp(2j*np.deg2rad([inner['angle_deg'],outer['angle_deg']]));centerline['angle_deg']=float(np.rad2deg(np.angle(z.mean()))/2)
 T,_=build_affine_rectifier(ellipse_from_dict(centerline),None);epts=transform_points(sample_ellipse_points(ellipse_from_dict(centerline)),T);diam=2*np.mean(np.linalg.norm(epts-epts.mean(0),axis=1));scale=(D+1.2)/diam
 base.update(valid=True,reason='ok',branch=branch,inner=inner,outer=outer,centerline=centerline,diagnostics=diag,transform_native_to_ringplane=T.tolist(),reference_kind='centerline',reference_diameter_mm=float(D+1.2),reference_diameter_rectified_px=float(diam),mm_per_rectified_px=float(scale))
 return base,payload


def coverage(pts,e):
 q=coords(pts,e);t=np.mod(np.arctan2(q[:,1]/e['minor_radius'],q[:,0]/e['major_radius']),2*np.pi);return np.histogram(t,36,(0,2*np.pi))[0].astype(bool).mean() if len(pts) else 0


def stability(points,e):
 if len(points)<25:return math.inf
 q=coords(points,e);ang=np.mod(np.arctan2(q[:,1]/e['minor_radius'],q[:,0]/e['major_radius']),2*np.pi);change=[]
 for start in np.linspace(0,2*np.pi,4,endpoint=False):
  sub=points[np.mod(ang-start,2*np.pi)>np.pi/3]
  if len(sub)<25:return math.inf
  try:
   (cx,cy),(d1,d2),_=cv2.fitEllipseDirect(sub.astype(np.float32).reshape(-1,1,2));a,b=max(d1,d2)/2,min(d1,d2)/2
   change.append(max(abs(a/e['major_radius']-1),abs(b/e['minor_radius']-1),math.hypot(cx-e['cx'],cy-e['cy'])/e['major_radius']))
  except cv2.error:return math.inf
 return max(change)


def coords(pts,e):
 t=np.deg2rad(e['angle_deg']);q=pts-[e['cx'],e['cy']];return np.column_stack([q[:,0]*np.cos(t)+q[:,1]*np.sin(t),-q[:,0]*np.sin(t)+q[:,1]*np.cos(t)])


def fill(shape,e):
 m=np.zeros(shape,np.uint8);cv2.fillPoly(m,[np.round(sample_ellipse_points(ellipse_from_dict(e))).astype(np.int32)],1);return m.astype(bool)


def fit(points):
 if len(points)<25:return None
 e,_,_=fit_ellipse_ransac(points.astype(np.float32),iterations=200)
 return e.as_dict() if e is not None else None


# Bind relocated dependencies after defining functions (including override hooks).
from .polypseg_ring_fitting import ellipse_residuals as ellipse_residuals
from .polypseg_ring_fitting import fit_ellipse_ransac as fit_ellipse_ransac
from .polypseg_ring_fitting import fit_single_component as fit_single_component
from .polypseg_ring_fitting import preprocess_ring_mask as preprocess_ring_mask
from .polypseg_ring_fitting import select_components as select_components
from .polypseg_single_view_measurement import build_affine_rectifier as build_affine_rectifier
from .polypseg_single_view_measurement import ellipse_from_dict as ellipse_from_dict
from .polypseg_single_view_measurement import sample_ellipse_points as sample_ellipse_points
from .polypseg_single_view_measurement import transform_points as transform_points
