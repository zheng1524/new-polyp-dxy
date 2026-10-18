"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import cv2 as cv2
import math as math
import numpy as np
RULE = {'version': 'ring_enclosed_small_background_v1', 'area_fraction_max': 0.025, 'local_width_area_factor': 0.5, 'width_reference_quantile': 0.9, 'max_hole_span_width_factor': 1.5, 'central_protection_area_radius_factor': 0.35, 'connectivity': 8, 'background_label': 0, 'ring_label': 2, 'polyp_label': 1, 'closing_used': False, 'polyp_mask_changed': False}

def run(labels):
 ring=(labels==2);result=labels.copy();fill=np.zeros(labels.shape,bool)
 ys,xs=np.nonzero(ring);area=int(len(xs))
 if not area:return result,fill,dict(rule=RULE,ring_area_px=0,filled_area_px=0,changed=False,holes=[])
 distance=cv2.distanceTransform(ring.astype(np.uint8),cv2.DIST_L2,5)
 width=2*float(np.quantile(distance[ring],RULE['width_reference_quantile']))
 limit=min(RULE['area_fraction_max']*area,RULE['local_width_area_factor']*width**2)
 centroid=np.array([xs.mean(),ys.mean()]);protected_radius=RULE['central_protection_area_radius_factor']*math.sqrt(area/math.pi)
 # Complement includes all non-ring labels, so polyp-containing center holes
 # and open occlusion gaps cannot be silently converted into material.
 n,cc,stats,centers=cv2.connectedComponentsWithStats((~ring).astype(np.uint8),8)
 external=set(np.r_[cc[0,:],cc[-1,:],cc[:,0],cc[:,-1]].tolist());holes=[]
 for i in range(1,n):
  if i in external:continue
  x,y,w,h,a=map(int,stats[i]);pixels=cc==i;reason=''
  if not np.all(labels[pixels]==0):reason='contains_polyp_or_nonbackground_preserve'
  elif a>limit:reason='area_above_small_hole_limit_preserve_center'
  elif max(w,h)>RULE['max_hole_span_width_factor']*width:reason='elongated_or_large_span_preserve'
  elif np.linalg.norm(centers[i]-centroid)<protected_radius:reason='near_ring_material_centroid_preserve_center'
  boundary=cv2.dilate(pixels.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&~pixels
  if not reason and not np.all(ring[boundary]):reason='not_completely_enclosed_by_actual_material'
  accepted=not reason
  if accepted:fill|=pixels
  holes.append(dict(component=i,area_px=a,bbox=[x,y,w,h],centroid=centers[i].tolist(),filled=accepted,reason=reason or 'small_closed_background_surrounded_by_ring'))
 result[fill]=2
 assert np.array_equal(result==1,labels==1)
 assert np.array_equal(result[~fill],labels[~fill])
 assert not np.any(fill&(labels!=0))
 if not fill.any():assert np.array_equal(result,labels)
 return result,fill,dict(rule=RULE,ring_area_px=area,ring_width_reference_px=width,area_limit_px=limit,central_protection_radius_px=protected_radius,ring_centroid=centroid.tolist(),filled_area_px=int(fill.sum()),changed=bool(fill.any()),holes=holes)


# Bind relocated dependencies after defining functions (including override hooks).
