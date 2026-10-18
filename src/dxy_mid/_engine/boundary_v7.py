"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.spatial._ckdtree import cKDTree as cKDTree
import cv2 as cv2
import numpy as np
PARAM = {'out_fractions': [0.1, 0.2, 0.35], 'default_out_fraction': 0.2, 'in_max_px': 1.0, 'in_width_fraction': 0.05, 'normal_window': 5, 'normal_stability_cos': 0.8, 'ring_alignment_cos': 0.65, 'side_dominance': 0.5}
VERSION7 = 'V7-phase1-polyp-anchor-1.0'

def runs(states,label,cid):
 sel=states==label
 if not sel.any():return []
 if sel.all():return [dict(contour_id=cid,start_index=0,indices=list(range(len(sel))),wraps=True)]
 out=[]
 for st in np.flatnonzero(sel&~np.roll(sel,1)):
  ids=[];j=int(st)
  while sel[j]:ids.append(j);j=(j+1)%len(sel)
  out.append(dict(contour_id=cid,start_index=int(st),indices=ids,wraps=any(b<a for a,b in zip(ids,ids[1:]))))
 return out


def normal(c,i,polyp):
 if len(c)<15:return None,'normal_short_contour',{}
 vectors=[]
 for k in [3,5,9]:
  tangent=c[(i+k)%len(c)].astype(float)-c[(i-k)%len(c)]
  if np.linalg.norm(tangent)<1:return None,'normal_degenerate',{}
  n=np.array([-tangent[1],tangent[0]])/np.linalg.norm(tangent);vectors.append(n)
 n=vectors[1]
 if min(abs(n@v) for v in vectors)<PARAM['normal_stability_cos']:return None,'normal_unstable_corner',{}
 q=c[i];a=RAW.frac(polyp,q+n*3,1);b=RAW.frac(polyp,q-n*3,1)
 if a>b:n=-n;a,b=b,a
 if a>.25 or b<.60:return None,'normal_orientation_ambiguous',dict(outside_polyp_fraction=a,inside_polyp_fraction=b)
 return n,'',dict(outside_polyp_fraction=a,inside_polyp_fraction=b)


def extract(frame,ring,base,raw):
 """Return six-component BoundaryResult and exhaustive native pixel index record."""
 if not ring.valid:raise ValueError('current_ring_invalid')
 polyp=frame.labels==1;rm=frame.labels==2;bg=frame.labels==0;rr=raw['rr']
 contours,hierarchy=cv2.findContours(polyp.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
 contours=[c.reshape(-1,2) for c in contours];hierarchy=hierarchy[0] if hierarchy is not None else np.empty((0,4),int)
 trusted=set(map(tuple,np.rint(base.trusted).astype(int)))
 RP,RN,RK,RS=RAW.ring_samples(rm,polyp,rr)
 tree=cKDTree(RP) if len(RP) else None
 # V2 body carrier before visibility gate: keep its descriptors, without fitting.
 carrier=np.zeros(polyp.shape,np.uint8);x0,y0,x1,y1=raw['crop']
 for desc,cm,*rest in raw['components']:
  if desc['broad_descriptor']:carrier[y0:y1,x0:x1]|=cm
 for ca in base.candidates:carrier|=ca['mask']
 _,polyp_components=cv2.connectedComponents(polyp.astype(np.uint8),8)
 matched_trust=[q for q in trusted if polyp[q[1],q[0]]]
 if matched_trust:
  body_ids={int(polyp_components[q[1],q[0]]) for q in matched_trust}
  carrier_source='original_polyp_components_with_exact_V2_trusted_pixels'
 else:
  body_ids=set(map(int,np.unique(polyp_components[carrier.astype(bool)])))-{0}
  carrier_source='V2_broad_carrier_before_visible_gate_zero_P'
 yy,xx=np.nonzero(carrier);ctree=cKDTree(np.c_[xx,yy]) if len(xx) else None
 records=[];arcs=[];normal_valid=0
 for cid,c in enumerate(contours):
  states=[]
  for i,q in enumerate(c):
   rec=dict(global_index=len(records),contour_id=cid,contour_index=i,q=q.tolist(),class_='X',reason='',parent_contour=int(hierarchy[cid,3]))
   if hierarchy[cid,3]>=0:rec['reason']='internal_polyp_hole'
   elif tuple(q) in trusted:rec.update(class_='P',reason='original_pixel_in_exact_V2_trusted')
   elif not base.unknown_mask[q[1],q[0]]:rec['reason']='not_in_V2_selected_body_or_sample'
   elif int(polyp_components[q[1],q[0]]) not in body_ids:rec['reason']='original_component_not_supported_by_V2_body'
   elif ctree is None or ctree.query(q)[0]>max(3,.16*rr):rec['reason']='not_near_V2_body_carrier'
   elif tree is None:rec['reason']='no_ring_adjacency_evidence'
   else:
    distance,j=tree.query(q);rec.update(nearest_ring_distance_px=float(distance),ring_adjacency=str(RK[j]))
    if distance>max(3,.12*rr):rec['reason']='guard_only_no_actual_ring_contact'
    elif RK[j]=='cut':rec['reason']='ring_both_sides_polyp'
    elif RK[j]!='outer':rec['reason']='ring_adjacency_ambiguous'
    else:
     p=RP[j];pn=RN[j]*RS[j];pe=RAW.beyond(rm,p,pn,max(4,round(RAW.MAX_CROSS*rr)));be=RAW.beyond(rm,p,-pn,max(4,round(RAW.MAX_CROSS*rr)))
     if pe is None or be is None:rec['reason']='ring_crossing_incomplete'
     else:
      forward=max(1,round(RAW.SAMPLE_FORWARD*rr));radius=max(1,round(RAW.SAMPLE_RADIUS*rr))
      pf=RAW.frac(polyp,pe+pn*forward,radius);bf=RAW.frac(bg,be-pn*forward,radius);rf=RAW.frac(rm,be-pn*forward,radius)
      rec.update(polyp_side_fraction=pf,background_side_fraction=bf,background_side_ring_fraction=rf)
      n,why,metrics=normal(c,i,polyp);rec.update(metrics)
      w=float(np.linalg.norm(be-pe));out_ring=(be-pe)/max(w,1e-9)
      if pf<.5 or bf<.5 or rf>.2:rec['reason']='adjacency_not_observed_polyp_background'
      elif n is None:rec['reason']=why
      elif float(n@out_ring)<PARAM['ring_alignment_cos']:rec['reason']='polyp_normal_ring_crossing_disagreement'
      elif w<max(3,.015*rr) or w>.85*rr:rec['reason']='ring_mask_thickness_unreliable'
      else:
       # Actual ring endpoints are solely a thickness reference, never Q anchors.
       normal_valid+=1
       rec.update(class_='Q',reason='original_polyp_contour_contact_verified',n=n.tolist(),local_ring_thickness_px=w,delta_in_px=min(1.,.05*w),delta_out_px={str(f):f*w for f in PARAM['out_fractions']},ring_width_evidence=dict(ring_point=p.tolist(),polyp_exit=pe.tolist(),background_exit=be.tolist()),normal_ring_alignment=float(n@out_ring))
   records.append(rec);states.append(rec['class_'])
  arcs+=runs(np.array(states),'Q',cid)
 pts=np.array([r['q'] for r in records],int).reshape(-1,2);states=np.array([r['class_'] for r in records]);prov=dict(base.provenance,method='V7_original_polyp_contour_directional_intervals',implementation_version=VERSION7,length_unit='px',Q_anchor='original_polyp_mask_contour_pixel')
 # A/B retained unchanged. New geometry consumers must explicitly use indexed Q.
 result=BoundaryResult(base.candidates,pts[states=='P'],pts[states=='X'],pts[states=='Q'],base.unknown_mask,prov)
 details=dict(version=VERSION7,provenance=prov,params=PARAM,contours=contours,hierarchy=hierarchy,records=records,Q_arcs=arcs,V2_original_trusted=base.trusted,normal_valid=normal_valid,body_component_ids=sorted(body_ids),body_carrier_source=carrier_source)
 return result,details


# Bind relocated dependencies after defining functions (including override hooks).
from . import raw_g3_helpers as RAW
from .contracts import BoundaryResult as BoundaryResult
