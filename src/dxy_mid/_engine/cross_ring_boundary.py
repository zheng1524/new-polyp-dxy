"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.spatial._ckdtree import cKDTree as cKDTree
import copy as copy
import cv2 as cv2
import numpy as np
RULE = {'version': 'V8.5-cross-ring-Q-1.0', 'normal_alignment_min': 0.65, 'contact_outward_steps_px': [1.0, 2.0, 3.0], 'anchor_exit_distance_min_px': 3.0, 'anchor_exit_distance_rr': 0.035, 'minimum_material_width_px': 3.0, 'maximum_material_width_rr': 0.85, 'background_component_area_min_px': 25, 'background_component_area_min_rr2': 0.05, 'background_seed_erosion_px': 1, 'highlight_gray_threshold': 245, 'minimum_nonhighlight_background_seed_pixels': 9, 'in_max_px': 1.0, 'in_width_fraction': 0.05, 'out_fractions': [0.1, 0.2, 0.35], 'P_frozen': True, 'existing_Q_intervals_frozen': True, 'endpoint_crop_fraction': 0.0, 'Q_anchor': 'original_polyp_contour', 'fit_solver_changed': False}

def run(frame,ring,base,raw,evidence):
 e=copy.deepcopy(evidence);d=e['original_boundary'];frozen_intervals={}
 for z in d['records']:
  if z['class_']=='Q':frozen_intervals[z['global_index']]=dict(n=np.array(z['n']),lo=-z['delta_in_px'],hi_by_fraction=z['delta_out_px'])
 for z in evidence['untrimmed_polyp_contact_Q']:frozen_intervals[z['global_index']]=dict(n=z['n'],lo=z['lo'],hi_by_fraction=z['hi_by_fraction'])
 pol=frame.labels==1;rm=frame.labels==2;bg=frame.labels==0;rr=raw['rr']
 _,pcc=cv2.connectedComponents(pol.astype(np.uint8),8)
 _,bcc,bstats,_=cv2.connectedComponentsWithStats(bg.astype(np.uint8),8)
 gray=cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY)
 seeds=cv2.erode(bg.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&(gray<RULE['highlight_gray_threshold'])
 seed_counts=np.bincount(bcc[seeds],minlength=len(bstats));minimum_area=max(25,.05*rr*rr)
 samples=r.RING_SIDES(frame,ring,base,raw,d)['ring_samples'];cross=[]
 h,w=pol.shape
 def label(q):
  x,y=np.rint(q).astype(int)
  return int(frame.labels[y,x]) if 0<=x<w and 0<=y<h else -1
 for s in samples:
  s.pop('q',None);s['cross_ring_verified']=False
  if not all(k in s for k in ['exit_plus','exit_minus','normal_raw','width_px']):continue
  exits=[np.array(s['exit_plus']),np.array(s['exit_minus'])];labs=[label(q) for q in exits];s['nearest_nonring_labels']=labs
  width=s['width_px']
  if not max(3.,.015*rr)<=width<=.85*rr:s['cross_reason']='actual_material_width_invalid';continue
  if sorted(labs)!=[0,1]:s['cross_reason']='nearest_other_side_polyp_or_ambiguous';continue
  pi=labs.index(1);bi=1-pi;pe=exits[pi];be=exits[bi];x,y=np.rint(be).astype(int);bc=int(bcc[y,x]);x,y=np.rint(pe).astype(int);pc=int(pcc[y,x])
  # The nearest exit must itself be background. No skipping an intervening
  # polyp strip to reach a farther background patch as the legacy probe could.
  s.update(background_component_id=bc,background_area_px=int(bstats[bc,cv2.CC_STAT_AREA]),observable_background_seed_pixels=int(seed_counts[bc]),polyp_component_id=pc,polyp_exit=pe.tolist(),background_exit=be.tolist())
  if bc==0 or bstats[bc,cv2.CC_STAT_AREA]<minimum_area or seed_counts[bc]<9:s['cross_reason']='other_side_has_no_connected_observable_background_or_highlight_hole';continue
  n=(be-pe)/width;s.update(n=n.tolist(),cross_ring_verified=True,cross_reason='nearest_background_exit_connected_to_observed_background',kind='PB')
  cross.append(s)
 tree=cKDTree([s['polyp_exit'] for s in cross]) if cross else None
 Pids={z['global_index'] for z in d['records'] if z['class_']=='P'};Pcoords={tuple(z['q']) for z in d['records'] if z['class_']=='P'};checks=[];Q=[]
 for z in d['records']:
  if z['global_index'] in Pids:continue
  oldclass=z['class_'];z['class_']='X';q=np.array(z['q'],float);x,y=np.rint(q).astype(int)
  ck=dict(global_index=z['global_index'],contour_id=z['contour_id'],contour_index=z['contour_index'],q=q.tolist(),old_class=oldclass,new_class='X',reason='',semantic='Q_constraint',source_kind='original_polyp_contour_cross_ring',Q_is_observed_P=False)
  if z['parent_contour']>=0:ck['reason']='internal_polyp_hole'
  elif x==0 or y==0 or x==w-1 or y==h-1:ck['reason']='frame_border_truncation'
  elif tuple(z['q']) in Pcoords:ck['reason']='same_pixel_already_has_frozen_observed_P'
  elif tree is None:ck['reason']='no_verified_cross_ring_background'
  else:
   n,why,metrics=r.boundary_v7.normal(d['contours'][z['contour_id']],z['contour_index'],pol);ck.update(metrics)
   if n is None:ck['reason']=why
   else:
    matches=tree.query_ball_point(q,max(3.,.035*rr));matches=[i for i in matches if cross[i]['polyp_component_id']==int(pcc[y,x]) and n@np.array(cross[i]['n'])>=.65]
    ray=np.rint(q+np.array([1.,2.,3.])[:,None]*n).astype(int);ray=np.clip(ray,[0,0],[w-1,h-1])
    if not rm[ray[:,1],ray[:,0]].any():ck['reason']='no_actual_ring_contact_in_outward_1_to_3px'
    elif not matches:ck['reason']='no_same_cross_section_polyp_exit_normal_match'
    else:
     s=min((cross[i] for i in matches),key=lambda a:(np.linalg.norm(q-np.array(a['polyp_exit'])),a['source_index']));width=s['width_px']
     ck.update(new_class='Q',reason='original_contour_cross_ring_background_confirmed',n=n.tolist(),delta_in_px=min(1.,.05*width),delta_out_px={str(f):f*width for f in [.1,.2,.35]},ring_sample_index=s['source_index'],ring_width_px=width,polyp_exit=s['polyp_exit'],background_exit=s['background_exit'],background_component_id=s['background_component_id'],background_seed_pixels=s['observable_background_seed_pixels'],normal_source='original_polyp_contour',anchor_source='original_polyp_contour',background_used_as_anchor=False)
     if z['global_index'] in frozen_intervals:
      old=frozen_intervals[z['global_index']];assert np.allclose(old['n'],n);ck.update(delta_in_px=-old['lo'],delta_out_px=copy.deepcopy(old['hi_by_fraction']),interval_parameter_source='same_current_mask_frozen_existing_Q')
     else:ck['interval_parameter_source']='newly_confirmed_current_ring_width'
     z.update(class_='Q',n=n.tolist(),delta_in_px=ck['delta_in_px'],delta_out_px=ck['delta_out_px'])
     Q.append(dict(q=q,n=n,lo=-ck['delta_in_px'],hi_by_fraction=ck['delta_out_px'],arc_id=0,global_index=z['global_index'],contour_id=z['contour_id'],contour_index=z['contour_index'],body_support=['A','B'],ring_width_px=width,source_kind='original_polyp_contour_cross_ring',source_index=s['source_index'],normal_source='original_polyp_contour',polyp_exit=s['polyp_exit'],background_exit=s['background_exit'],semantic='Q_constraint',is_observed_P=False,is_postfit_inference=False))
  z['cross_ring_check']=ck;z['reason']=ck['reason'];checks.append(ck)
 by={(q['contour_id'],q['contour_index']):q for q in Q};arcs=[]
 for cid,c in enumerate(d['contours']):
  states=np.array(['Q' if (cid,i) in by else 'X' for i in range(len(c))])
  for arc in r.boundary_v7.runs(states,'Q',cid):
   aid=len(arcs);gl=[]
   for i in arc['indices']:by[(cid,i)]['arc_id']=aid;gl.append(by[(cid,i)]['global_index'])
   arcs.append(dict(arc,arc_id=aid,global_indices=gl))
 d['Q_arcs']=arcs # Public raw classifications also carry Q: bbox/center sees P union Q.
 e.update(version=RULE['version'],rules=RULE,ring_samples=samples,anchor_checks=checks,polyp_contact_Q=[],source_Q_constraints=Q,new_Q_arcs=arcs,untrimmed_polyp_contact_Q=copy.deepcopy(Q),untrimmed_new_Q_arcs=copy.deepcopy(arcs),trimmed_Q_points=[],endpoint_trim_audit=[],cross_ring_background_seeds=seeds,cross_ring_bg_component_labels=bcc)
 result=copy.deepcopy(base);result.uncertain=np.array([q['q'] for q in Q]).reshape(-1,2);Qids={q['global_index'] for q in Q};result.excluded=np.array([z['q'] for z in d['records'] if z['class_']=='X']).reshape(-1,2)
 assert Pids=={z['global_index'] for z in d['records'] if z['class_']=='P'}
 assert np.array_equal(result.trusted,base.trusted)
 result.provenance.update(Boundary_experiment=RULE['version'],P_frozen=True,Q_anchor='original_polyp_contour',Q_crop=0)
 return result,e


def make_data(frame,base,raw,evidence,body,fraction):
 data=ORIGINAL_DATA_BUILDER(frame,base,raw,evidence,body,fraction)
 sources={q['global_index']:q for q in evidence.get('source_Q_constraints',[])}
 for q in data['Q']:
  if q['global_index'] in sources:
   metadata=copy.deepcopy(sources[q['global_index']]);metadata.pop('hi_by_fraction',None);metadata.pop('arc_id',None);q.update(metadata)
 if not next(c for c in base.candidates if c['name']==body)['mask'].any():
  pts=np.vstack([data['P'],np.array([q['q'] for q in data['Q']]).reshape(-1,2)])
  if len(pts):
   expected=(pts.min(0)+pts.max(0))/2;assert np.allclose(data['center'],expected),'Fallback center dropped Q constraints'
 return data


# Bind relocated dependencies after defining functions (including override hooks).
from . import run_native as r
from .v84_boundary import make_data84 as ORIGINAL_DATA_BUILDER
