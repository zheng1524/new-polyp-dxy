"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import copy as copy
import cv2 as cv2
import numpy as np
RULE = {'version': 'consecutive-Q-geometric-arclength-v1', 'contour_perimeter_fraction': 0.02, 'minimum_arclength_px': 3.0, 'contour_mode': 'RETR_TREE_CHAIN_APPROX_NONE', 'closed_index_wrap': True, 'length': 'sum_native_pixel_edge_lengths; full-Q contour closes', 'threshold_scope': 'each original source contour perimeter', 'removed_label': 'X', 'GT_used': False}

def apply(base,evidence,expected_removed):
 e=copy.deepcopy(evidence);d=e['original_boundary'];Pids=[z['global_index'] for z in d['records'] if z['class_']=='P'];Q=e['source_Q_constraints']
 removed,arcs=analyze(d['contours'],Q);assert removed==set(expected_removed)
 keep=[q for q in Q if q['global_index'] not in removed];by={(q['contour_id'],q['contour_index']):q for q in keep};kept_arcs=[]
 for z in d['records']:
  if z['global_index'] in removed:
   assert z['class_']=='Q';z.update(class_='X',reason='short_Q_arc_removed_whole_run',Q_removed_before_solver=True)
 for cid,c in enumerate(d['contours']):
  states=np.array(['Q' if (cid,i) in by else 'X' for i in range(len(c))])
  for arc in r.boundary_v7.runs(states,'Q',cid):
   aid=len(kept_arcs);gids=[]
   for i in arc['indices']:by[(cid,i)]['arc_id']=aid;gids.append(by[(cid,i)]['global_index'])
   kept_arcs.append(dict(arc,arc_id=aid,global_indices=gids))
 d['Q_arcs']=kept_arcs
 e.update(source_Q_constraints=keep,polyp_contact_Q=[],new_Q_arcs=kept_arcs,
          untrimmed_polyp_contact_Q=copy.deepcopy(keep),untrimmed_new_Q_arcs=copy.deepcopy(kept_arcs),
          short_Q_filter=dict(rule=RULE,removed_global_indices=sorted(removed),arcs=arcs))
 for ck in e['anchor_checks']:
  if ck['global_index'] in removed:ck['used_after_short_arc_filter']=False
 new=copy.deepcopy(base);new.uncertain=np.array([q['q'] for q in keep]).reshape(-1,2)
 new.excluded=np.array([z['q'] for z in d['records'] if z['class_']=='X']).reshape(-1,2)
 assert np.array_equal(new.trusted,base.trusted) and Pids==[z['global_index'] for z in d['records'] if z['class_']=='P']
 assert not removed&{q['global_index'] for q in keep}
 return new,e


def analyze(contours,Q):
 by={(int(q['contour_id']),int(q['contour_index'])):q for q in Q};removed=set();arcs=[]
 assert len(by)==len(Q)
 for cid,contour in enumerate(contours):
  c=contour.reshape(-1,2);states=np.array(['Q' if (cid,i) in by else 'X' for i in range(len(c))])
  perimeter=float(cv2.arcLength(c.astype(np.float32).reshape(-1,1,2),True))
  threshold=max(RULE['minimum_arclength_px'],RULE['contour_perimeter_fraction']*perimeter)
  for arc in r.boundary_v7.runs(states,'Q',cid):
   ids=arc['indices'];pts=c[ids].astype(float)
   length=float(np.linalg.norm(np.diff(pts,axis=0),axis=1).sum())
   if len(ids)==len(c) and len(c)>1:length+=float(np.linalg.norm(pts[-1]-pts[0]))
   gids=[int(by[(cid,i)]['global_index']) for i in ids];short=length<threshold
   if short:removed.update(gids)
   arcs.append(dict(contour_id=cid,indices=ids,global_indices=gids,n_points=len(ids),arc_length_px=length,contour_perimeter_px=perimeter,threshold_px=threshold,removed=short,wraps=arc['wraps']))
 return removed,arcs


# Bind relocated dependencies after defining functions (including override hooks).
from . import run_native as r
