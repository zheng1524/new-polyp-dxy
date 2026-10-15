"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.spatial._ckdtree import cKDTree as cKDTree
import copy as copy
import cv2 as cv2
import math as math
import numpy as np
VERSION84 = 'V8.4.4-conflict-Q-nearest-current-candidate-1.0'
RULE84 = {'anchor_to_polyp_exit_max_px': 3.0, 'anchor_to_exit_rr': 0.035, 'normal_alignment_min': 0.65, 'actual_material_ray_steps': [1.0, 2.0, 3.0], 'body_inside_probe_px': 3.0, 'body_inside_fraction_min': 0.6, 'minimum_arc_points': 3, 'in_max_px': 1.0, 'in_width_fraction': 0.05, 'out_fractions': [0.1, 0.2, 0.35], 'original_P_unchanged': True, 'original_Q_unchanged': True, 'PP_bridge_not_body_ownership': True, 'body_ownership_exclusion': False, 'trim_each_end_fraction': 0.2, 'trim_scope': 'new_contact_Q_only_when_original_P_positive', 'trim_distance_unit': 'native_contour_arclength_px'}

def make_data84(frame,base,raw,evidence,body,fraction):
    data=gc.make_data(frame,base,raw,evidence['original_boundary'],body,fraction)
    for q in evidence['polyp_contact_Q']:
        if body not in q['body_support']:continue
        item=copy.deepcopy(q);item['hi']=item.pop('hi_by_fraction')[str(fraction)];data['Q'].append(item)
    # The effective classification is original Q union newly accepted X→Q.
    # Reuse V7 runs on the SAME indexed polyp contour, rather than treating a
    # gap-fill fragment as an independent arc with an extra equal mass.
    by_index={(q['contour_id'],q['contour_index']):q for q in data['Q']}
    aid=0
    for cid,contour in enumerate(evidence['original_boundary']['contours']):
        states=np.array(['Q' if (cid,i) in by_index else 'X' for i in range(len(contour))])
        for arc in boundary_v7.runs(states,'Q',cid):
            for i in arc['indices']:
                q=by_index[(cid,i)];q['source_arc_id']=q['arc_id'];q['arc_id']=aid;q['effective_arc_source']='original_polyp_contour_contiguous_Q_run'
            aid+=1
    data['effective_Q_arc_count']=aid
    data['original_Q_count']=len(data['Q']);return data


def extract84(frame,ring,base,raw,frozen):
    samples=RING_SIDES(frame,ring,base,raw,frozen);records=samples['ring_samples']
    for s in records:s.pop('q',None) # Ring evidence coordinates are not Q anchors.
    pol=frame.labels==1;rm=frame.labels==2
    _,cc=cv2.connectedComponents(pol.astype(np.uint8),8)
    PB=[s for s in records if s['kind']=='PB'];tree=cKDTree([s['polyp_exit'] for s in PB]) if PB else None
    inspections=[];accepted={};radius=max(RULE84['anchor_to_polyp_exit_max_px'],RULE84['anchor_to_exit_rr']*raw['rr'])
    for rec in frozen['records']:
        if rec['class_']!='X':continue
        q=np.array(rec['q'],float);x,y=map(int,q);check=dict(global_index=rec['global_index'],contour_id=rec['contour_id'],contour_index=rec['contour_index'],q=q.tolist(),old_class='X',old_reason=rec['reason'],new_class='X',reason='',body_support=[],source_kind='original_polyp_mask_contour_pixel')
        if rec['parent_contour']>=0:check['reason']='internal_polyp_hole';inspections.append(check);continue
        if tree is None:check['reason']='no_independent_PB_evidence';inspections.append(check);continue
        matches=tree.query_ball_point(q,radius)
        if not matches:check['reason']='no_local_PB_polyp_side_exit_match';inspections.append(check);continue
        n,why,metrics=boundary_v7.normal(frozen['contours'][rec['contour_id']],rec['contour_index'],pol);check.update(metrics)
        if n is None:check['reason']=why;inspections.append(check);continue
        local=[]
        for i in matches:
            s=PB[i]
            if s['polyp_component_id']!=int(cc[y,x]):continue
            if float(n@np.array(s['n']))<RULE84['normal_alignment_min']:continue
            local.append(s)
        if not local:check['reason']='PB_component_or_polyp_normal_mismatch';inspections.append(check);continue
        local.sort(key=lambda s:(np.linalg.norm(np.array(s['polyp_exit'])-q),s['source_index']));s=local[0]
        ray=np.rint(q+np.array(RULE84['actual_material_ray_steps'])[:,None]*n).astype(int);ray=np.clip(ray,[0,0],[rm.shape[1]-1,rm.shape[0]-1])
        if not rm[ray[:,1],ray[:,0]].any():check['reason']='outward_ray_has_no_actual_ring_material';inspections.append(check);continue
        supports=[];ownership={}
        established_Q_contours={r['contour_id'] for r in frozen['records'] if r['class_']=='Q'}
        for ca in base.candidates:
            inward=boundary_v7.RAW.frac(ca['mask'].astype(bool),q-n*RULE84['body_inside_probe_px'],1)
            owned=bool(ca['mask'][y,x]) and inward>=RULE84['body_inside_fraction_min']
            # Preserve the existing V7 P=0 body contract. An empty post-visibility
            # V2 mask is not the Q-only carrier; original Q already establishes it.
            carrier=(not ca['mask'].any() and frozen['body_carrier_source']=='V2_broad_carrier_before_visible_gate_zero_P' and rec['contour_id'] in established_Q_contours and int(cc[y,x]) in frozen['body_component_ids'])
            owned=owned or carrier
            ownership[ca['name']]=dict(original_anchor_in_candidate_mask=bool(ca['mask'][y,x]),inward_candidate_fraction=inward,owned=owned,existing_V7_Q_only_carrier=carrier,ownership_source='frozen_V7_zero_P_same_contour_Q_carrier' if carrier else 'current_V2_AB_mask_and_inward_probe')
            # User-reviewed removal: ownership remains diagnostic only.
            ownership[ca['name']]['used_as_acceptance_condition']=False
            supports.append(ca['name'])
        check.update(n=n.tolist(),local_ring_thickness_px=s['width_px'],ring_sample_index=s['source_index'],ring_contour_id=s['ring_contour_id'],ring_contour_index=s['ring_contour_index'],polyp_exit=s['polyp_exit'],background_exit=s['background_exit'],PB_side_labels=s['side_labels'],normal_ring_alignment=float(n@np.array(s['n'])),body_ownership=ownership,body_support=supports,normal_source='smoothed_original_polyp_contour',anchor_source='original_polyp_mask_contour_pixel')
        # No rejection or solver filtering by V2 A/B mask ownership.
        w=s['width_px'];check.update(delta_in_px=min(RULE84['in_max_px'],RULE84['in_width_fraction']*w),delta_out_px={str(f):f*w for f in RULE84['out_fractions']},new_class='Q',reason='indexed_polyp_contour_PB_contact_verified',semantic='Q_constraint',is_observed_P=False,is_postfit_inference=False)
        accepted[rec['global_index']]=check;inspections.append(check)
    extra=[];arcs=[];arc_offset=len(frozen['Q_arcs'])
    for cid,c in enumerate(frozen['contours']):
        ids={r['contour_index']:r['global_index'] for r in frozen['records'] if r['contour_id']==cid};states=np.array(['Q' if ids[i] in accepted else 'X' for i in range(len(c))])
        for arc in boundary_v7.runs(states,'Q',cid):
            if len(arc['indices'])<RULE84['minimum_arc_points']:
                for j in arc['indices']:accepted[ids[j]].update(new_class='X',reason='verified_contact_arc_too_short')
                continue
            aid=arc_offset+len(arcs);gl=[]
            for j in arc['indices']:
                rec=accepted[ids[j]];gl.append(rec['global_index']);extra.append(dict(q=np.array(rec['q']),n=np.array(rec['n']),lo=-rec['delta_in_px'],hi_by_fraction=rec['delta_out_px'],arc_id=aid,global_index=rec['global_index'],contour_id=cid,contour_index=j,body_support=rec['body_support'],ring_width_px=rec['local_ring_thickness_px'],source_kind='original_polyp_contour_ring_contact_Q',source_index=rec['ring_sample_index'],normal_source='smoothed_original_polyp_contour',polyp_exit=rec['polyp_exit'],background_exit=rec['background_exit'],semantic='Q_constraint',is_observed_P=False,is_postfit_inference=False))
            arcs.append(dict(**arc,arc_id=aid,global_indices=gl))
    evidence=dict(version=VERSION84,rules=RULE84,original_boundary=frozen,ring_samples=records,anchor_checks=inspections,polyp_contact_Q=extra,new_Q_arcs=arcs,original_P_unchanged=True,original_Q_unchanged=True,no_ring_exit_Q_anchor=True)
    return trim_contact_arcs(evidence)


def trim_contact_arcs(evidence):
    # Boundary changes only: retained coordinates/normals/intervals are identical.
    original=evidence['original_boundary'];has_P=any(r['class_']=='P' for r in original['records'])
    raw_Q=evidence['polyp_contact_Q'];raw_arcs=evidence['new_Q_arcs']
    evidence['untrimmed_polyp_contact_Q']=copy.deepcopy(raw_Q)
    evidence['untrimmed_new_Q_arcs']=copy.deepcopy(raw_arcs)
    lookup={q['global_index']:q for q in raw_Q};checks={r['global_index']:r for r in evidence['anchor_checks']}
    retained=[];arcs=[];audit=[];removed=[]
    for arc in raw_arcs:
        ids=arc['global_indices'];points=np.array([lookup[i]['q'] for i in ids]);closed=len(arc['indices'])==len(original['contours'][arc['contour_id']])
        keep,distance,length=arc_trim_mask(points,RULE84['trim_each_end_fraction'])
        apply=has_P and not closed
        if not apply:keep[:]=True
        too_short=apply and int(keep.sum())<RULE84['minimum_arc_points']
        endpoint_removed=~keep.copy()
        if too_short:keep[:]=False
        kept=[i for i,on in zip(ids,keep) if on]
        for j,i in enumerate(ids):
            ck=checks[i];ck.update(trim_applied=apply,trim_source_arc_id=arc['arc_id'],source_arclength_px=float(distance[j]),source_arc_length_px=length,trim_fraction_each_end=RULE84['trim_each_end_fraction'])
            if keep[j]:retained.append(lookup[i])
            else:
                why='mixed_PQ_Q_arc_endpoint_trim_20pct' if endpoint_removed[j] else 'remaining_trimmed_Q_arc_too_short'
                ck.update(new_class='X',reason=why)
                removed.append(dict(**copy.deepcopy(lookup[i]),trim_reason=why,source_arclength_px=float(distance[j]),source_arc_length_px=length,trim_source_arc_id=arc['arc_id']))
        if kept:
            indices=[j for j,on in zip(arc['indices'],keep) if on]
            updated=copy.deepcopy(arc);updated.update(start_index=indices[0],indices=indices,global_indices=kept,wraps=any(b<a for a,b in zip(indices,indices[1:])))
            arcs.append(updated)
        audit.append(dict(contour_id=arc['contour_id'],arc_id=arc['arc_id'],original_indices=arc['indices'],original_global_indices=ids,kept_global_indices=kept,original_points=len(ids),retained_points=len(kept),removed_points=len(ids)-len(kept),original_arc_length_px=length,kept_arc_length_px=float(np.ptp(distance[keep])) if keep.any() else 0.,cut_start_target_px=.2*length if apply else 0.,cut_end_target_px=.2*length if apply else 0.,retained_start_distance_px=float(distance[keep][0]) if keep.any() else math.nan,retained_end_distance_px=float(distance[keep][-1]) if keep.any() else math.nan,applied=apply,closed_arc=closed,frame_has_original_P=has_P,remaining_too_short=too_short,decision='trim_each_end_20pct' if apply else 'closed_arc_has_no_endpoints' if closed and has_P else 'zero_P_unchanged'))
    evidence.update(polyp_contact_Q=retained,new_Q_arcs=arcs,endpoint_trim_audit=audit,trimmed_Q_points=removed)
    return evidence


def arc_trim_mask(points, fraction):
    """Select existing samples in the middle of an OPEN arc, by Euclidean arclength."""
    points=np.asarray(points,float)
    distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    length=float(distance[-1]) if len(distance) else 0.
    keep=(distance>=fraction*length-1e-9)&(distance<=(1-fraction)*length+1e-9)
    return keep,distance,length


# Bind relocated dependencies after defining functions (including override hooks).
from . import boundary_v7 as boundary_v7
from . import geometry_competition as gc
from .v83_boundary import extract as RING_SIDES
