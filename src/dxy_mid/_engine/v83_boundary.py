"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from scipy.spatial._ckdtree import cKDTree as cKDTree
import cv2 as cv2
import numpy as np
RULE83 = {'normal_offsets': [3, 5, 9], 'normal_cos_min': 0.8, 'side_polyp_min': 0.6, 'side_background_min': 0.6, 'side_ring_max': 0.2, 'closed_background_min_area_rr2': 0.05, 'background_min_area_px': 25, 'ring_width_min_px': 3.0, 'ring_width_min_rr': 0.015, 'ring_width_max_rr': 0.85, 'local_arc_link_px': 3.0, 'local_arc_link_rr': 0.04, 'minimum_arc_points': 3, 'minimum_arc_extent_px': 5.0, 'minimum_arc_extent_rr': 0.05, 'arc_sample_max': 16, 'duplicate_Q_distance_px': 3.0, 'duplicate_Q_distance_rr': 0.04, 'normal_alignment_min': 0.65, 'in_margin_max_px': 1.0, 'in_margin_width_factor': 0.05, 'out_fractions': [0.1, 0.2, 0.35], 'no_original_PQX_changes': True}
VERSION83 = 'V8.3-independent-ring-contact-Q-1.0'

def extract(frame,ring,base,raw,frozen):
    """Original Q retained. New constraints have a separate provenance namespace.

    q=background-side exit of an observed local material run; n points from
    polyp side to background. [-w-min(1,.05w), f*w] covers the occluding material
    plus uncertainty. q is never asserted to be an observed polyp edge.
    """
    labels=frame.labels;rm=labels==2;pol=labels==1;bg=labels==0;rr=raw['rr'];RAW=boundary_v7.RAW
    _,pcc,pstats,_=cv2.connectedComponentsWithStats(pol.astype(np.uint8),8)
    _,bcc,bstats,_=cv2.connectedComponentsWithStats(bg.astype(np.uint8),8)
    border_bg=set(np.unique(np.r_[bcc[0],bcc[-1],bcc[:,0],bcc[:,-1]]))-{0}
    contours,hierarchy=cv2.findContours(rm.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    contours=[c.reshape(-1,2).astype(float) for c in contours]
    forward=max(1,round(RAW.SAMPLE_FORWARD*rr));radius=max(1,round(RAW.SAMPLE_RADIUS*rr));maxcross=max(4,round(RAW.MAX_CROSS*rr))
    records=[];bridges=[];candidates=[]
    for cid,c in enumerate(contours):
        for i,point in enumerate(c):
            rec=dict(source_index=len(records),ring_contour_id=cid,ring_contour_index=i,ring_point=point.tolist(),kind='unknown',reason='')
            n=smooth_normal(c,i) if len(c)>=19 else None
            if n is None:rec['reason']='local_ring_normal_unreliable';records.append(rec);continue
            exits=[RAW.beyond(rm,point,n*sgn,maxcross) for sgn in [1,-1]]
            if any(e is None for e in exits):rec['reason']='material_run_exit_missing';records.append(rec);continue
            sites=[exits[j]+sgn*n*forward for j,sgn in enumerate([1,-1])]
            fractions=[dict(polyp=RAW.frac(pol,s,radius),background=RAW.frac(bg,s,radius),ring=RAW.frac(rm,s,radius)) for s in sites]
            side=['P' if x['polyp']>=RULE83['side_polyp_min'] and x['ring']<=RULE83['side_ring_max'] else 'B' if x['background']>=RULE83['side_background_min'] and x['ring']<=RULE83['side_ring_max'] else 'U' for x in fractions]
            ids=[mode_component(pcc,s,radius) if kind=='P' else mode_component(bcc,s,radius) if kind=='B' else 0 for s,kind in zip(sites,side)]
            w=float(np.linalg.norm(exits[1]-exits[0]));rec.update(side_labels=side,side_fractions=fractions,side_component_ids=ids,exit_plus=exits[0].tolist(),exit_minus=exits[1].tolist(),width_px=w,normal_raw=n.tolist())
            if not max(RULE83['ring_width_min_px'],RULE83['ring_width_min_rr']*rr)<=w<=RULE83['ring_width_max_rr']*rr:
                rec['reason']='local_material_thickness_invalid';records.append(rec);continue
            if side==['P','P']:
                rec.update(kind='PP',reason='ring_inside_polyp_not_outer_constraint');bridges.append((ids[0],ids[1],rec['source_index']))
            elif set(side)=={'P','B'}:
                pi=side.index('P');bi=side.index('B');pc,bc=ids[pi],ids[bi]
                if bc==0 or (bc not in border_bg and bstats[bc,cv2.CC_STAT_AREA]<max(RULE83['background_min_area_px'],RULE83['closed_background_min_area_rr2']*rr*rr)):
                    rec.update(kind='PB_small_hole',reason='small_closed_background_hole_not_external_evidence')
                else:
                    pe,be=exits[pi],exits[bi];normal=(be-pe)/w
                    rec.update(kind='PB',reason='',polyp_component_id=pc,background_component_id=bc,polyp_exit=pe.tolist(),background_exit=be.tolist(),n=normal.tolist(),q=be.tolist(),midpoint=((pe+be)/2).tolist())
                    candidates.append(rec)
            else:rec['reason']='two_sides_ambiguous_or_background_only'
            records.append(rec)
    graph={}
    for a,b,idx in bridges:
        if a and b and a!=b:graph.setdefault(a,set()).add(b);graph.setdefault(b,set()).add(a)
    def reachable(seeds):
        seen=set(seeds);todo=list(seen)
        while todo:
            for nxt in graph.get(todo.pop(),[]):
                if nxt not in seen:seen.add(nxt);todo.append(nxt)
        return seen
    body_ids={}
    for ca in base.candidates:
        seeds={int(pcc[int(q[1]),int(q[0])]) for q in ca['trusted'] if pol[int(q[1]),int(q[0])]}-{0}
        if not seeds:seeds=set(frozen['body_component_ids'])
        body_ids[ca['name']]=reachable(seeds)
    old_Q=[r for r in frozen['records'] if r['class_']=='Q'];eligible=[]
    oldq=np.array([r['q'] for r in old_Q],float).reshape(-1,2);oldn=np.array([r['n'] for r in old_Q],float).reshape(-1,2)
    for rec in candidates:
        supported=[body for body,ids in body_ids.items() if rec['polyp_component_id'] in ids]
        if not supported:rec['reason']='not_connected_to_current_V2_body_by_observed_PP_bridge';continue
        rec['body_support']=supported;q=np.array(rec['q']);n=np.array(rec['n']);pe=np.array(rec['polyp_exit']);be=q
        t=(oldq-pe)@n;across=np.linalg.norm(oldq-(pe+np.clip(t,0,rec['width_px'])[:,None]*n),axis=1)
        duplicate=bool(np.any((across<=max(RULE83['duplicate_Q_distance_px'],RULE83['duplicate_Q_distance_rr']*rr))&(t>=-.05*rec['width_px'])&(t<=1.05*rec['width_px'])&(oldn@n>=RULE83['normal_alignment_min'])))
        if duplicate:rec['reason']='already_supported_by_original_polyp_Q';continue
        eligible.append(rec)
    # Adjacent crossbars from inner/outer ring edges merge into one physical arc,
    # rather than letting duplicate dense material contours gain double weight.
    parents=list(range(len(eligible)))
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    link=max(RULE83['local_arc_link_px'],RULE83['local_arc_link_rr']*rr)
    if eligible:
        mids=np.array([r['midpoint'] for r in eligible]);tree=cKDTree(mids)
        for a,b in sorted(tree.query_pairs(1.5*link)):
            x,y=eligible[a],eligible[b]
            if x['polyp_component_id']==y['polyp_component_id'] and x['body_support']==y['body_support'] and np.array(x['n'])@np.array(y['n'])>=.8 and .5<=x['width_px']/y['width_px']<=2:
                parents[root(b)]=root(a)
    grouped={}
    for i,rec in enumerate(eligible):grouped.setdefault(root(i),[]).append(rec)
    Q=[];arcs=[];center=np.array([ring.geometry['inner']['cx'],ring.geometry['inner']['cy']]);offset=len(frozen['records']);arc_offset=len(frozen['Q_arcs'])
    for cluster in grouped.values():
        mids=np.array([x['midpoint'] for x in cluster]);extent=float(np.linalg.norm(np.ptp(mids,axis=0)))
        if len(cluster)<RULE83['minimum_arc_points'] or extent<max(RULE83['minimum_arc_extent_px'],RULE83['minimum_arc_extent_rr']*rr):
            for rec in cluster:rec['reason']='independent_ring_arc_too_short'
            continue
        angles=np.mod(np.arctan2(mids[:,1]-center[1],mids[:,0]-center[0]),2*np.pi);order=np.argsort(angles);gap=np.diff(np.r_[angles[order],angles[order][0]+2*np.pi]);start=(int(np.argmax(gap))+1)%len(order);order=np.roll(order,-start)
        # Existing coordinates only; spatial thinning removes duplicate crossbars.
        ordered=[]
        for i in order:
            if not ordered or np.linalg.norm(mids[i]-np.array(ordered[-1]['midpoint']))>=link:ordered.append(cluster[i])
        chosen=[ordered[i] for i in gc.uniform(np.arange(len(ordered)),RULE83['arc_sample_max'])]
        aid=arc_offset+len(arcs);arcids=[]
        for rec in chosen:
            idx=offset+len(Q);w=rec['width_px'];margin=min(RULE83['in_margin_max_px'],RULE83['in_margin_width_factor']*w)
            q=dict(q=np.array(rec['q']),n=np.array(rec['n']),lo=-w-margin,hi_by_fraction={str(f):f*w for f in RULE83['out_fractions']},arc_id=aid,global_index=idx,contour_id=None,contour_index=None,source_kind='ring_contact_independent_constraint',source_index=rec['source_index'],ring_contour_id=rec['ring_contour_id'],ring_contour_index=rec['ring_contour_index'],body_support=rec['body_support'],ring_width_px=w,polyp_exit=rec['polyp_exit'],background_exit=rec['background_exit'],polyp_component_id=rec['polyp_component_id'],background_component_id=rec['background_component_id'],semantic='Q_constraint',is_observed_P=False,is_postfit_inference=False)
            Q.append(q);arcids.append(idx);rec['reason']='independent_ring_Q_selected'
        arcs.append(dict(arc_id=aid,global_indices=arcids,body_support=cluster[0]['body_support'],dense_crossbars=len(cluster),extent_px=extent))
    return dict(version=VERSION83,rules=RULE83,ring_samples=records,PP_bridges=[dict(component_a=a,component_b=b,source_index=i) for a,b,i in bridges],body_component_ids={k:sorted(v) for k,v in body_ids.items()},ring_contact_Q=Q,ring_contact_arcs=arcs,original_boundary=frozen,original_PQX_unchanged=True)


# Bind relocated dependencies after defining functions (including override hooks).
from . import boundary_v7 as boundary_v7
from . import geometry_competition as gc
from .v83_boundary_slice import mode_component as mode_component
from .v83_boundary_slice import smooth_normal as smooth_normal
