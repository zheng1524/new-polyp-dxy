"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import cv2 as cv2
import numpy as np
TASK_VERSION = 'V8.4.7-observable-native-P-frozen-Q-1.1'

def boundary_native(frame,ring):
    # Reuse the complete approved Q extraction/trim and interface integration.
    saved=prior.native_P,prior.TASK_VERSION
    prior.native_P=observable_P;prior.TASK_VERSION=TASK_VERSION
    try:base,raw,e=prior.boundary_native(frame,ring)
    finally:prior.native_P,prior.TASK_VERSION=saved
    records=e['original_boundary']['records'];extra={q['global_index'] for q in e['polyp_contact_Q']}
    # Cropped contact-Q tails remain uncertain/excluded. A one-pixel mask gap
    # cannot upgrade independently verified ring-occlusion evidence to exact P.
    protected_X={z['global_index']:z['reason'] for z in e['anchor_checks'] if z['new_class']=='X' and 'delta_in_px' in z}
    for rec in records:
        rec['native_evidence']['effective_P_qualified']=rec['class_']=='P'
        if rec['global_index'] in protected_X:
            rec.update(class_='X',reason=protected_X[rec['global_index']])
            rec['native_evidence'].update(effective_P_qualified=False,independent_occlusion_veto='approved_contact_Q_endpoint_or_short_arc_exclusion')
        elif rec['class_']=='P':rec['reason']='observed_original_polyp_contour_background_neighbor_no_actual_ring_contact'
    P=np.array([r['q'] for r in records if r['class_']=='P'],float).reshape(-1,2)
    for ca in base.candidates:ca['trusted']=P.copy();ca['points_local']=P-ca['crop_offset']
    points={cl:[] for cl in ['P','Q','X']}
    for r in records:
        cl='Q' if r['global_index'] in extra else r['class_'];points[cl].append(r['q'])
    base.trusted=np.array(points['P'],float).reshape(-1,2)
    base.excluded=np.array(points['X'],float).reshape(-1,2)
    base.uncertain=np.array(points['Q'],float).reshape(-1,2)
    base.provenance.update(method='original_contour_observed_mask_adjacency',P_quality_veto=False,legacy_P_exclusion_used=False)
    return base,raw,e


def observable_P(frame):
    pol=frame.labels==1;rm=frame.labels==2;bg=frame.labels==0
    contours,hierarchy=cv2.findContours(pol.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    hierarchy=hierarchy[0] if hierarchy is not None else np.empty((0,4),int)
    P=[];checks=[];h,w=pol.shape
    for cid,contour in enumerate(contours):
        contour=contour.reshape(-1,2)
        for i,q in enumerate(contour):
            x,y=map(int,q);ck=dict(contour_id=cid,contour_index=i,q=q.tolist(),accepted_P=False,repair_reject='',evidence_source='original_contour_actual_3x3_mask_adjacency')
            if hierarchy[cid,3]>=0:ck['repair_reject']='internal_polyp_hole'
            elif x==0 or y==0 or x==w-1 or y==h-1:ck['repair_reject']='frame_border_truncation'
            else:
                ring_count=int(rm[y-1:y+2,x-1:x+2].sum());background_count=int(bg[y-1:y+2,x-1:x+2].sum())
                ck.update(actual_ring_neighbor_count=ring_count,background_neighbor_count=background_count)
                if ring_count:ck['repair_reject']='actual_ring_contacts_original_polyp_pixel_boundary'
                elif not background_count:ck['repair_reject']='no_observed_background_neighbor'
                else:ck['accepted_P']=True;P.append(q)
            checks.append(ck)
    return np.array(P,float).reshape(-1,2),checks


# Bind relocated dependencies after defining functions (including override hooks).
from . import native_boundary as prior
