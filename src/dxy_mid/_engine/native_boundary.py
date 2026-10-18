"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import copy as copy
import cv2 as cv2
import numpy as np
TASK_VERSION = 'V8.4.6-native-P-frozen-Q-1.1'
removed = ['legacy_trusted_membership_for_P', 'V2_body_component_for_P', 'V2_unknown_mask_for_P', 'V2_carrier_distance_for_P', 'Geometry_legacy_trusted_intersection_for_P']

def native_P(frame):
    pol=frame.labels==1;rm=frame.labels==2;bg=frame.labels==0
    contours,hierarchy=cv2.findContours(pol.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours=[c.reshape(-1,2) for c in contours]
    hierarchy=hierarchy[0] if hierarchy is not None else np.empty((0,4),int)
    gray=cv2.cvtColor(frame.image,cv2.COLOR_BGR2GRAY);P=[];checks=[]
    details=dict(contours=contours)
    for cid,c in enumerate(contours):
        for i,q in enumerate(c):
            rec=dict(contour_id=cid,contour_index=i,q=q.tolist())
            check=dict(**rec,accepted_P=False,repair_reject='internal_polyp_hole')
            if hierarchy[cid,3]>=0:checks.append(check);continue
            if not near_ring_evidence(rec,details,pol,rm,q,check):checks.append(check);continue
            n,why,metrics=boundary_v7.normal(c,i,pol)
            ins=[boundary_v7.RAW.frac(pol,q-n*hop,1) for hop in v81.RULES['normal_probe_px']]
            outs=[boundary_v7.RAW.frac(bg,q+n*hop,1) for hop in v81.RULES['normal_probe_px']]
            a=np.rint(q+n*3).astype(int);b=np.rint(q-n*3).astype(int)
            contrast=abs(float(gray[a[1],a[0]])-float(gray[b[1],b[0]]))
            check.update(inside_fraction=min(ins),background_fraction=min(outs),image_contrast=contrast,n=n.tolist())
            if min(ins)<v81.RULES['inside_min'] or min(outs)<v81.RULES['background_min']:check['repair_reject']='polyp_background_normal_evidence_insufficient'
            elif contrast<v81.RULES['image_contrast_min']:check['repair_reject']='image_edge_contrast_insufficient'
            else:P.append(q);check.update(accepted_P=True,repair_reject='')
            checks.append(check)
    return np.array(P,float).reshape(-1,2),checks


def boundary_native(frame,ring):
    # Reuse the approved current-frame Q extraction and A/B descriptors only.
    # Its inherited P admissions are discarded below and never veto new P.
    oldbase,raw,old=current_boundary(frame,ring)
    evidence=extract84(frame,ring,oldbase,raw,old)
    frozen=copy.deepcopy(evidence['original_boundary'])
    _,checks=native_P(frame)
    lookup={(r['contour_id'],r['contour_index']):r for r in checks}
    protected_Q={r['global_index'] for r in frozen['records'] if r['class_']=='Q'}|{q['global_index'] for q in evidence['polyp_contact_Q']}
    for r in frozen['records']:
        ck=lookup[(r['contour_id'],r['contour_index'])];r['native_evidence']=ck;r['previous_class_']=r['class_'];r['previous_reason']=r['reason']
        if r['global_index'] in protected_Q:
            # Retain the existing independent, uncertain Q semantics.
            continue
        if ck['accepted_P']:r.update(class_='P',reason='observed_polyp_background_current_mask_and_image')
        else:r.update(class_='X',reason=ck['repair_reject'],P_rejection_reason=ck['repair_reject'])
    P=np.array([r['q'] for r in frozen['records'] if r['class_']=='P'],float).reshape(-1,2)
    base=copy.deepcopy(oldbase)
    for ca in base.candidates:
        ca['trusted']=P.copy();ca['points_local']=P-ca['crop_offset']
        ca['P_admission_source']='current_mask_image_evidence_no_V2_membership_intersection'
    base.trusted=P;base.provenance.update(implementation_version=TASK_VERSION,method='native_P_frozen_Q')
    frozen.update(version=TASK_VERSION,P_checks=checks,removed_V2_predicates=removed,legacy_trusted_used_for_P_admission=False)
    evidence.update(version=TASK_VERSION,original_boundary=frozen,original_P_unchanged=False,original_Q_unchanged=True,Q_extraction_and_endpoint_trim_frozen=True)
    for body in ['A','B']:
        data=make_data84(frame,base,raw,evidence,body,.2)
        expected=[r['global_index'] for r in frozen['records'] if r['class_']=='P']
        assert data['P_global_indices']==expected,'Boundary P dropped before Geometry'
    return base,raw,evidence


# Bind relocated dependencies after defining functions (including override hooks).
from . import boundary_v7 as boundary_v7
from . import v81_components as v81
from .v82_boundary import near_ring_evidence as near_ring_evidence
from .v83_context import current_boundary as current_boundary
from .v84_boundary import extract84 as extract84
from .v84_boundary import make_data84 as make_data84
