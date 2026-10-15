"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import numpy as np
RULE83 = {'normal_offsets': [3, 5, 9], 'normal_cos_min': 0.8, 'side_polyp_min': 0.6, 'side_background_min': 0.6, 'side_ring_max': 0.2, 'closed_background_min_area_rr2': 0.05, 'background_min_area_px': 25, 'ring_width_min_px': 3.0, 'ring_width_min_rr': 0.015, 'ring_width_max_rr': 0.85, 'local_arc_link_px': 3.0, 'local_arc_link_rr': 0.04, 'minimum_arc_points': 3, 'minimum_arc_extent_px': 5.0, 'minimum_arc_extent_rr': 0.05, 'arc_sample_max': 16, 'duplicate_Q_distance_px': 3.0, 'duplicate_Q_distance_rr': 0.04, 'normal_alignment_min': 0.65, 'in_margin_max_px': 1.0, 'in_margin_width_factor': 0.05, 'out_fractions': [0.1, 0.2, 0.35], 'no_original_PQX_changes': True}

def smooth_normal(contour,i):
    normals=[]
    for k in RULE83['normal_offsets']:
        v=contour[(i+k)%len(contour)]-contour[(i-k)%len(contour)]
        if np.linalg.norm(v)<1:return None
        normals.append(np.array([-v[1],v[0]])/np.linalg.norm(v))
    n=normals[1]
    return n if min(abs(n@x) for x in normals)>=RULE83['normal_cos_min'] else None


def mode_component(cc,q,radius):
    x,y=np.rint(q).astype(int);h,w=cc.shape
    patch=cc[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
    ids=patch[patch>0]
    return int(np.bincount(ids).argmax()) if len(ids) else 0


# Bind relocated dependencies after defining functions (including override hooks).
