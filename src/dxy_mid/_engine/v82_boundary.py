"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations

VERSION82B = 'V8.2-B-local-observed-edge-guard-1.0'

def near_ring_evidence(rec,d,polyp,rm,q,check):
    """Replace proximity veto with actual ring on the local normal corridor.

    The subsequent *unchanged* V8.1 3/5px inside/background and contrast tests
    still apply. A ring lying between those probes cannot be jumped over.
    Normal-tangent +/-1px samples accommodate the raster boundary footprint.
    """
    n,reason,metrics=v81.boundary_v7.normal(d['contours'][rec['contour_id']],rec['contour_index'],polyp)
    check.update(metrics)
    if n is None:check['repair_reject']=reason;return False
    tangent=v81.np.array([-n[1],n[0]])
    points=v81.np.array([q+n*t+tangent*k for t in range(-5,6) for k in [-1,0,1]])
    xy=v81.np.rint(points).astype(int);h,w=rm.shape
    outside=bool(((xy[:,0]<0)|(xy[:,0]>=w)|(xy[:,1]<0)|(xy[:,1]>=h)).any())
    check['normal_corridor_outside_frame']=outside
    if outside:check['repair_reject']='normal_corridor_outside_frame';return False
    overlap=float(rm[xy[:,1],xy[:,0]].mean());check['normal_corridor_ring_fraction']=overlap
    check['normal_corridor_samples']=33
    if overlap>0:check['repair_reject']='actual_ring_occludes_local_normal_corridor';return False
    check['near_ring_evidence']='observed_local_polyp_background_no_actual_ring_in_normal_corridor'
    return True


def repair_boundary(frame,base,raw,old):
    result,d=FROZEN_ADAPTED_REPAIR(frame,base,raw,old)
    for rec in d['records']:
        if rec.get('repair_evidence',{}).get('near_ring_evidence'):
            rec['reason']='observed_polyp_background_near_actual_ring_clear_normal_corridor'
    for rec in d['classification_changes']:
        rec['recovery_reason']='local_normal_corridor_and_unchanged_V81_image_evidence' if rec.get('near_ring_evidence') else 'unchanged_V81_outside_6px_evidence'
    d.update(version=VERSION82B,base_version=old['version'],guard_change='only replace 6px veto with current-mask normal-corridor evidence',adapted_source_sha256='b0ce69b1ed1f7f999821a8cc67029c6f3c2f39476fc3cdc4d3f3d56a3166042f')
    result.provenance.update(method='V8.2_local_observed_edge_guard_repair',implementation_version=VERSION82B)
    return result,d


# Bind relocated dependencies after defining functions (including override hooks).
from . import v81_components as v81
from .v81_components_slice import repair_boundary as FROZEN_ADAPTED_REPAIR
