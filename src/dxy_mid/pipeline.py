"""DXY-MID: current image/mask evidence only, frozen V9.1 numerical behavior."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import threading
import cv2
import numpy as np
from . import _engine as engine

VERSION='DXY-MID-1.0.0rc1'
# Frozen helpers temporarily override module hooks. Threads must never interleave
# these overrides. Batch CLI uses separate worker processes instead.
_LOCK=threading.RLock()

@dataclass(frozen=True)
class MeasurementConfig:
    geometry_version: str = 'V9.1'
    scale_method: str = 'mid_major'
    ring_wire_diameter_mm: float = 1.2
    coordinate_system: str = 'native_image_px'
    def validate(self):
        if asdict(self)!=asdict(MeasurementConfig()):
            raise ValueError('This release preserves the frozen V9.1 + Mid configuration')

@dataclass
class Measurement:
    audit: dict
    ring: object
    candidate: object = None
    evidence: dict | None = None
    variants: dict | None = None


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def measure_frame(image: np.ndarray, labels: np.ndarray, *, inner_diameter_mm: float,
                  source_id: str, stem: str | None=None, image_sha256: str='',
                  mask_sha256: str='', config: MeasurementConfig | None=None) -> Measurement:
    """Compute from BGR image and aligned uint8 {0:background,1:polyp,2:ring}.

    source_id and stem are reproducibility identifiers used only as RNG seeds.
    No GT, group type, saved geometry or other-route predictions are accepted.
    Full five-variant stability protocol is retained. Numeric non-strict results
    remain measurement-eligible; hard failures return NaN and a recorded reason.
    """
    (config or MeasurementConfig()).validate()
    if inner_diameter_mm not in (5.,10.):raise ValueError('inner_diameter_mm must be 5 or 10')
    if image is None or labels is None or image.ndim!=3 or image.shape[:2]!=labels.shape:
        raise ValueError('Image and class mask must have the same native dimensions')
    if labels.dtype!=np.uint8 or not set(np.unique(labels))<={0,1,2}:
        raise ValueError('Expected uint8 class IDs 0/1/2, not color overlays')
    if not source_id:raise ValueError('A stable source_id is required')
    with _LOCK:
        return _measure(image,labels,inner_diameter_mm,source_id,stem or source_id,
                        image_sha256,mask_sha256)


def _measure(image,labels,inner_mm,source_id,stem,image_sha,mask_sha):
    cv2.setNumThreads(1)
    effective,filled,hole_info=engine.holes(labels)
    frame=engine.FrameInput(image,effective,float(inner_mm),source_id,image_sha,mask_sha)
    ring,attempts=engine.ring(frame,None)
    audit=dict(source_id=source_id,stem=stem,version=VERSION,Geometry='V9.1',Scale='Mid',
        ring_valid=bool(ring.valid),ring_reason=ring.reason,
        measurement_eligible=False,D_frame_mm=math.nan,L_px=math.nan,
        scale_mm_per_px=math.nan,L_mid_px=math.nan,D_mid_mm=inner_mm+1.2,
        normal_Gate_pass=False,Q_conflict_recovery=False,strict_stable=False,
        accepted=False,candidate_generated=False,stability_review_pass=False,
        geometry_state='fit_failure',quality_class='ring_failure',reason='ring_failure',
        stability_flags='ring_failure',P_count=0,Q_count=0,X_count=0,
        ring_filled_area_px=int(filled.sum()),Ring_branch=ring.geometry.get('branch',''),
        history_prediction_fallback=0,GT_read=False,coordinate_system='native_image_px',
        image_sha256=image_sha,mask_sha256=mask_sha,
        processed_mask_sha256=hashlib.sha256(effective.tobytes()).hexdigest())
    if not ring.valid:return Measurement(audit,ring)
    base,raw,e=engine.boundary(frame,ring)
    base,e=engine.cross_ring(frame,ring,base,raw,e)
    contours=e['original_boundary']['contours']
    removed,arcs=engine.short_Q_analyze(contours,e['source_Q_constraints'])
    base,e=engine.short_Q(base,e,removed)
    row=dict(source_id=source_id,stem=stem,group='current_frame',role='mainline')
    current,_,failures=engine.baseline(frame,base,raw,row)
    globals_=engine.configure.__globals__;old=globals_['make_data84']
    globals_['make_data84']=engine.make_data
    try:results=engine.configure(frame,base,raw,e,current,'anchor')
    finally:globals_['make_data84']=old
    selected=engine.choose(results['default'])
    audit.update(engine.audit(row,results,None))
    candidate=selected['chosen'];data=selected['data']
    length=float(candidate.length_native_px) if candidate is not None else math.nan
    numeric=bool(np.isfinite(length) and length>0)
    recovery=bool(audit.get('Q_distance_recovery',False) and numeric)
    normal=bool(numeric and not recovery and audit.get('accepted',False))
    scale=engine.scale(frame,ring,'mid_major')
    if scale.coordinate_system!='native_image_px' or not np.array_equal(scale.matrix,np.eye(3)):
        raise ValueError('Mid scale must operate in native image coordinates')
    if not np.isfinite(scale.mm_per_px) or scale.mm_per_px<=0:
        raise ValueError('Invalid scale for valid Ring')
    audit.update(L_px=length,measurement_eligible=numeric,
        scale_mm_per_px=scale.mm_per_px,L_mid_px=scale.reference_diameter_px,
        D_frame_mm=length*scale.mm_per_px if numeric else math.nan,
        Q_conflict_recovery=recovery,normal_Gate_pass=normal,
        strict_stable=bool(numeric and audit.get('stability_review_pass',False)),
        P_count=len(data['P']),Q_count=len(data['Q']),
        X_count=sum(z['class_']=='X' for z in e['original_boundary']['records']),
        Q_removed_short_arc_count=len(removed),
        P_zero_occlusion_inference=bool(len(data['P'])==0 and len(data['Q'])>0),
        shape=candidate.shape if candidate else '',body=candidate.name if candidate else '',
        quality_class='Q_conflict_recovery' if recovery else 'normal_Gate_pass' if normal else 'numeric_only' if numeric else 'hard_no_selected_numeric')
    if candidate is not None:
        _,basis,axes,endpoints=engine.axes(candidate)
        major,minor=2*axes[0],2*axes[1]
        audit.update(major_axis_px=float(major),minor_axis_px=float(minor),
            axis_x1=float(endpoints[0][0]),axis_y1=float(endpoints[0][1]),
            axis_x2=float(endpoints[1][0]),axis_y2=float(endpoints[1][1]))
    evidence=dict(P=data['P'],Q=data['Q'],X=base.excluded,
        sampled_Q_indices=selected['actual_sampled_Q_indices'],
        scored_Q_indices=selected['plan']['score_qi'],ring_holes=hole_info,
        V2_seed_failures=failures)
    variants={k:[dict(body=z['data']['name'],chosen=z['chosen'],UD=z['UD'],
                      range_px=z['range_px'],reason=z['reason'],branch=z['branch']) for z in v]
              for k,v in results.items()}
    return Measurement(audit,ring,candidate,evidence,variants)


def measure_files(image_path,mask_path,*,inner_diameter_mm,source_id=None,stem=None,**kwargs):
    image_path=Path(image_path);mask_path=Path(mask_path)
    image_sha=sha256(image_path);mask_sha=sha256(mask_path)
    # For new inputs this content identifier is reproducible; frozen manifests
    # retain their original IDs and stems to preserve their random sequences.
    source_id=source_id or hashlib.sha256((image_sha+'|'+mask_sha).encode()).hexdigest()[:24]
    return measure_frame(cv2.imread(str(image_path)),cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED),
        inner_diameter_mm=float(inner_diameter_mm),source_id=source_id,stem=stem or source_id,
        image_sha256=image_sha,mask_sha256=mask_sha,**kwargs)
