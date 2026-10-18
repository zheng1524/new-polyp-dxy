"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from dataclasses import dataclass as dataclass
from dataclasses import field as field
import hashlib as hashlib
import numpy as np
VERSION = 'component-diagnostics-v3.1'

@dataclass
class RingResult:
 valid:bool
 reason:str
 geometry:dict
 provenance:dict
 evidence:dict=field(default_factory=dict)


def ellipse_axes(c,scale=None):
 A=np.eye(2) if scale is None else scale.matrix[:2,:2]
 center=c.center_native if scale is None else legacy.u1.transform_points(c.center_native[None],scale.matrix)[0]
 U,s,_=np.linalg.svd(A@c.basis_native)
 ends=np.stack([center-U[:,0]*s[0],center+U[:,0]*s[0]])
 return center,U,s,ends


def provenance(frame,method):return dict(version=VERSION,method=method,input_hash=hashlib.sha256((frame.image_sha256+'|'+frame.mask_sha256+'|'+str(float(frame.inner_diameter_mm))+'|1.2').encode()).hexdigest(),source_id=frame.source_id,image_sha256=frame.image_sha256,mask_sha256=frame.mask_sha256,coordinate_system='native_image_px',length_unit='px',history_prediction_used=False)


@dataclass
class ScaleResult:
 matrix:np.ndarray
 mm_per_px:float
 reference_diameter_px:float
 reference_diameter_mm:float
 coordinate_system:str
 assumption:str
 provenance:dict


@dataclass
class GateResult:
 accepted:bool
 reasons:list
 metrics:dict
 provenance:dict


@dataclass
class GeometryCandidate:
 name:str
 shape:str
 center_native:np.ndarray
 basis_native:np.ndarray # x=c+B*[cos(t),sin(t)] ; columns are radius vectors
 length_native_px:float
 quality:dict
 fit:dict
 provenance:dict


@dataclass
class BoundaryResult:
 candidates:list
 trusted:np.ndarray
 excluded:np.ndarray
 uncertain:np.ndarray
 unknown_mask:np.ndarray
 provenance:dict


# Bind relocated dependencies after defining functions (including override hooks).
from . import components as legacy
