"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
from dataclasses import dataclass as dataclass
import numpy as np


@dataclass
class FrameInput:
 image:np.ndarray
 labels:np.ndarray
 inner_diameter_mm:float
 source_id:str
 image_sha256:str
 mask_sha256:str
 coordinate_system:str='native_image_px'
 audit_adjacency:bool=False


# Bind relocated dependencies after defining functions (including override hooks).
from . import geometry as u1
from . import v2 as v2
