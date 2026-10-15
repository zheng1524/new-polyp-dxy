"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations



def current_boundary(frame,ring):
    base,raw=exact_v2.boundary(frame,ring);_,old=boundary_v7.extract(frame,ring,base,raw)
    repaired,details=repaired_boundary(frame,base,raw,old)
    return repaired,raw,details


# Bind relocated dependencies after defining functions (including override hooks).
from . import boundary_v7 as boundary_v7
from . import exact_v2 as exact_v2
from .v82_boundary import repair_boundary as repaired_boundary
