"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import numpy as np


def canonical_shape_key(record):
 p=np.asarray(record['p']);a,b=np.exp(p[2:4]);theta=float(p[4])
 if b>a:a,b=b,a;theta+=np.pi/2
 theta=theta%np.pi
 if abs(a-b)<.001 or theta<.00005 or np.pi-theta<.00005:theta=0.
 return (*np.round(p[:2],3),round(float(a),3),round(float(b),3),round(theta,4))


# Bind relocated dependencies after defining functions (including override hooks).
