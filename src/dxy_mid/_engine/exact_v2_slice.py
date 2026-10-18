"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import cv2 as cv2
import numpy as np
VERSION5 = 'v5-exact-v2-scalar-1.0'

def boundary(frame,ring):
 if not ring.valid:raise ValueError('valid_ring_required')
 p=process_candidates(frame.labels,ring.geometry,frame.source_id);x0,y0,x1,y1=p['crop'];offset=np.array([x0,y0]);candidates=[];trusted=[];excluded=[];unknown=np.zeros(frame.labels.shape,np.uint8);unknown[y0:y1,x0:x1]=cv2.dilate(p['band'],legacy.v2.disk(max(2,.06*p['rr'])))
 for name in ['A','B']:
  cm=p[name];visible,removed,center,extent=EDGE(cm,p['band'],p['center'],p['rr'],'unused',name);native_contour=legacy.v2.contour(cm)+offset;full=np.zeros(frame.labels.shape,np.uint8);full[y0:y1,x0:x1]=cm
  # Full contours are diagnostic only; the exact <=2500 ordered points fit.
  points=legacy.v2.contour(cm);ix,iy=points[:,0].astype(int),points[:,1].astype(int);bad=cv2.dilate(p['band'],legacy.v2.disk(max(2,.06*p['rr'])))[iy,ix]>0
  candidates.append(dict(name=name,mask=full,contour=native_contour,trusted=visible+offset,excluded=points[bad]+offset,points_local=visible,geom_center_local=center,extent=extent,rr_native=p['rr'],crop_offset=offset))
  trusted.append(visible+offset);excluded.append(points[bad]+offset)
 b=BoundaryResult(candidates,np.concatenate(trusted),np.concatenate(excluded),np.concatenate(excluded),unknown,dict(provenance(frame,'historical_V2_AB_exact'),implementation_version=VERSION5));return b,p


# Bind relocated dependencies after defining functions (including override hooks).
from . import components as legacy
from .contracts import BoundaryResult as BoundaryResult
from .contracts import provenance as provenance
from .v2_candidates import process_candidates as process_candidates
from .v2_slice import candidate_fit as EDGE
