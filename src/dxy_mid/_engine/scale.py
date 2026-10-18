"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import math as math
import numpy as np
METHODS = ('mid_affine', 'inner_area', 'mid_major', 'legacy_affine', 'u1_half_affine')

def run(frame,ring,method='mid_affine'):
 if not ring.valid:raise ValueError('scale_requires_valid_ring')
 if method not in METHODS:raise ValueError(method)
 g=ring.geometry;D=frame.inner_diameter_mm
 if method=='mid_affine':
  e=g['centerline'];M,_=build_affine_rectifier(ellipse_from_dict(e),None);physical=D+1.2;kind='centerline';space='midplane_rectified_px';assumption='weak perspective; polyp and ring share plane and depth'
 elif method in ['legacy_affine','u1_half_affine']:
  e=g['inner'];M,_=build_affine_rectifier(ellipse_from_dict(e),ellipse_from_dict(g['outer']));physical=D;kind='inner';space='inner_outer_weighted_rectified_px';assumption='weighted ring-plane affine; coplanar polyp; legacy native-L multiplication excluded'
  if method=='u1_half_affine':M=np.diag([.5,.5,1])@M;space='half_rectified_px'
 else:
  e=g['inner'] if method=='inner_area' else g['centerline'];physical=D if method=='inner_area' else D+1.2;M=np.eye(3);kind='inner' if method=='inner_area' else 'centerline';space='native_image_px';assumption='area-equivalent projection scalar; not direction invariant physical calibration' if method=='inner_area' else 'ring major-axis scalar; no polyp plane rectification; equal magnification assumption'
 if method=='inner_area':diam=2*math.sqrt(e['major_radius']*e['minor_radius'])
 elif method=='mid_major':diam=2*e['major_radius']
 else:
  p=transform_points(sample_ellipse_points(ellipse_from_dict(e)),M);diam=2*np.mean(np.linalg.norm(p-p.mean(0),axis=1))
 if not np.isfinite(diam) or diam<=0:raise ValueError('scale_invalid')
 prov=provenance(frame,method);prov.update(coordinate_system=space,reference_kind=kind,wire_diameter_mm=1.2,length_unit='px_diameter')
 return ScaleResult(M,float(physical/diam),float(diam),physical,space,assumption,prov)


# Bind relocated dependencies after defining functions (including override hooks).
from .contracts import ScaleResult as ScaleResult
from .contracts import provenance as provenance
from .polypseg_single_view_measurement import build_affine_rectifier as build_affine_rectifier
from .polypseg_single_view_measurement import ellipse_from_dict as ellipse_from_dict
from .polypseg_single_view_measurement import sample_ellipse_points as sample_ellipse_points
from .polypseg_single_view_measurement import transform_points as transform_points
