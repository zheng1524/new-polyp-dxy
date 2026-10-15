"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import numpy as np
VERSION5 = 'v5-exact-v2-scalar-1.0'

def geometry(frame,ring,boundary,method='historical_V2_exact',stem=None):
 if method!='historical_V2_exact':raise ValueError(method)
 if boundary.provenance['coordinate_system']!='native_image_px':raise ValueError('boundary_coordinate_mismatch')
 stem=stem or frame.source_id;cs=[];failures=[]
 for c in boundary.candidates:
  points=c['points_local'];center=c['geom_center_local'];extent=c['extent'];rr=c['rr_native'];offset=c['crop_offset'];assert np.array_equal(points+offset,c['trusted'])
  circle=legacy.v2.fit_circle(points,center,extent,rr,stem+'_'+c['name']);ell,local=old_geometry.traced_call(legacy.v2.fit_ellipse,points,center,extent,rr)
  if 'major_diameter_px' in ell:ell.update(cv_d1=float(local['d1']),cv_d2=float(local['d2']))
  for shape,f in [('circle',circle),('ellipse',ell)]:
   required=['cx','cy','radius_px'] if shape=='circle' else ['cx','cy','major_diameter_px','minor_diameter_px']
   if not all(k in f for k in required):failures.append(dict(candidate=c['name'],shape=shape,stage='v2_candidate',reason=f['reason'],n_points=len(points),**{k:v for k,v in f.items() if k not in ['reason','valid']}));continue
   if shape=='circle':a=b=f['radius_px'];angle=0
   else:a,b=f['major_diameter_px']/2,f['minor_diameter_px']/2;angle=f['angle_deg']+(90 if f['cv_d1']<f['cv_d2'] else 0)
   t=np.deg2rad(angle);basis=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])@np.diag([a,b]);q=dict(f)
   q.update(n_points=len(points),center_distance_norm=float(np.linalg.norm(np.array([f['cx'],f['cy']])-center)/max(extent,1e-9)),min_radius_bound=max(.15*rr,.08*extent),max_radius_bound=max(3*rr,1.6*extent),fit_minor_radius=b,fit_major_radius=a,legacy_valid=f['valid'],legacy_reason=f['reason'],fit_coordinate='native_crop_px',crop_offset=offset.tolist(),rng_seed=stem+'_'+c['name'],coordinate_note='only translation crop to native, no affine/stretch or resize')
   prov=dict(provenance(frame,'historical_V2_exact'),implementation_version=VERSION5,fit_coordinate='native_crop_px',shape_coordinate='native_image_px',length_unit='px_diameter')
   cs.append(GeometryCandidate(c['name'],shape,np.array([f['cx'],f['cy']])+offset,basis,float(2*a),q,f,prov))
 return cs,failures


def gate(candidate):return old_gate.run(candidate,'legacy')


# Bind relocated dependencies after defining functions (including override hooks).
from . import components as legacy
from . import gate as old_gate
from . import geometry_component as old_geometry
from .contracts import GeometryCandidate as GeometryCandidate
from .contracts import provenance as provenance
from .exact_v2_slice import boundary as boundary
