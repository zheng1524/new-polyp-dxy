"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations

INTERFACE_VERSION = 'V7-Geometry-verified-audit-1.1'

def run(frame,data,current_V2_candidates,method='B',seed=0,q_cap=None):
 """No files/GT/prediction access; caller supplies frozen native-pixel evidence."""
 if method not in ['A','B','C']:raise ValueError('unknown_geometry_method')
 result=normalise_audit(fit(frame,data,current_V2_candidates,method,seed,q_cap))
 result['candidates']=[to_candidate(frame,data,r,method) for r in result['records'] if not r['origin'].startswith('identifiability_')] if result['records'] else list(current_V2_candidates)
 result['diagnostic_candidates']=[to_candidate(frame,data,r,method) for r in result['records'] if r['origin'].startswith('identifiability_')]
 return result


def normalise_audit(result):
 # Some early rejection branches have no cached counts; traces remain complete.
 records=result['records']
 result['actual_sampled_P_indices']=sorted({t['P_index'] for r in records for t in r['trace'] if t['kind']=='P'})
 result['actual_sampled_Q_indices']=sorted({t['Q_index'] for r in records for t in r['trace'] if t['kind'].startswith('Q')})
 result['near_distinct_geometric']=len({canonical_shape_key(r) for r in result['near'] if not r['origin'].startswith('identifiability_')})
 result['interface_version']=INTERFACE_VERSION
 return result


# Bind relocated dependencies after defining functions (including override hooks).
from .geometry_competition import fit as fit
from .geometry_competition import to_candidate as to_candidate
from .geometry_identity import canonical_shape_key as canonical_shape_key
