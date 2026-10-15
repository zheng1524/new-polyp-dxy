"""Verified DXY-MID source closure. See docs/SOURCE_PROVENANCE.csv."""
from __future__ import annotations
import hashlib as hashlib


def configs84(frame,base,raw,evidence,current,policy='anchor'):
    seed=int(hashlib.sha256((frame.source_id+'|0.2').encode()).hexdigest()[:8],16);schedule=[('default',.2,seed,None),('tolerance_10',.1,seed,None),('tolerance_35',.35,seed,None),('seed_plus_1',.2,seed+1,None),('Q_pool_cap_24',.2,seed,24)];results={}
    for variant,fraction,rseed,cap in schedule:
        results[variant]=[run_policy(frame,make_data84(frame,base,raw,evidence,body,fraction),[c for c in current if c.name==body],rseed,cap,policy) for body in ['A','B']]
        if policy=='anchor':results[variant]=recover_conflicted_candidates(frame,results[variant])
        if variant=='default' and not any(r['data']['Q'] for r in results[variant]):
            for name,_,_,_ in schedule[1:]:results[name]=results['default']
            break
    return results


# Bind relocated dependencies after defining functions (including override hooks).
from .q_distance_selection import recover_conflicted_candidates as recover_conflicted_candidates
from .q_evidence_policy import run_policy as run_policy
from .v84_boundary import make_data84 as make_data84
