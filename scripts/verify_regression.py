"""Evaluation only: compare independent recomputation against frozen V9.1.

This module is not imported by the measurement pipeline. No GT needed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

NUMERIC=['L_px','scale_mm_per_px','L_mid_px','D_mm','axis_x1','axis_y1','axis_x2','axis_y2','P_count','Q_count']
FLAGS=['ring_valid','default_Gate_pass','Q_distance_recovery','stability_review_pass','measurement_eligible']
TEXT=['ring_reason','quality_class','reason','stability_flags']

def boolean(series):
    x=series.map(lambda v:str(v).lower())
    if not x.isin(['true','false','1','0']).all():raise ValueError('Missing/invalid boolean; never bool(NaN)')
    return x.isin(['true','1'])

def independent_aggregate(frame):
    # Deliberately independent arithmetic, not production Fusion.
    valid=boolean(frame.measurement_eligible)&np.isfinite(frame.D_mm)&(frame.D_mm>0)
    videos=frame.loc[valid].groupby(['group','video'],sort=True).D_mm.median()
    groups=videos.groupby(level='group').median()
    return videos,groups

def compare(actual,expected,output):
    actual=pd.read_csv(actual);expected=pd.read_csv(expected)
    if actual.source_id.duplicated().any() or expected.source_id.duplicated().any():raise ValueError('Duplicate source ID')
    # A run containing only hard failures has never produced a candidate and
    # legitimately has no candidate-axis columns. Normalize these *evaluation*
    # columns to NaN, only when no frame is measurement-eligible. Never invent
    # axes or suppress missing numerical fields for a successful measurement.
    for data in (actual,expected):
        for key in ('axis_x1','axis_y1','axis_x2','axis_y2'):
            if key not in data:
                if boolean(data.measurement_eligible).any():
                    raise ValueError('Eligible measurement missing candidate axis: '+key)
                data[key]=np.nan
    # Filesystem paths are opaque legacy video labels, not numerical inputs.
    for d in (actual,expected):
        mapping=d[['group','video']].drop_duplicates().copy()
        mapping['portable']=mapping.video.map(lambda x:Path(x).name)
        if mapping.duplicated(['group','portable']).any():raise ValueError('Video basename collision')
        d['video']=d.video.map(lambda x:Path(x).name)
    ids=set(actual.source_id)==set(expected.source_id)
    checks=[dict(check='identical_input_source_ids',pass_=ids)]
    if not ids:
        result=dict(passed=False,checks=checks,missing=list(set(expected.source_id)-set(actual.source_id)),extra=list(set(actual.source_id)-set(expected.source_id)))
        Path(output).write_text(json.dumps(result,indent=2));return result
    j=actual.set_index('source_id').sort_index().join(expected.set_index('source_id').sort_index(),lsuffix='_new',rsuffix='_frozen')
    details=pd.DataFrame(index=j.index)
    for key in NUMERIC:
        x,y=j[key+'_new'].to_numpy(float),j[key+'_frozen'].to_numpy(float)
        ok=np.isclose(x,y,atol=1e-7,rtol=1e-10,equal_nan=True)
        delta=np.abs(x-y);finite=delta[np.isfinite(delta)]
        checks.append(dict(check=key,pass_=bool(ok.all()),mismatches=int((~ok).sum()),max_abs_delta=float(finite.max()) if len(finite) else None))
        details[key+'_delta']=x-y;details[key+'_match']=ok
    for key in FLAGS:
        ok=boolean(j[key+'_new'])==boolean(j[key+'_frozen'])
        checks.append(dict(check=key,pass_=bool(ok.all()),mismatches=int((~ok).sum())))
        details[key+'_match']=ok
    for key in TEXT:
        ok=j[key+'_new'].fillna('')==j[key+'_frozen'].fillna('')
        checks.append(dict(check=key,pass_=bool(ok.all()),mismatches=int((~ok).sum())))
        details[key+'_match']=ok
    for newkey,oldkey in [('body','selected_body'),('shape','selected_shape')]:
        if newkey in actual and oldkey in expected:
            ok=actual.set_index('source_id')[newkey].fillna('').sort_index()==expected.set_index('source_id')[oldkey].fillna('').sort_index()
            checks.append(dict(check=newkey,pass_=bool(ok.all()),mismatches=int((~ok).sum())))
    for name in ['group','video']:
        ok=j[name+'_new']==j[name+'_frozen'];checks.append(dict(check=name+'_mapping',pass_=bool(ok.all())))
    va,ga=independent_aggregate(actual);ve,ge=independent_aggregate(expected)
    for name,x,y in [('video_medians',va,ve),('group_medians',ga,ge)]:
        ok=x.index.equals(y.index) and np.allclose(x,y,atol=1e-7,rtol=1e-10,equal_nan=True)
        checks.append(dict(check=name,pass_=bool(ok),n=len(y),max_abs_delta=float(np.max(np.abs(x-y))) if len(x)==len(y) and len(x) else None))
        pd.DataFrame(dict(recomputed=x,frozen=y,delta=x-y)).to_csv(Path(output).with_name(name+'_comparison.csv'))
    if 'history_prediction_fallback' in actual:
        checks.append(dict(check='zero_historical_fallback',pass_=bool((actual.history_prediction_fallback==0).all())))
    result=dict(passed=all(c['pass_'] for c in checks),frames=len(actual),groups=len(ge),videos=len(ve),checks=checks,
        actual_counts=dict(numeric=int(boolean(actual.measurement_eligible).sum()),ring_failed=int((~boolean(actual.ring_valid)).sum()),
            normal=int(boolean(actual.default_Gate_pass).sum()),Q_recovery=int(boolean(actual.Q_distance_recovery).sum()),strict=int(boolean(actual.stability_review_pass).sum())))
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2));details.to_csv(Path(output).with_name('frame_regression.csv'))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--actual',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);r=compare(a.actual,a.expected,a.output);print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['passed'] else 1)
