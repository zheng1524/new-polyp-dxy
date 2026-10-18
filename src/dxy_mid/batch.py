"""Portable mask-manifest runner. Input columns are projected, never predictions."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import pickle
import hashlib
import os
import pandas as pd
from .pipeline import measure_files, sha256, VERSION
from . import _engine

REQUIRED=('source_id','group','video','stem','image_path','mask_path','ring_mm')
OPTIONAL=('image_sha256','mask_sha256','frame_index','timestamp_sec','cohort')

def source_fingerprint():
    root=Path(__file__).parent
    h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def load_manifest(path,data_root=None):
    path=Path(path);base=Path(data_root) if data_root else path.resolve().parent
    data=pd.read_csv(path,usecols=lambda c:c in REQUIRED+OPTIONAL,keep_default_na=False)
    missing=set(REQUIRED)-set(data.columns)
    if missing:raise ValueError('Missing input columns: '+','.join(sorted(missing)))
    if not len(data) or data.source_id.duplicated().any():raise ValueError('Empty or duplicate source_id')
    rows=data.to_dict('records')
    for row in rows:
        for key in ('image_path','mask_path'):
            p=Path(row[key]);row[key]=str((base/p).resolve() if not p.is_absolute() else p)
            digest=sha256(row[key]);declared=row.get(key.replace('_path','_sha256'))
            if declared and digest!=declared:raise ValueError('Input SHA256 mismatch: '+row['source_id']+' '+key)
            row[key.replace('_path','_sha256')]=digest
    return rows


def _job(row,output,fingerprint):
    # Only own-route cache, keyed by current content, configuration and source.
    # No pickle is ever accepted from a manifest or arbitrary external path.
    key=hashlib.sha256(json.dumps(dict(image=row['image_sha256'],mask=row['mask_sha256'],
        ring_mm=float(row['ring_mm']),source_id=row['source_id'],stem=row['stem'],
        engine=fingerprint,version=VERSION),sort_keys=True).encode()).hexdigest()
    output=Path(output);cache=output/'cache'/f'{key}.pkl'
    if cache.exists():
        with cache.open('rb') as f:result=pickle.load(f)
        if result['key']!=key:raise ValueError('Own cache key mismatch')
        result=result['measurement']
    else:
        result=measure_files(row['image_path'],row['mask_path'],inner_diameter_mm=row['ring_mm'],
            source_id=row['source_id'],stem=row['stem'])
        tmp=cache.with_suffix('.tmp')
        with tmp.open('wb') as f:pickle.dump(dict(key=key,measurement=result),f,pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache)
    record=dict(result.audit,**{k:v for k,v in row.items() if k not in ('ring_mm',)})
    record.update(D_mm=record['D_frame_mm'],default_Gate_pass=record['normal_Gate_pass'],
        Q_distance_recovery=record['Q_conflict_recovery'],numeric_valid=record['measurement_eligible'],
        cache_key=key,source_fingerprint=fingerprint)
    return record


def run_manifest(manifest,output,*,data_root=None,workers=4,render=False):
    """Recompute all masks, reuse only this output directory's hashed results."""
    if not 1<=workers<=8:raise ValueError('workers must be 1..8; BLAS/OpenMP=1 recommended')
    rows=load_manifest(manifest,data_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'cache').mkdir(exist_ok=True);fingerprint=source_fingerprint();records=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(_job,row,output,fingerprint) for row in rows]
        for future in as_completed(futures):
            records.append(future.result())
            if len(records)%10==0 or len(records)==len(rows):print(f'Measured {len(records)}/{len(rows)}',flush=True)
    frames=pd.DataFrame(records).sort_values(['group','video','source_id'])
    frames.to_csv(output/'frames.csv',index=False)
    groups,videos=_engine.fusion(frames.to_dict('records'))
    strict_groups,strict_videos=_engine.fusion(frames.to_dict('records'),True)
    groups.to_csv(output/'groups.csv',index=False);videos.to_csv(output/'videos.csv',index=False)
    strict_groups.to_csv(output/'groups_strict.csv',index=False);strict_videos.to_csv(output/'videos_strict.csv',index=False)
    (output/'run.json').write_text(json.dumps(dict(version=VERSION,manifest_sha256=sha256(manifest),
        source_fingerprint=fingerprint,frames=len(frames),groups=len(groups),
        eligible=int(frames.measurement_eligible.sum()),ring_failures=int((~frames.ring_valid).sum()),
        normal_Gate=int(frames.normal_Gate_pass.sum()),Q_recovery=int(frames.Q_conflict_recovery.sum()),
        strict=int(frames.strict_stable.sum()),history_prediction_fallback=0,GT_read=False,
        workers=workers,input_columns=list(REQUIRED+OPTIONAL)),indent=2))
    if render:
        from .visualize import render_groups
        render_groups(frames,groups,videos,output)
    return frames,groups,videos
