"""Evaluation-only Ring comparison after current-mask measurement completes."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd

def compare_ring(actual_dir,expected_json,output):
    actual_dir=Path(actual_dir);frames=pd.read_csv(actual_dir/'frames.csv');expected=json.loads(Path(expected_json).read_text())
    differences=[];maximum=0.;ids=set(frames.source_id)==set(expected)
    if not ids:differences.append(dict(reason='source_id_set_differs'))
    for z in frames.itertuples():
        with (actual_dir/'cache'/(z.cache_key+'.pkl')).open('rb') as f:ring=pickle.load(f)['measurement'].ring
        old=expected[z.source_id]
        if ring.valid!=old['valid'] or ring.reason!=old['reason'] or ring.geometry.get('branch','')!=old['branch']:
            differences.append(dict(source_id=z.source_id,reason='valid_reason_or_branch_differs'))
        for kind,e in old['ellipses'].items():
            for key,value in e.items():
                new=ring.geometry.get(kind,{}).get(key,np.nan)
                if isinstance(value,(int,float)):
                    if np.isfinite(value) and np.isfinite(new):maximum=max(maximum,abs(value-new))
                    if not np.isclose(value,new,atol=1e-7,rtol=1e-10,equal_nan=True):differences.append(dict(source_id=z.source_id,ellipse=kind,field=key,frozen=value,new=new))
    result=dict(passed=not differences,frames=len(frames),max_abs_parameter_delta=maximum,differences=differences)
    Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual-dir',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare_ring(a.actual_dir,a.expected,a.output);print(json.dumps(r,ensure_ascii=False,indent=2));raise SystemExit(0 if r['passed'] else 1)
