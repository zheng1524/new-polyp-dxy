"""Explicit evaluation after prediction freeze; never imported by measurement."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--groups',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    gt=pd.read_csv(root/'validation/frozen39/GT_mapping.csv')
    if 'GT_mm' not in gt:raise ValueError('GT mapping needs authoritative GT_mm')
    pred=pd.read_csv(a.groups);j=gt.merge(pred[['group','prediction_mm']],on='group',how='left',validate='one_to_one')
    valid=np.isfinite(j.prediction_mm)&np.isfinite(j.GT_mm)&(j.GT_mm>0)
    abs_mm=np.abs(j.loc[valid,'prediction_mm']-j.loc[valid,'GT_mm']);rel=100*abs_mm/j.loc[valid,'GT_mm']
    result=dict(scope='39 silicone-model retrospective groups; exploratory, not clinical generalization',
        target_groups=len(j),numeric_groups=int(valid.sum()),failed_groups=j.loc[~valid,'group'].tolist(),historical_prediction_fallback=0,
        GT_used_to_select_algorithm=False,median_absolute_relative_error_pct=float(np.median(rel)),
        MAE_mm=float(np.mean(abs_mm)),p95_absolute_relative_error_pct=float(np.quantile(rel,.95)),
        max_absolute_relative_error_pct=float(np.max(rel)),within10pct=int((rel<=10).sum()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
