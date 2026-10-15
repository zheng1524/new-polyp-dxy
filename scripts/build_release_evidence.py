"""Post-run reporting only: quality tables and existing-result PNGs, no fitting."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--render',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(a.run/'frames.csv');g=pd.read_csv(a.run/'groups.csv');v=pd.read_csv(a.run/'videos.csv')
    failures=[];ring_audit=[]
    for z in f.itertuples():
        with (a.run/'cache'/(z.cache_key+'.pkl')).open('rb') as h:m=pickle.load(h)['measurement']
        d=m.ring.geometry.get('diagnostics',{});attempts=m.ring.geometry.get('attempts',[])
        rec=dict(source_id=z.source_id,group=z.group,stem=z.stem,ring_valid=m.ring.valid,ring_reason=m.ring.reason,
            branch=m.ring.geometry.get('branch',''),specific_reason=d.get('reason',''),
            gate_diagnostics=json.dumps(d,ensure_ascii=False,default=str),attempts=json.dumps(attempts,ensure_ascii=False,default=str),
            filled_area_px=m.audit['ring_filled_area_px'])
        ring_audit.append(rec)
        if not m.ring.valid:failures.append(rec)
    pd.DataFrame(ring_audit).to_csv(a.output/'Ring_audit_790.csv',index=False)
    pd.DataFrame(failures).to_csv(a.output/'Ring_failed_frames.csv',index=False)
    summary=[]
    for group,z in f.groupby('group',sort=True):
        good=z[z.measurement_eligible.astype(bool)];group_value=g[g.group==group].iloc[0];video_values=v[v.group==group]
        summary.append(dict(group=group,n_input=len(z),numeric_frames=len(good),Ring_failed=int((~z.ring_valid).sum()),
            normal_Gate=int(z.default_Gate_pass.sum()),Q_recovery=int(z.Q_distance_recovery.sum()),strict_frames=int(z.stability_review_pass.sum()),
            Q_recovery_fraction=float(good.Q_distance_recovery.mean()) if len(good) else np.nan,
            strict_fraction=float(good.stability_review_pass.mean()) if len(good) else np.nan,
            n_input_videos=int(group_value.n_input_videos),n_valid_videos=int(group_value.n_valid_videos),prediction_mm=group_value.prediction_mm,
            frame_IQR_mm=float(good.D_mm.quantile(.75)-good.D_mm.quantile(.25)) if len(good) else np.nan,
            video_medians_mm=json.dumps(dict(zip(video_values.video,video_values.median_mm))),
            insufficient_repetition=bool(group_value.n_valid_videos<2 or len(good)<5),history_prediction_fallback=0))
    pd.DataFrame(summary).to_csv(a.output/'group_quality_39.csv',index=False)
    # Publish compact numeric audit, not runtime image paths or unreviewed photos.
    cols=['source_id','group','video','stem','L_px','L_mid_px','scale_mm_per_px','D_mm','ring_valid','default_Gate_pass','Q_distance_recovery','stability_review_pass','quality_class','reason','stability_flags','P_count','Q_count','history_prediction_fallback']
    out=f[cols].copy();out.video=out.video.map(lambda x:Path(x).name);out.to_csv(a.output/'recomputed_frames.csv',index=False)
    if a.render:
        from dxy_mid.visualize import render_groups
        render_groups(f,g,v,a.run)
    print('Reported',len(f),'frames,',len(g),'groups,',len(failures),'Ring failures')
