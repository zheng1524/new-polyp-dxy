#!/usr/bin/env python3
"""Raw-video, GT-blind V2 frame-selector experiment. Existing V2 is unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get('DXY_DATA_ROOT', REPO/'data'))
OUT = Path(os.environ.get('DXY_OUTPUT_ROOT', REPO/'outputs/raw_selector'))
AUDIT = Path(os.environ.get('DXY_AUDIT_ROOT', DATA_ROOT/'scale_video_audit'))
PRIOR = Path(os.environ.get('DXY_PRIOR_SELECTOR_ROOT', DATA_ROOT/'prior_selector'))
MODEL_DIR = Path(os.environ.get('DXY_MODEL_DIR', DATA_ROOT/'models/init_segmentation'))
for p in ('inputs','candidates/images','candidates/masks','candidates/ring_fit','tables','gallery/overlays'):
    (OUT/p).mkdir(parents=True,exist_ok=True)
sys.path.insert(0, str(REPO/'src'))
from polypseg.common import load_pretrained_model
from polypseg.ring_fitting import process_mask_with_calibration
from dxy_s5cpag import frame_selection as PREP
from dxy_s5cpag import v2 as v2mod
from dxy_s5cpag import scale as aud
from dxy_s5cpag.segmentation import run_segmentation
PROTOCOL={
 'raw_candidates':'79 formal Dxy videos; remove first/last 5%; 5 bins; reuse prepare script max 24 evenly sampled indices per bin',
 'segmentation':'frozen INIT model 256px, argmax, old.run_segmentation',
 'ring':'current process_mask_with_calibration; no modified thresholds',
 'v2':'unmodified v2.process_view, choose, choose_geometry; R0 valid only; no new rescue',
 'motion':'mean absolute grayscale difference to original raw frame index -1/+1 in mask-union bbox expanded 25%; divide by 255; average two differences',
 'sharpness':'existing prepare._frame_sharpness on cached JPEG',
 'frontality':'existing ring inner ellipse minor_radius/major_radius; proxy only',
 'literature_cascade5':'per bin V2-valid only; lowest-motion ceil(50%); then highest-sharpness ceil(50%); max b/a, tie sharpness then earliest frame',
 'local_medoid5':'per bin V2-valid only; q=L_px/ring_px, choose min abs(log(q/median(q))), tie earliest frame',
 'video_B1':'R_mm*median(L_selected)/median(R_selected); >=4 valid selected and ring p95/p5-1>=.05',
 'fusion':'replace eligible capture existing valid formal video rows with B1; photo and group median unchanged; otherwise S0 fallback',
 'result_sharp5':'same raw candidate pool; highest sharpness per bin before V2 validity, then use only measured selections; unchanged method definition',
 'historical_controls':'S0, fixed1_all and old geometry5_valid read unchanged; non-identical candidate pool flagged; old 20-frame result_sharp5 reported separately as context',
 'GT_used':False,
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(df,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(df).to_csv(path,index=False)


def finite(x):
    try:return math.isfinite(float(x))
    except (TypeError,ValueError):return False


def extract():
    (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(PROTOCOL,ensure_ascii=False,indent=2)+'\n')
    # The dense audit's frozen formal-video universe has 79 captures; only
    # 65 have surviving formal keyframe rows, so the S0 manifest alone is not
    # the video inventory.
    universe=pd.read_csv(AUDIT/'inputs/dense_frame_manifest.csv')
    videos=universe[['video','group']].drop_duplicates().rename(columns={'video':'source_video'})
    assert len(videos)==79
    rows=[]
    for i,r in enumerate(videos.itertuples(),1):
        path=Path(r.source_video)
        cap=cv2.VideoCapture(str(path))
        assert cap.isOpened(),path
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        assert count>0 and fps>0,(path,count,fps)
        for segno,segment in enumerate(PREP._candidate_indices(count,5,.05),1):
            for candno,idx in enumerate(PREP._sample_indices(segment,24),1):
                stem=f'{path.stem}-raw{segno:02d}-{candno:02d}'
                image=OUT/'candidates/images'/f'{stem}.jpg'
                ok=image.exists()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
                    decoded,frame=cap.read()
                    ok=bool(decoded and frame is not None and cv2.imwrite(str(image),frame,[int(cv2.IMWRITE_JPEG_QUALITY),95]))
                rows.append({'group':r.group,'capture':path.stem,'video':str(path),'stem':stem,
                             'segment_id':segno,'candidate_index':candno,'frame_index':idx,
                             'timestamp_sec':idx/fps,'fps':fps,'video_frame_count':count,
                             'image_path':str(image) if ok else '', 'decode_valid':ok,
                             'failure_reason':'' if ok else 'decode_failed'})
        cap.release()
        if i%10==0:print('extract',i,'/',len(videos),flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==len(df[['capture','frame_index']].drop_duplicates())
    write(df,OUT/'inputs/raw_candidate_manifest.csv')
    (OUT/'inputs/extract_sha256.txt').write_text(f'{sha(OUT/"inputs/frozen_protocol.json")}  inputs/frozen_protocol.json\n'
                                              f'{sha(OUT/"inputs/raw_candidate_manifest.csv")}  inputs/raw_candidate_manifest.csv\n')
    print('extracted',len(df),'decoded',df.decode_valid.sum())


def motion_roi(video:Path,idx:int,label:np.ndarray)->tuple[float,str]:
    # The ROI is fixed in image coordinates for the current and adjacent frames.
    ys,xs=np.nonzero((label==1)|(label==2))
    if not len(xs):return math.nan,'empty_mask_roi'
    h,w=label.shape
    pad=int(round(.25*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)))
    x0=max(0,int(xs.min())-pad);x1=min(w,int(xs.max())+pad+1)
    y0=max(0,int(ys.min())-pad);y1=min(h,int(ys.max())+pad+1)
    cap=cv2.VideoCapture(str(video))
    try:
        images=[]
        for j in (idx-1,idx,idx+1):
            if j<0:return math.nan,'adjacent_out_of_range'
            cap.set(cv2.CAP_PROP_POS_FRAMES,j)
            ok,im=cap.read()
            if not ok or im is None:return math.nan,'adjacent_decode_failed'
            images.append(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY))
        a,b,c=images
        score=(np.mean(cv2.absdiff(a,b))+np.mean(cv2.absdiff(b,c)))/(2*255.)
        return float(score),'ok'
    finally:
        cap.release()


def process_candidate(r:dict)->dict:
    base={'group':r['group'],'capture':r['capture'],'video':r['video'],'stem':r['stem'],
          'segment_id':r['segment_id'],'frame_index':r['frame_index'],
          'timestamp_sec':r['timestamp_sec'],'image_path':r['image_path']}
    mask=OUT/'candidates/masks'/f'{r["stem"]}.png'
    ring_json=OUT/'candidates/ring_fit/json'/f'{r["stem"]}.json'
    try:
        if not r['decode_valid']:raise ValueError('decode_failed')
        if not ring_json.exists():
            process_mask_with_calibration(mask,Path(r['image_path']),OUT/'candidates/ring_fit',None)
        payload=json.loads(ring_json.read_text())
        if not payload.get('rings') or not payload['rings'][0].get('inner_ellipse'):
            raise ValueError('missing_inner_ring_fit')
        scale,ring_px,ring,q=aud.current_rectified_scale(payload)
        v,_,_,_=v2mod.process_view({'stem':r['stem'],'group_key':r['group'],
                                    'image_path':r['image_path'],'mask_path':str(mask),
                                    'ring_fit_json':str(ring_json),'rect_scale':scale})
        inner=ring['inner_ellipse']
        major=2*float(inner['major_radius']);minor=2*float(inner['minor_radius'])
        label=cv2.imread(str(mask),cv2.IMREAD_UNCHANGED)
        motion,motion_reason=motion_roi(Path(r['video']),int(r['frame_index']),label)
        image=cv2.imread(str(r['image_path']),cv2.IMREAD_COLOR)
        sharp=PREP._frame_sharpness(image)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hist=np.bincount(gray.ravel(),minlength=256).astype(float);p=hist/hist.sum();p=p[p>0]
        entropy=float(-(p*np.log2(p)).sum())
        base.update(v)
        base.update({'ring_px':ring_px,'mm_per_px':scale,'ring_frontality':minor/major,
                     'ring_inner_residual':q.get('inner_mean_residual',math.nan),
                     'motion':motion,'motion_reason':motion_reason,'sharpness':sharp,
                     'entropy':entropy,'ring_json_path':str(ring_json),'mask_path':str(mask),
                     'processing_failure_reason':''})
    except Exception as exc:
        base.update({'processing_failure_reason':f'{type(exc).__name__}:{exc}'})
    return base


def measure(device:str='cuda',limit:int|None=None):
    import torch
    m=pd.read_csv(OUT/'inputs/raw_candidate_manifest.csv')
    if limit is not None:
        keep=sorted(m.capture.unique())[:limit]
        m=m[m.capture.isin(keep)].copy()
    good=m[m.decode_valid.astype(bool)]
    model=load_pretrained_model(str(MODEL_DIR),torch.device(device),local_files_only=True,strict=True)
    model.eval()
    run_segmentation(model,torch.device(device),[(r.stem,Path(r.image_path)) for r in good.itertuples()],OUT/'candidates/masks',256,.5)
    del model
    if device.startswith('cuda'):torch.cuda.empty_cache()
    checkpoint=OUT/'tables/v2_proposals_checkpoint.csv'
    if checkpoint.exists():
        prev=pd.read_csv(checkpoint)
        done=set(prev.stem)
        rows=prev.to_dict('records')
    else:rows=[];done=set()
    todo=[r._asdict() for r in m.itertuples(index=False) if r.stem not in done]
    ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=3,mp_context=ctx) as pool:
        for i,base in enumerate(pool.map(process_candidate,todo,chunksize=3),1):
            rows.append(base);done.add(base['stem'])
            if i%50==0:
                write(rows,checkpoint)
                print('ring/V2',len(done),'/',len(m),flush=True)
    write(rows,checkpoint)
    assert set(m.stem)==set(pd.DataFrame(rows).stem)
    by=defaultdict(list)
    for z in rows:
        reason=z.get('processing_failure_reason')
        if not isinstance(reason,str) or reason=='':
            by[z['group']].append(z)
    for cohort in by.values():
        v2mod.choose(cohort);v2mod.choose_geometry(cohort)
    for z in rows:
        dxy=z.get('chosen_model_Dxy_mm',math.nan)
        scale=z.get('mm_per_px',math.nan)
        z['valid']=finite(dxy) and finite(scale) and float(scale)>0
        z['Dxy_mm']=dxy if z['valid'] else math.nan
        z['L_px']=float(dxy)/float(scale) if z['valid'] else math.nan
        z['q_L_over_ring']=z['L_px']/z['ring_px'] if z['valid'] and finite(z.get('ring_px')) else math.nan
        reason=z.get('processing_failure_reason')
        z['failure_reason']='' if z['valid'] else (reason if isinstance(reason,str) and reason else z.get('chosen_model_reason','no_v2_geometry'))
    df=pd.DataFrame(rows).sort_values(['capture','frame_index'])
    write(df,OUT/'tables/raw_candidates_v2_blind.csv')
    print('V2 measured',len(df),'valid',df.valid.sum(),flush=True)


def select():
    df=pd.read_csv(OUT/'tables/raw_candidates_v2_blind.csv')
    df['valid']=df.valid.astype(bool)
    # The historical result selector scores image quality before measurement.
    # V2/ring failures therefore still need a sharpness score and remain
    # eligible to be picked (then fail measurement), not silently removed.
    missing=df.sharpness.isna() & df.image_path.notna()
    for idx,r in df[missing].iterrows():
        image=cv2.imread(str(r.image_path),cv2.IMREAD_COLOR)
        if image is not None:df.at[idx,'sharpness']=PREP._frame_sharpness(image)
    write(df,OUT/'tables/raw_candidate_features_blind.csv')
    selected=[];stage=[];caps=[]
    for cap,g0 in df.groupby('capture',sort=True):
        for seg,bin0 in g0.groupby('segment_id',sort=True):
            sharp_pool=bin0[bin0.sharpness.map(finite).astype(bool)].copy()
            if len(sharp_pool):
                sharp=sharp_pool.sort_values(['sharpness','frame_index'],ascending=[False,True]).iloc[0]
                selected.append({'method':'result_sharp5','group':sharp.group,'capture':cap,'segment_id':seg,
                                 'stem':sharp.stem,'frame_index':sharp.frame_index,'timestamp_sec':sharp.timestamp_sec,
                                 'valid':bool(sharp.valid),'L_px':sharp.L_px,'ring_px':sharp.ring_px,
                                 'q_L_over_ring':sharp.q_L_over_ring,'Dxy_mm':sharp.Dxy_mm,
                                 'motion':sharp.motion,'sharpness':sharp.sharpness,'entropy':sharp.entropy,
                                 'ring_frontality':sharp.ring_frontality,'ring_inner_residual':sharp.ring_inner_residual,
                                 'n_v2_valid_bin':int(bin0.valid.sum()),'stage1_kept':math.nan,
                                 'stage2_kept':math.nan,'bin_q_median':math.nan,'medoid_log_distance':math.nan})
            med_pool=bin0[bin0.valid & bin0.q_L_over_ring.map(finite).astype(bool) & (bin0.q_L_over_ring>0)].copy()
            lit_pool=med_pool[med_pool.motion.map(finite).astype(bool) & med_pool.sharpness.map(finite).astype(bool) & med_pool.ring_frontality.map(finite).astype(bool)].copy()
            stage.append({'capture':cap,'segment_id':seg,'n_candidates':len(bin0),'n_v2_valid':int(bin0.valid.sum()),'n_medoid_usable':len(med_pool),'n_literature_usable':len(lit_pool)})
            choices=[];n1=n2=0
            if len(lit_pool):
                n1=math.ceil(.5*len(lit_pool))
                a=lit_pool.sort_values(['motion','frame_index'],ascending=[True,True]).head(n1)
                n2=math.ceil(.5*len(a))
                b=a.sort_values(['sharpness','frame_index'],ascending=[False,True]).head(n2)
                lit=b.sort_values(['ring_frontality','sharpness','frame_index'],ascending=[False,False,True]).iloc[0]
                choices.append(('literature_cascade5',lit))
            qmed=float(med_pool.q_L_over_ring.median()) if len(med_pool) else math.nan
            if len(med_pool):
                med_pool['medoid_distance']=np.abs(np.log(med_pool.q_L_over_ring/qmed))
                med=med_pool.sort_values(['medoid_distance','frame_index'],ascending=[True,True]).iloc[0]
                choices.append(('local_medoid5',med))
            for method,r in choices:
                selected.append({'method':method,'group':r.group,'capture':cap,'segment_id':seg,
                                 'stem':r.stem,'frame_index':r.frame_index,'timestamp_sec':r.timestamp_sec,
                                 'valid':True,
                                 'L_px':r.L_px,'ring_px':r.ring_px,'q_L_over_ring':r.q_L_over_ring,
                                 'Dxy_mm':r.Dxy_mm,'motion':r.motion,'sharpness':r.sharpness,
                                 'entropy':r.entropy,'ring_frontality':r.ring_frontality,
                                 'ring_inner_residual':r.ring_inner_residual,
                                 'n_v2_valid_bin':int(bin0.valid.sum()),'stage1_kept':n1,'stage2_kept':n2,
                                 'bin_q_median':qmed,'medoid_log_distance':abs(math.log(r.q_L_over_ring/qmed))})
    s=pd.DataFrame(selected)
    write(s,OUT/'tables/selected_frames_blind.csv')
    write(stage,OUT/'tables/stage_counts_blind.csv')
    for method in ('literature_cascade5','local_medoid5','result_sharp5'):
        for cap,g0 in df.groupby('capture',sort=True):
            g=s[(s.method==method)&(s.capture==cap)]
            gv=g[g.valid]
            R,L=gv.ring_px.to_numpy(float),gv.L_px.to_numpy(float)
            dyn=np.percentile(R,95)/np.percentile(R,5)-1 if len(R)>=2 else math.nan
            eligible=len(gv)>=4 and finite(dyn) and dyn>=.05
            ring_mm=10. if '_R10_' in g0.group.iloc[0] else 5.
            dxy=ring_mm*np.median(L)/np.median(R) if eligible else math.nan
            caps.append({'method':method,'group':g0.group.iloc[0],'capture':cap,'n_candidates':len(g0),
                         'n_v2_valid':int(g0.valid.sum()),'n_selected':len(g),'n_selected_valid':len(gv),'ring_dynamic':dyn,
                         'eligible':eligible,'prediction_mm':dxy,
                         'fallback_reason':'' if eligible else ('fewer_than_4_valid_bins' if len(gv)<4 else 'ring_range_below_5pct')})
    c=pd.DataFrame(caps)
    write(c,OUT/'tables/capture_predictions_blind.csv')
    formal=pd.read_csv(AUDIT/'inputs/immutable_scale_manifest.csv')
    formal['capture']=formal.capture_id.str.replace('video:','',regex=False)
    formal=formal[formal.current_valid.astype(bool) & formal.current_Dxy_mm.map(finite)].copy()
    group=[];rows=[]
    for method in ('literature_cascade5','local_medoid5','result_sharp5'):
        sub=c[(c.method==method)&c.eligible]
        mp=sub.set_index('capture').prediction_mm.to_dict()
        v=formal.copy()
        replaced=v.source_kind.eq('video_keyframe') & v.capture.isin(mp)
        v['value_mm']=v.current_Dxy_mm.astype(float)
        v.loc[replaced,'value_mm']=v.loc[replaced,'capture'].map(mp)
        v['replaced']=replaced;v['method']=method
        rows.append(v[['method','group','stem','capture','source_kind','value_mm','replaced']])
        for key,z in v.groupby('group'):
            group.append({'method':method,'group':key,'prediction_mm':z.value_mm.median(),
                          'formal_rows':len(z),'video_captures_used':z.loc[z.replaced,'capture'].nunique(),
                          'fallback_video_captures':z.loc[z.source_kind.eq('video_keyframe') & ~z.replaced,'capture'].nunique()})
    write(group,OUT/'tables/group_predictions_blind.csv')
    write(pd.concat(rows),OUT/'tables/integrated_rows_blind.csv')
    behavior=[]
    for cap,g0 in df.groupby('capture',sort=True):
        valid=g0[g0.valid]
        for method in ('literature_cascade5','local_medoid5','result_sharp5'):
            chosen=s[(s.capture==cap)&s.method.eq(method)]
            rec={'method':method,'group':g0.group.iloc[0],'capture':cap,
                 'n_v2_valid':len(valid),'n_selected':len(chosen)}
            for col in ('motion','sharpness','ring_frontality','ring_inner_residual','q_L_over_ring'):
                rec[f'all_valid_median_{col}']=valid[col].median()
                rec[f'selected_median_{col}']=chosen[col].median()
            q=valid.q_L_over_ring.dropna()
            rec['q_mad']=float(np.median(np.abs(q-np.median(q)))) if len(q) else math.nan
            behavior.append(rec)
    write(behavior,OUT/'tables/selector_behavior_blind.csv')
    cor=[]
    for level,g in [('all',df[df.valid]),('R5',df[df.valid & df.group.str.contains('_R5_')]),
                    ('R10',df[df.valid & df.group.str.contains('_R10_')])]:
        for col in ('ring_inner_residual','motion','sharpness'):
            x=g[['ring_frontality',col]].dropna()
            cor.append({'subset':level,'feature':col,'n':len(x),
                        'spearman_r':x.ring_frontality.corr(x[col],method='spearman')})
    write(cor,OUT/'tables/frontality_correlations_blind.csv')
    valid=df[df.valid & df.ring_frontality.map(finite).astype(bool) & df.ring_inner_residual.map(finite).astype(bool)]
    if len(valid):
        bcut=valid.ring_frontality.quantile(.9)
        rcut=valid.ring_inner_residual.quantile(.9)
        high=valid[(valid.ring_frontality>=bcut)&(valid.ring_inner_residual>=rcut)].copy()
        high=high.sort_values(['ring_inner_residual','ring_frontality'],ascending=[False,False])
        write(high[['group','capture','stem','frame_index','ring_frontality','ring_inner_residual','motion','sharpness','L_px','ring_px','q_L_over_ring']],OUT/'tables/high_frontality_bad_ring_audit_blind.csv')
    paths=['inputs/frozen_protocol.json','inputs/raw_candidate_manifest.csv','tables/raw_candidates_v2_blind.csv',
           'tables/raw_candidate_features_blind.csv',
           'tables/selected_frames_blind.csv','tables/stage_counts_blind.csv',
           'tables/capture_predictions_blind.csv','tables/group_predictions_blind.csv','tables/integrated_rows_blind.csv',
           'tables/selector_behavior_blind.csv','tables/frontality_correlations_blind.csv',
           'tables/high_frontality_bad_ring_audit_blind.csv']
    (OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/p)}  {p}\n' for p in paths))
    print('frozen',c.groupby('method').eligible.sum().to_dict(),flush=True)


def metrics(z,method,stratum,level):
    z=z.dropna(subset=['prediction_mm']).copy()
    e=(z.prediction_mm-z.reference_mm)/z.reference_mm*100;a=e.abs()
    return {'method':method,'stratum':stratum,'level':level,'coverage':len(z),
            'median_abs_error_pct':a.median(),'MAE_mm':(z.prediction_mm-z.reference_mm).abs().mean(),
            'mean_abs_relative_error_pct':a.mean(),'median_signed_error_pct':e.median(),
            'within5_pct':100*(a<=5).mean(),'within10_pct':100*(a<=10).mean(),
            'p90_pct':a.quantile(.9),'p95_pct':a.quantile(.95),'max_pct':a.max()}


def evaluate():
    for line in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():
        digest,rel=line.split('  ',1)
        assert sha(OUT/rel)==digest,rel
    current=pd.read_csv(OUT/'tables/group_predictions_blind.csv')
    old=pd.read_csv(PRIOR/'tables/group_predictions_blind.csv')
    old=old[old.method.isin(['fixed1_all','geometry5_valid'])]
    s0=pd.read_csv(AUDIT/'tables/baseline_per_group.csv')[['group','prediction_mm']].copy();s0['method']='S0'
    pred=pd.concat([current[['method','group','prediction_mm']],old[['method','group','prediction_mm']],s0],ignore_index=True)
    refs=pd.read_csv(AUDIT/'tables/baseline_per_group.csv')[['group','reference_mm','reference_source']]
    z=pred.merge(refs,on='group',how='left',validate='many_to_one')
    z['abs_error_pct']=(z.prediction_mm-z.reference_mm).abs()/z.reference_mm*100
    z['abs_error_mm']=(z.prediction_mm-z.reference_mm).abs()
    z['ring_code']=np.where(z.group.str.contains('_R10_'),'R10','R5')
    z['paris_type']=z.group.str.split('_').str[1]
    z['branch']=np.where(z.paris_type.eq('Ip'),'Ip','non-Ip')
    write(z,OUT/'tables/group_evaluated.csv')
    mets=[]
    for method,g in z.groupby('method'):
        mets.append(metrics(g,method,'overall','all'))
        for col in ('ring_code','branch','paris_type'):
            for level,q in g.groupby(col):mets.append(metrics(q,method,col,level))
    write(mets,OUT/'tables/metrics_summary.csv')
    w=z.pivot(index='group',columns='method',values=['prediction_mm','abs_error_pct'])
    pairs=[]
    for group,r in w.iterrows():
        example=z[z.group.eq(group)].iloc[0]
        rec={'group':group,'reference_mm':example.reference_mm,
             'ring_code':example.ring_code,'branch':example.branch}
        for method in pred.method.unique():
            rec[f'{method}_prediction_mm']=r.get(('prediction_mm',method),math.nan)
            rec[f'{method}_error_pct']=r.get(('abs_error_pct',method),math.nan)
        for method in pred.method.unique():
            rec[f'{method}_minus_S0_abs_error_pp']=rec[f'{method}_error_pct']-rec['S0_error_pct']
        rec['literature_minus_medoid_abs_error_pp']=rec['literature_cascade5_error_pct']-rec['local_medoid5_error_pct']
        for method in ('literature_cascade5','local_medoid5','result_sharp5'):
            gp=current[(current.group.eq(group)) & (current.method.eq(method))].iloc[0]
            rec[f'{method}_video_captures_used']=gp.video_captures_used
            rec[f'{method}_fallback_video_captures']=gp.fallback_video_captures
        pairs.append(rec)
    write(pairs,OUT/'tables/group_paired_comparison.csv')
    print(pd.DataFrame(mets).query("stratum == 'overall'").to_string(index=False),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['extract','measure','select','evaluate'])
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--limit-captures',type=int)
    args=ap.parse_args()
    if args.stage=='extract':extract()
    elif args.stage=='measure':measure(args.device,args.limit_captures)
    elif args.stage=='select':select()
    else:evaluate()


if __name__=='__main__':main()
