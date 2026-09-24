#!/usr/bin/env python3
"""Capture-level post-selection relaxed area safety audit.

Raw result_sharp5 is finished first.  Only an entire raw capture whose median
support across its already selected valid frames is in the formal-S0 p2 tail is
opened for repair; then only its individually failing frames are retried.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import pandas as pd

ROOT=Path('/home/liu/polyp_research')
RAW=ROOT/'experiments/dxy_raw_video_frame_selection_20260921'
AREA=ROOT/'experiments/dxy_area_guarded_sharp5_20260922'
AUDIT=ROOT/'experiments/dxy_scale_video_audit_20260919'
OUT=ROOT/'experiments/dxy_post_capture_relaxed_area_guard_20260922'
for d in ('inputs','tables'):(OUT/d).mkdir(parents=True,exist_ok=True)
PROTOCOL={
 'sequence':'frozen raw result_sharp5 selection first; post-check at capture completion, before unchanged B1 aggregation',
 'threshold':'global formal S0/R0 p2 for C and P, calculated without GT',
 'capture_trigger':'at least four original V2-valid/support-assessed selected bins and median(C)<p2(C) OR median(P)<p2(P)',
 'repair':'only triggered capture, and only its individual raw frames below either p2 threshold: retry first frozen V2-valid candidate by existing sharpness passing both p2 thresholds',
 'unchanged':'all non-triggered captures retain raw selected frames byte-for-byte in value; INIT/V2/ring/L/scale/sharpness/B1/capture replacement/group fusion unchanged',
 'gt_used':False,
}
def truth(x):return x.strip().lower() in ('true','1','yes') if isinstance(x,str) else bool(x)
def finite(x):
 try:return math.isfinite(float(x))
 except (TypeError,ValueError):return False
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(x,p):pd.DataFrame(x).to_csv(OUT/p,index=False)

def blind():
 (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(PROTOCOL,ensure_ascii=False,indent=2)+'\n')
 formal=pd.read_csv(AREA/'tables/formal_s0_area_support_blind.csv'); formal=formal[formal.support_assessed.map(truth)]
 th={'quantile':.02,'C':float(formal.mask_coverage_C.quantile(.02)),'P':float(formal.fit_occupancy_P.quantile(.02)),'formal_assessed_n':len(formal)}
 (OUT/'inputs/relaxed_thresholds_blind.json').write_text(json.dumps(th,indent=2)+'\n')
 sup=pd.read_csv(AREA/'tables/frame_area_support.csv');sup=sup[sup.support_assessed.map(truth)].copy()
 scols=['stem','mask_coverage_C','fit_occupancy_P','support_assessed','chosen_model_candidate','chosen_shape']
 sel=pd.read_csv(RAW/'tables/selected_frames_blind.csv');sel=sel[sel.method.eq('result_sharp5')].merge(sup[scols],on='stem',how='left',validate='one_to_one')
 sel['raw_valid']=sel.valid.map(truth);sel['support_assessed']=sel.support_assessed.fillna(False).map(truth)
 usable=sel[sel.raw_valid&sel.support_assessed]
 caps=usable.groupby(['group','capture']).agg(original_supported_bins=('stem','size'),raw_median_C=('mask_coverage_C','median'),raw_median_P=('fit_occupancy_P','median')).reset_index()
 caps['capture_trigger']=caps.original_supported_bins.ge(4)&((caps.raw_median_C<th['C'])|(caps.raw_median_P<th['P']))
 # Include raw captures without any valid support explicitly, but do not invent a
 # group-level area decision when B1 itself cannot use four valid frames.
 allcaps=sel[['group','capture']].drop_duplicates().merge(caps,on=['group','capture'],how='left')
 allcaps[['original_supported_bins']]=allcaps[['original_supported_bins']].fillna(0)
 allcaps['capture_trigger']=allcaps.capture_trigger.fillna(False).map(truth)
 write(allcaps,'tables/capture_postcheck_blind.csv')
 trigger=allcaps.set_index('capture').capture_trigger.to_dict()
 cmeta=allcaps.set_index('capture')[['original_supported_bins','raw_median_C','raw_median_P']].to_dict('index')
 feats=pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv');feats['valid']=feats.valid.map(truth)
 rows=[]
 for r in sel.itertuples(index=False):
  bad=r.raw_valid and r.support_assessed and (r.mask_coverage_C<th['C'] or r.fit_occupancy_P<th['P'])
  cap_trigger=bool(trigger.get(r.capture,False)); repair=cap_trigger and bad
  rec={'method':'post_capture_relaxed_area_guard','group':r.group,'capture':r.capture,'segment_id':r.segment_id,
   'raw_stem':r.stem,'raw_valid':r.raw_valid,'raw_C':r.mask_coverage_C,'raw_P':r.fit_occupancy_P,'raw_Dxy_mm':r.Dxy_mm,
   'threshold_C':th['C'],'threshold_P':th['P'],'capture_trigger':cap_trigger,'individual_extreme':bad,
   'raw_capture_supported_bins':cmeta.get(r.capture,{}).get('original_supported_bins',0),
   'raw_capture_median_C':cmeta.get(r.capture,{}).get('raw_median_C',math.nan),'raw_capture_median_P':cmeta.get(r.capture,{}).get('raw_median_P',math.nan)}
  if not repair:
   rec.update({'stem':r.stem,'valid':r.raw_valid,'selection_outcome':'raw_kept','retry_rank':math.nan,'C':r.mask_coverage_C,'P':r.fit_occupancy_P,'L_px':r.L_px,'ring_px':r.ring_px,'Dxy_mm':r.Dxy_mm})
  else:
   pool=feats[feats.capture.eq(r.capture)&feats.segment_id.eq(r.segment_id)&feats.valid].merge(sup[scols],on='stem',how='left',validate='one_to_one')
   pool=pool[pool.support_assessed.map(truth)].sort_values(['sharpness','frame_index'],ascending=[False,True]).reset_index(drop=True)
   good=pool[(pool.mask_coverage_C>=th['C'])&(pool.fit_occupancy_P>=th['P'])]
   if len(good):
    q=good.iloc[0];rec.update({'stem':q.stem,'valid':True,'selection_outcome':'post_retry','retry_rank':int(q.name)+1,'C':q.mask_coverage_C,'P':q.fit_occupancy_P,'L_px':q.L_px,'ring_px':q.ring_px,'Dxy_mm':q.Dxy_mm})
   else:rec.update({'stem':'','valid':False,'selection_outcome':'post_missing','retry_rank':math.nan,'C':math.nan,'P':math.nan,'L_px':math.nan,'ring_px':math.nan,'Dxy_mm':math.nan})
  rows.append(rec)
 frame=pd.DataFrame(rows);write(frame,'tables/post_selected_frames_blind.csv')
 cp=[]
 for cap,z in frame.groupby('capture',sort=True):
  good=z[z.valid.map(truth)];R=good.ring_px.to_numpy(float);L=good.L_px.to_numpy(float);dyn=pd.Series(R).quantile(.95)/pd.Series(R).quantile(.05)-1 if len(R)>=2 else math.nan
  eligible=len(good)>=4 and finite(dyn) and dyn>=.05;ring=10. if '_R10_' in z.group.iloc[0] else 5.
  cp.append({'method':'post_capture_relaxed_area_guard','group':z.group.iloc[0],'capture':cap,'n_bins':len(z),'capture_trigger':bool(z.capture_trigger.iloc[0]),'n_post_retry':int(z.selection_outcome.eq('post_retry').sum()),'n_post_missing':int(z.selection_outcome.eq('post_missing').sum()),'n_selected_valid':len(good),'ring_dynamic':dyn,'eligible':eligible,'prediction_mm':ring*pd.Series(L).median()/pd.Series(R).median() if eligible else math.nan,'fallback_reason':'' if eligible else ('fewer_than_4_valid_bins' if len(good)<4 else 'ring_range_below_5pct')})
 cp=pd.DataFrame(cp);write(cp,'tables/capture_predictions_blind.csv')
 f=pd.read_csv(AUDIT/'inputs/immutable_scale_manifest.csv');f['capture']=f.capture_id.str.replace('video:','',regex=False);f=f[f.current_valid.map(truth)&f.current_Dxy_mm.map(finite)].copy()
 mp=cp[cp.eligible].set_index('capture').prediction_mm.to_dict();f['prediction_mm']=f.current_Dxy_mm.astype(float);f['replaced']=f.source_kind.eq('video_keyframe')&f.capture.isin(mp);f.loc[f.replaced,'prediction_mm']=f.loc[f.replaced,'capture'].map(mp)
 gp=[]
 for group,z in f.groupby('group'):gp.append({'method':'post_capture_relaxed_area_guard','group':group,'prediction_mm':z.prediction_mm.median(),'formal_rows':len(z),'video_captures_used':z.loc[z.replaced,'capture'].nunique(),'fallback_video_captures':z.loc[z.source_kind.eq('video_keyframe')&~z.replaced,'capture'].nunique()})
 write(gp,'tables/group_predictions_blind.csv');write(f[['group','stem','capture','source_kind','current_Dxy_mm','prediction_mm','replaced']],'tables/integrated_rows_blind.csv')
 paths=['inputs/frozen_protocol.json','inputs/relaxed_thresholds_blind.json','tables/capture_postcheck_blind.csv','tables/post_selected_frames_blind.csv','tables/capture_predictions_blind.csv','tables/group_predictions_blind.csv','tables/integrated_rows_blind.csv']
 (OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/p)}  {p}\n' for p in paths))
 print('threshold',th,'capture triggers',int(allcaps.capture_trigger.sum()),'frame outcomes',frame.selection_outcome.value_counts().to_dict())

def metrics(d,m):
 d=d.dropna(subset=['prediction_mm']);e=(d.prediction_mm-d.reference_mm)/d.reference_mm*100;a=e.abs()
 return {'method':m,'coverage':len(d),'median_abs_error_pct':a.median(),'MAE_mm':(d.prediction_mm-d.reference_mm).abs().mean(),'mean_abs_relative_error_pct':a.mean(),'median_signed_error_pct':e.median(),'within5_pct':100*(a<=5).mean(),'within10_pct':100*(a<=10).mean(),'p90_pct':a.quantile(.9),'p95_pct':a.quantile(.95),'max_pct':a.max()}
def evaluate():
 for line in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(OUT/p)==h,p
 old=pd.read_csv(RAW/'tables/group_evaluated.csv');s0=old[old.method.eq('S0')][['group','prediction_mm','reference_mm']].copy();s0['method']='S0';raw=old[old.method.eq('result_sharp5')][['group','prediction_mm','reference_mm']].copy();raw['method']='result_sharp5_raw'
 new=pd.read_csv(OUT/'tables/group_predictions_blind.csv').merge(s0[['group','reference_mm']],on='group',how='left',validate='one_to_one');z=pd.concat([s0,raw,new[['method','group','prediction_mm','reference_mm']]],ignore_index=True);z['abs_error_pct']=(z.prediction_mm-z.reference_mm).abs()/z.reference_mm*100;z['abs_error_mm']=(z.prediction_mm-z.reference_mm).abs();write(z,'tables/group_evaluated.csv');write([metrics(q,m) for m,q in z.groupby('method')],'tables/metrics_summary.csv')
 w=z.pivot(index='group',columns='method',values=['prediction_mm','abs_error_pct']);fr=pd.read_csv(OUT/'tables/post_selected_frames_blind.csv');summ=fr.groupby('group').agg(trigger_captures=('capture_trigger','max'),retry_bins=('selection_outcome',lambda x:int((x=='post_retry').sum())),missing_bins=('selection_outcome',lambda x:int((x=='post_missing').sum())))
 rows=[]
 for g,r in w.iterrows():
  q=summ.loc[g] if g in summ.index else {}
  rows.append({'group':g,'ring_code':'R10' if '_R10_' in g else 'R5','morphology':g.split('_')[1],'reference_mm':z[(z.group==g)&(z.method=='S0')].reference_mm.iloc[0],'S0_prediction_mm':r[('prediction_mm','S0')],'S0_error_pct':r[('abs_error_pct','S0')],'raw_prediction_mm':r[('prediction_mm','result_sharp5_raw')],'raw_error_pct':r[('abs_error_pct','result_sharp5_raw')],'post_prediction_mm':r[('prediction_mm','post_capture_relaxed_area_guard')],'post_error_pct':r[('abs_error_pct','post_capture_relaxed_area_guard')],'post_minus_raw_abs_error_pp':r[('abs_error_pct','post_capture_relaxed_area_guard')]-r[('abs_error_pct','result_sharp5_raw')],'trigger_capture':q.get('trigger_captures',False),'retry_bins':q.get('retry_bins',0),'missing_bins':q.get('missing_bins',0)})
 write(rows,'tables/group_paired.csv');print(pd.read_csv(OUT/'tables/metrics_summary.csv').to_string(index=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['blind','evaluate']);a=p.parse_args();blind() if a.stage=='blind' else evaluate()
