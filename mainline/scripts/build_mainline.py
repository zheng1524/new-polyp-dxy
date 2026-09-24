#!/usr/bin/env python3
"""Freeze S5-CPAG-39 and render its evaluated 39-group error ranking."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path('/home/liu/polyp_research')
OUT=ROOT/'experiments/dxy_mainline_s5_cpag39_20260923'
BASE=ROOT/'experiments/dxy_post_capture_relaxed_area_guard_20260922'
IP88=ROOT/'experiments/dxy_ip88_raw_video_reacquisition_20260923'
for d in ('inputs','tables','figures'):(OUT/d).mkdir(parents=True,exist_ok=True)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(x,p): pd.DataFrame(x).to_csv(OUT/p,index=False)

def metrics(d):
 e=d.prediction_mm-d.reference_mm;a=d.abs_error_pct
 return {'coverage':len(d),'median_absolute_error_pct':a.median(),'MAE_mm':e.abs().mean(),'mean_absolute_relative_error_pct':a.mean(),'median_signed_error_pct':(100*e/d.reference_mm).median(),'within5_pct':100*(a<=5).mean(),'within10_pct':100*(a<=10).mean(),'p90_pct':a.quantile(.9),'p95_pct':a.quantile(.95),'max_pct':a.max()}

def build():
 base=pd.read_csv(BASE/'tables/group_evaluated.csv')
 base=base[base.method.eq('post_capture_relaxed_area_guard')][['group','prediction_mm','reference_mm']].copy()
 ip=pd.read_csv(IP88/'tables/group_prediction_evaluated.csv')[['group','prediction_mm','reference_mm']].copy()
 assert len(base)==38 and len(ip)==1 and not set(base.group)&set(ip.group)
 d=pd.concat([base,ip],ignore_index=True)
 d['method']='S5-CPAG-39';d['error_mm']=d.prediction_mm-d.reference_mm;d['relative_error_pct']=100*d.error_mm/d.reference_mm;d['abs_error_pct']=d.relative_error_pct.abs()
 d['paris_type']=d.group.str.split('_').str[1];d['ring_code']=d.group.str.extract(r'_(R(?:5|10))_')[0]
 d['measurement_branch']='base_S5_CPAG' ;d.loc[d.group.eq(ip.group.iloc[0]),'measurement_branch']='Ip8_raw_video_centerline_inner_adapter'
 d['short_label']=d.paris_type+' '+d.group.str.extract(r'_(\d+(?:\.\d+)?)mm_')[0]+' '+d.ring_code
 d=d.sort_values(['abs_error_pct','group'],ascending=[False,True]).reset_index(drop=True);d.index+=1;d.index.name='rank';d=d.reset_index()
 write(d,'tables/group_errors_descending.csv')
 met=metrics(d);met.update({'mainline_id':'S5-CPAG-39','coverage_text':'39/39','base_38_mainline':'dxy_video_result_sharp5_post_capture_relaxed_area_guard_v1','Ip8_extension':'raw_video_reacquisition_centerline_adapter_post_area_guard_v1'})
 write([met],'tables/metrics_summary.csv')
 manifest={'mainline_id':'S5-CPAG-39','expanded_name':'Sharp5 + Capture-level Post-area Guard, 39-group scope','status':'USER_PROMOTED_FROZEN_MAINLINE','promotion_date':'2026-09-23','scope':'39/39 research Dxy evaluation groups; result/ remains untouched','base_38_groups':{'source':'dxy_video_result_sharp5_post_capture_relaxed_area_guard_v1','rule':'frozen raw result_sharp5 plus post-capture relaxed area guard v1, unmodified V2/scale/B1/fusion'},'Ip8_R10_inclusion':{'group':ip.group.iloc[0],'source':'raw_video_reacquisition_centerline_adapter_post_area_guard_v1','rule':'raw videos 004/005 re-decoded; frozen INIT rerun; only missing-inner frames use current outer+mid centerline-symmetry inner reconstruction; unmodified V2, S0-p2 post-capture guard and B1; two eligible video captures median fused','old_V2_R3_used':False},'not_included':'all_forced_centerline_inner_trial: rejected due 34/39 coverage','final_evaluation':met,'inputs_sha256':{'base_group_evaluated':sha(BASE/'tables/group_evaluated.csv'),'Ip8_group_evaluated':sha(IP88/'tables/group_prediction_evaluated.csv'),'Ip8_blind_output_list':sha(IP88/'inputs/blind_sha256.txt'),'combined_errors':sha(OUT/'tables/group_errors_descending.csv')}}
 (OUT/'MAINLINE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 # Horizontal ranking is more legible for the full group names.
 fig,ax=plt.subplots(figsize=(13,13))
 colors=d.ring_code.map({'R5':'#2a9d8f','R10':'#e76f51'}).fillna('#777777')
 ax.barh(range(len(d)),d.abs_error_pct,color=colors,edgecolor='none')
 ax.set_yticks(range(len(d)));ax.set_yticklabels(d.short_label,fontsize=9);ax.invert_yaxis();ax.set_xlabel('Absolute relative error (%)');ax.set_title('S5-CPAG-39: group error ranking (high to low)')
 ax.axvline(5,color='#555',ls='--',lw=1,label='5%');ax.axvline(10,color='#999',ls='--',lw=1,label='10%')
 for y,(err,branch) in enumerate(zip(d.abs_error_pct,d.measurement_branch)):
  ax.text(err+.22,y,f'{err:.2f}%',va='center',fontsize=8)
  if branch!='base_S5_CPAG':ax.scatter([err],[y],marker='*',s=55,c='#5e3c99',zorder=3)
 ax.legend(title='R5 teal / R10 orange; star = Ip8 video extension',loc='lower right');ax.set_xlim(0,max(d.abs_error_pct)*1.19);fig.tight_layout();fig.savefig(OUT/'figures/group_abs_error_descending.png',dpi=180);plt.close(fig)
 md=['# S5-CPAG-39 组误差排序','',f"coverage **39/39**；median **{met['median_absolute_error_pct']:.3f}%**；MAE **{met['MAE_mm']:.3f} mm**；p95 **{met['p95_pct']:.3f}%**；max **{met['max_pct']:.3f}%**。",'', '图中从上到下为绝对相对误差从高到低。青色 R5，橙色 R10；紫色星号为 Ip8 R10 的原视频中心线内环适配，其余 38 组均来自原冻结 S5-CPAG 主线。','', '![误差排序](figures/group_abs_error_descending.png)','', '- 完整排序 CSV：`tables/group_errors_descending.csv`','- 机器可读定义：`MAINLINE_MANIFEST.json`']
 (OUT/'ERROR_RANKING.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
 print(met)
if __name__=='__main__':build()
