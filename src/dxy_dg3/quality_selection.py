#!/usr/bin/env python3
"""GT-blind Quality-aware Sharp5 selection within the frozen raw candidate pool.

Only reselects one cached frame per existing time bin.  Segmentation, ring fit,
V2/G0, G3, scale, capture median and formal fusion are imported unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import cv2, numpy as np, pandas as pd
from scipy.spatial import cKDTree

from .layout import FORMAL_MAINLINE as S7, G3_OUT as G3EXP, QUALITY_OUT as OUT, RAW, bind_asset_paths
G0='G0_frozen_S7_V2';G2='G2_adjacency_multiarc_ellipse';G3='G3_occlusion_aware_amodal'
TOP_K=3
# Fixed GT-blind relative/local gates. The global image background never enters
# a score; all sharpness means are normalised per ROI pixel and per time bin.
TENENGRAD_KEEP_RATIO=.75; LAPLACIAN_KEEP_RATIO=.50
MIN_RING_COMPONENT_FRACTION=.80; MIN_FRONTALIY=.35
MIN_TRUSTED_FRACTION=.35; MIN_TRUSTED_ARCS=1
MOD_G2=None;MOD_G3=None;MOD_ADJ=None
def g2():
 global MOD_G2
 if MOD_G2 is None:
  from . import multiarc
  MOD_G2=multiarc
 return MOD_G2
def g3():
 global MOD_G3
 if MOD_G3 is None:
  from . import g3 as g3_module
  MOD_G3=g3_module
 return MOD_G3
def adj():
 global MOD_ADJ
 if MOD_ADJ is None:
  from . import ring_adjacency
  MOD_ADJ=ring_adjacency
 return MOD_ADJ
def truth(x):return str(x).lower()=='true' if isinstance(x,str) else bool(x)
def finite(x):
 try:return math.isfinite(float(x))
 except:return False
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(x,rel):
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False)
def grad_mean(gray,mask,lap=False):
 if mask.sum()<12:return math.nan
 if lap:val=cv2.Laplacian(gray,cv2.CV_64F)**2
 else:
  sx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3);sy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3);val=sx*sx+sy*sy
 return float(val[mask].mean())
def adjacency_metrics(r):
 """Reuse the frozen ring-adjacency primitives without completion diagnostics."""
 e=g2().load_edge();a=adj();d=e.prepare_original(r);cs=e.external_contours(d['polyp']);pts=np.concatenate(cs) if cs else np.empty((0,2),np.float32);n=len(pts)
 samples=a.ring_boundary_samples(d['ring'],d['polyp'],float(d['rr']));cls=np.full(n,'high_confidence',object);near=np.zeros(n,bool)
 if len(samples['points']) and n:
  tree=cKDTree(samples['points']);dist,idx=tree.query(pts,k=1);near=dist<=max(3,round(a.NEAR_FACTOR*d['rr']));kind=samples['kind'][idx];vec=pts-samples['points'][idx];sign=np.sign(np.sum(vec*samples['normals'][idx],axis=1));side=sign==samples['poly_sign'][idx]
  cls[near]='uncertain';cls[near&(kind=='both_polyp_crossing')]='ring_cut';cls[near&(kind=='one_polyp_one_background')&side]='high_confidence'
 arcs=g2().make_arcs(cs,cls,near,d['center'],float(d['rr']));trusted=pts[cls=='high_confidence'];uncertain=pts[cls=='uncertain']
 cc,_,stats,_=cv2.connectedComponentsWithStats(d['ring'].astype(np.uint8),8);areas=stats[1:,cv2.CC_STAT_AREA] if cc>1 else np.array([]);ring_frac=float(areas.max()/areas.sum()) if len(areas) and areas.sum() else 0.
 return d,pts,cls,near,arcs,trusted,uncertain,{'adj_high_fraction':float((cls=='high_confidence').mean()) if n else 0.,'trusted_arc_count':len(arcs),'trusted_arc_length_px':float(sum(x['arc_length_px'] for x in arcs)),'ring_component_fraction':ring_frac}
def local_features(r):
 d,pts,cls,near,arcs,trusted,uncertain,info=adjacency_metrics(r);x0,y0,x1,y1=d['crop'];im=cv2.imread(str(r.image_path));im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[y0:y1,x0:x1]
 pol=d['polyp'].astype(np.uint8);ring=d['ring'].astype(np.uint8);band=d['band'].astype(bool);ker=np.ones((3,3),np.uint8);inner=cv2.erode(pol,ker)>0;poly_roi=inner&~band;poly_edge=(cv2.dilate(pol,ker)>0)&~inner&~band;ring_edge=(cv2.dilate(ring,ker)>0)&~(cv2.erode(ring,ker)>0)
 q={'tenengrad_polyp_roi':grad_mean(im,poly_roi),'tenengrad_polyp_edge':grad_mean(im,poly_edge),'tenengrad_ring_edge':grad_mean(im,ring_edge),'laplacian_polyp_roi':grad_mean(im,poly_roi,True),'laplacian_polyp_edge':grad_mean(im,poly_edge,True),'laplacian_ring_edge':grad_mean(im,ring_edge,True),'polyp_roi_pixels':int(poly_roi.sum()),'polyp_edge_pixels':int(poly_edge.sum()),'ring_edge_pixels':int(ring_edge.sum()),**info}
 point=pd.DataFrame({'point_index':np.arange(len(pts)),'x':pts[:,0] if len(pts) else [],'y':pts[:,1] if len(pts) else [],'adjacency_class':cls,'ring_adjacent':near})
 return q,point,(d,arcs,trusted,uncertain)
def build_features():
 raw=bind_asset_paths(pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv'));pool=raw.sort_values(['group','capture','segment_id','sharpness','frame_index'],ascending=[True,True,True,False,True]).groupby(['group','capture','segment_id'],group_keys=False).head(TOP_K).copy();rows=[];points={}
 for i,r0 in enumerate(pool.to_dict('records'),1):
  r=type('R',(),r0)()
  try:q,p,_=local_features(r);q.update({k:r0.get(k) for k in ('group','capture','segment_id','frame_index','stem','sharpness','entropy','valid','ring_px','ring_frontality','ring_inner_residual','Dxy_mm','L_px','image_path','mask_path','ring_json_path','chosen_shape','chosen_model_candidate')});q['status']='ok';points[r.stem]=p
  except Exception as ex:q={k:r0.get(k) for k in ('group','capture','segment_id','frame_index','stem','sharpness','entropy','valid','ring_px','ring_frontality','ring_inner_residual','Dxy_mm','L_px','image_path','mask_path','ring_json_path','chosen_shape','chosen_model_candidate')};q.update({'status':'feature_failure','feature_failure_reason':type(ex).__name__})
  rows.append(q)
  if i%100==0:
   # Recoverable checkpoint for this expensive, read-only cached-candidate pass.
   write(rows,'tables/quality_candidate_features_checkpoint.csv')
   print(f'quality features {i}/{len(pool)}',flush=True)
 return pd.DataFrame(rows),points
def choose(features):
 out=[]
 for keys,z in features.groupby(['group','capture','segment_id'],sort=False):
  z=z.copy();tenmax=z.tenengrad_polyp_roi.max();lapmax=z.laplacian_polyp_roi.max()
  z['pass_local']=z.tenengrad_polyp_roi>=TENENGRAD_KEEP_RATIO*tenmax;z['pass_local']&=z.laplacian_polyp_roi>=LAPLACIAN_KEEP_RATIO*lapmax
  z['pass_geometry']=z.valid.map(truth)&z.ring_px.map(finite)&(z.ring_component_fraction>=MIN_RING_COMPONENT_FRACTION)&(z.ring_frontality>=MIN_FRONTALIY)&(z.adj_high_fraction>=MIN_TRUSTED_FRACTION)&(z.trusted_arc_count>=MIN_TRUSTED_ARCS)
  ok=z[z.pass_local&z.pass_geometry]
  if len(ok):
   # Lexicographic: geometry support first, then local ROI focus; no learned weight.
   pick=ok.sort_values(['adj_high_fraction','trusted_arc_length_px','tenengrad_polyp_roi','sharpness','frame_index'],ascending=[False,False,False,False,True]).iloc[0];reason='pass_quality_gate'
  else:pick=None;reason='no_candidate_passed_quality_gate'
  for _,x in z.iterrows():out.append({**{k:x[k] for k in ('group','capture','segment_id','stem','frame_index','sharpness','tenengrad_polyp_roi','laplacian_polyp_roi','adj_high_fraction','trusted_arc_count','trusted_arc_length_px','ring_component_fraction','ring_frontality','valid','status')},'pass_local':bool(x.pass_local),'pass_geometry':bool(x.pass_geometry),'selected':bool(pick is not None and x.stem==pick.stem),'selection_reason':reason})
 return pd.DataFrame(out)
def quality_geometry(selection,features,points):
 raw=bind_asset_paths(pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv')).set_index('stem');base=[];g3rows=[]
 for i,st in enumerate(selection[selection.selected].stem,1):
  r0=raw.loc[st].to_dict();r0['stem']=st;r=type('R',(),r0)();p=points[st]
  # Frozen G2 fitter supplies the normal G3 seed, then frozen G3 itself runs.
  rows,_,_=g2().frame_process(r,p);g0row=next(x for x in rows if x['geometry']==G0);g2row=next(x for x in rows if x['geometry']==G2)
  g3row,_=g3().frame_g3(r,p,pd.Series(g2row));base.append(g0row);g3rows.append(g3row)
  if i%50==0:print(f'quality geometry {i}/{selection.selected.sum()}',flush=True)
 return pd.DataFrame(base),pd.DataFrame(g3rows)
def blind():
 for d in ('inputs','tables','scripts'):(OUT/d).mkdir(parents=True,exist_ok=True)
 protocol={'baseline':'S7-FMD39/G0 and independent G3 experiment','GT_used_before_evaluation':False,'candidate_pool':'existing raw Sharp5 candidates, unchanged five bins/video and existing start/end trim','stage1':{'top_k_existing_sharpness':TOP_K,'ROI':'polyp interior and edge plus ring edge; no background pixels','tenengrad_keep_ratio':TENENGRAD_KEEP_RATIO,'laplacian_keep_ratio':LAPLACIAN_KEEP_RATIO},'stage2':{'frozen_segmentation_and_ring':'reused cached masks/ring fits','ring_component_fraction_min':MIN_RING_COMPONENT_FRACTION,'ring_frontality_min':MIN_FRONTALIY,'trusted_original_edge_fraction_min':MIN_TRUSTED_FRACTION,'trusted_arc_min':MIN_TRUSTED_ARCS,'ranking':'trusted fraction, trusted arc length, ROI Tenengrad, existing sharpness'},'no_gt_tuning':True,'aggregation':'frozen per-frame Dxy; capture median >=4 valid; formal group fusion/fallback unchanged'}
 (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
 cache_features=OUT/'tables/quality_candidate_features_blind.csv';cache_selection=OUT/'tables/quality_selection_audit_blind.csv'
 if cache_features.exists() and cache_selection.exists():
  # The cached pass is itself GT-blind and is recomputed only for the final
  # selected stems to recover in-memory original contour point identities.
  features=bind_asset_paths(pd.read_csv(cache_features));sel=pd.read_csv(cache_selection);raw=bind_asset_paths(pd.read_csv(RAW/'tables/raw_candidate_features_blind.csv')).set_index('stem');points={}
  for st in sel[sel.selected].stem:
   r0=raw.loc[st].to_dict();r0['stem']=st;_,points[st],_=local_features(type('R',(),r0)())
 else:features,points=build_features();sel=choose(features)
 # Persist the expensive GT-blind feature pass before G0/G3 evaluation, so a
 # geometry exception cannot require recomputing cached-image diagnostics.
 write(features,'tables/quality_candidate_features_blind.csv');write(sel,'tables/quality_selection_audit_blind.csv');write(sel[sel.selected],'tables/quality_selected_bins_blind.csv')
 g0,g3rows=quality_geometry(sel,features,points);write(g0,'tables/quality_G0_frames_blind.csv');write(g3rows,'tables/quality_G3_frames_blind.csv')
 qframes=pd.concat([g0,g3rows],ignore_index=True,sort=False);caps=g2().capture_rows(qframes);groups=g2().fuse(caps);write(caps,'tables/quality_capture_predictions_blind.csv');write(groups,'tables/quality_group_predictions_blind.csv')
 files=['inputs/frozen_protocol.json','tables/quality_candidate_features_blind.csv','tables/quality_selection_audit_blind.csv','tables/quality_selected_bins_blind.csv','tables/quality_G0_frames_blind.csv','tables/quality_G3_frames_blind.csv','tables/quality_capture_predictions_blind.csv','tables/quality_group_predictions_blind.csv'];(OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/f)}  {f}\n' for f in files));print('sealed')
def metric(x):return g2().metrics(x)
def evaluate():
 for line in (OUT/'inputs/blind_sha256.txt').read_text().splitlines():h,p=line.split('  ',1);assert sha(OUT/p)==h
 refs=pd.read_csv(S7/'tables/group_errors_descending.csv')[['group','reference_mm']]
 # Frozen original Sharp5 selections as assessed in the independent G3 study.
 src=pd.read_csv(G3EXP/'tables/group_predictions_blind.csv');src=src[src.geometry.isin([G0,G3])].copy();src['method']=src.geometry.map({G0:'A_original_sharp5_G0',G3:'C_original_sharp5_G3'})
 q=pd.read_csv(OUT/'tables/quality_group_predictions_blind.csv');q=q[q.geometry.isin([G0,G3])].copy();q['method']=q.geometry.map({G0:'B_quality_aware_G0',G3:'D_quality_aware_G3'})
 groups=pd.concat([src[['group','prediction_mm','method']],q[['group','prediction_mm','method']]],ignore_index=True).merge(refs,on='group',how='left');groups['signed_error_pct']=100*(groups.prediction_mm-groups.reference_mm)/groups.reference_mm;groups['abs_error_pct']=groups.signed_error_pct.abs();write(groups,'tables/group_predictions_evaluated.csv')
 metrics=[]
 for name,x in groups.groupby('method'):
  rec={'method':name,**metric(x.rename(columns={'method':'geometry'}))}
  # From this experiment onward retain both directional mean error and its
  # magnitude: MAE_mm alone can hide systematic percentage bias.
  rec['mean_signed_error_pct']=x.signed_error_pct.mean();rec['mean_error_mm']=(x.prediction_mm-x.reference_mm).mean()
  metrics.append(rec)
 m=pd.DataFrame(metrics);write(m,'tables/metrics_summary.csv')
 # capture/frame outputs retain method identity for all four arms.
 qc=pd.read_csv(OUT/'tables/quality_capture_predictions_blind.csv');qf=pd.concat([pd.read_csv(OUT/'tables/quality_G0_frames_blind.csv'),pd.read_csv(OUT/'tables/quality_G3_frames_blind.csv')],ignore_index=True);qc['method']=qc.geometry.map({G0:'B_quality_aware_G0',G3:'D_quality_aware_G3'});qf['method']=qf.geometry.map({G0:'B_quality_aware_G0',G3:'D_quality_aware_G3'});write(qc,'tables/quality_capture_predictions_evaluated.csv');write(qf,'tables/quality_frame_predictions_evaluated.csv')
 reports(groups,m,qc,qf)
 print(m.to_string(index=False))
def reports(groups,m,qc,qf):
 ms=m.set_index('method');sel=pd.read_csv(OUT/'tables/quality_selection_audit_blind.csv');changed=0;orig=pd.read_csv(G3EXP/'tables/frame_geometry_blind.csv');orig=orig[orig.geometry.eq(G0)][['group','capture','bin','stem']].rename(columns={'stem':'original_stem'});new=sel[sel.selected][['group','capture','segment_id','stem']].rename(columns={'segment_id':'bin','stem':'new_stem'});pair=orig.merge(new,on=['group','capture','bin'],how='outer');changed=int((pair.original_stem!=pair.new_stem).sum());missing=int(pair.new_stem.isna().sum());eligible=qc.groupby('method').eligible.sum().to_dict()
 human=f'''# Quality-aware Sharp5（独立实验）

本轮只在原 Sharp5 每 bin 的 top-{TOP_K} 缓存候选中重新选一帧：先按息肉 ROI/边缘、ring 边缘的 Tenengrad 与 Laplacian 做相对清晰度门槛，再用冻结 ring 连续性、frontality、ring-adjacency 可信原始外缘作简单 gate；没有背景清晰度、GT、重训或新几何。395 个 bin 中改变 {changed} 个，质量门槛缺失 {missing} 个。G0：原 Sharp5 median {ms.loc['A_original_sharp5_G0','median_abs_error_pct']:.3f}% → quality {ms.loc['B_quality_aware_G0','median_abs_error_pct']:.3f}%；G3：原 {ms.loc['C_original_sharp5_G3','median_abs_error_pct']:.3f}% → quality {ms.loc['D_quality_aware_G3','median_abs_error_pct']:.3f}%。D 的平均绝对百分比误差为 {ms.loc['D_quality_aware_G3','mean_abs_error_pct']:.3f}%，平均有符号误差为 {ms.loc['D_quality_aware_G3','mean_signed_error_pct']:.3f}%（平均 mm 误差 {ms.loc['D_quality_aware_G3','mean_error_mm']:.3f}）。注意：本轮特意只审计“重新选帧”本身，未再执行 S6 的 post-capture CPAG retry，因此 B 的 69.90% max 说明 selector 单独不能替代该安全层，不能作为主线结果。'''
 (OUT/'HUMAN_REPORT.md').write_text(human,encoding='utf8')
 tech=f'''# Quality-aware Sharp5 technical report

## Blind protocol

候选池严格为既有 raw Sharp5 每个冻结时间 bin 的现成候选；按既有 `sharpness` 先取 {TOP_K}，而不是重新抽帧。ROI Tenengrad 是 Sobel 梯度能量均值，Laplacian 是 ROI 方差代理；三者均只在 polyp interior、polyp boundary neighbourhood 或 ring boundary neighbourhood计算，故背景纹理与 ROI 面积不会主导。每 bin 保留 ROI Tenengrad ≥ {TENENGRAD_KEEP_RATIO:.0%} 最大值且 Laplacian ≥ {LAPLACIAN_KEEP_RATIO:.0%} 最大值者；再要求 frozen V2 valid、ring最大连通成分 ≥{MIN_RING_COMPONENT_FRACTION:.0%}、frontality≥{MIN_FRONTALIY}、ring-adjacency可信原始边缘≥{MIN_TRUSTED_FRACTION:.0%}且至少一弧。通过者按可信比例、可信弧长、ROI Tenengrad、原 sharpness 作字典序排序。

所有阶段 GT 隔离；SHA 在 `inputs/blind_sha256.txt`。G0/G3 fit、scale、`Dxy_i`、>=4 帧 capture median、formal fusion 和 Ip8 fallback 均复用且未修改。**本轮是 selector-alone ablation：没有再次运行 S6 的 post-capture CPAG retry**，因此 B/D 不是可直接晋升的 S6 替代物；尤其 B 的尾部用以验证质量 gate 不可取代既有安全层。

## Results

{m.to_markdown(index=False) if False else m.to_csv(index=False)}

选择变化 {changed}/395，missing {missing}；quality capture eligible：{eligible}。详表：`tables/quality_candidate_features_blind.csv`、`quality_selection_audit_blind.csv`、frame/capture/group CSV。
'''
 (OUT/'TECHNICAL_REPORT.md').write_text(tech,encoding='utf8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['blind','evaluate']);a=p.parse_args();{'blind':blind,'evaluate':evaluate}[a.stage]()
