#!/usr/bin/env python3
"""GT-free ring-adjacency classification of ORIGINAL polyp contour points.

This is a review-only companion to completion classification.  It does not
fit geometry, modify segmentation, or add morphology-created contour points.
"""
from __future__ import annotations
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .layout import EDGE_OUT as OUT
EDGE=None
NEAR_FACTOR=.12          # fixed ring-normalized contour-to-ring assignment
SAMPLE_RADIUS_FACTOR=.025
SAMPLE_FORWARD_FACTOR=.035
MAX_RING_CROSS_FACTOR=.75 # fixed maximum ray traverse; not a fitted quantity
POLYP_FRACTION=.25       # fixed local-neighborhood label threshold

def load_edge():
 global EDGE
 if EDGE is None:
  from . import edge_completion
  EDGE=edge_completion
 return EDGE
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(x,rel):
 p=OUT/rel;p.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False)
def sample_fraction(mask,xy,radius):
 x,y=xy; x0=max(0,int(round(x-radius)));x1=min(mask.shape[1],int(round(x+radius))+1);y0=max(0,int(round(y-radius)));y1=min(mask.shape[0],int(round(y+radius))+1)
 if x0>=x1 or y0>=y1:return math.nan
 yy,xx=np.ogrid[y0:y1,x0:x1];disk=(xx-x)**2+(yy-y)**2<=radius**2
 return float(mask[y0:y1,x0:x1][disk].mean()) if disk.any() else math.nan
def ray_beyond_ring(ring,p,vec,maxdist):
 """Starting from a ring boundary pixel, cross the ring along a local normal.

 The returned coordinate is just beyond the first ring run.  Thus the two
 sides classify the two physical regions separated by that ring segment.
 """
 h,w=ring.shape; seen=False
 for dist in range(0,maxdist+1):
  x=int(round(p[0]+vec[0]*dist));y=int(round(p[1]+vec[1]*dist))
  if x<0 or x>=w or y<0 or y>=h:return None
  if ring[y,x]>0:seen=True;continue
  if seen:return np.array([float(x),float(y)])
 return None
def ring_boundary_samples(ring,polyp,rr):
 """Classify each original ring boundary sample by the two regions it separates."""
 cs,_=cv2.findContours(ring.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
 points=[]; normals=[]; kinds=[]; poly_sign=[]; contour_ids=[]
 radius=max(1,round(SAMPLE_RADIUS_FACTOR*rr)); forward=max(1,round(SAMPLE_FORWARD_FACTOR*rr)); maxcross=max(4,round(MAX_RING_CROSS_FACTOR*rr))
 for cid,c in enumerate(cs):
  pts=c.reshape(-1,2).astype(float);n=len(pts)
  if n<3:continue
  for i,p in enumerate(pts):
   tangent=pts[(i+1)%n]-pts[(i-1)%n];norm=np.linalg.norm(tangent)
   if norm<1e-6:continue
   normal=np.array([-tangent[1],tangent[0]])/norm
   side=[]
   for sign in (1,-1):
    q=ray_beyond_ring(ring,p,normal*sign,maxcross)
    if q is None: side.append(('U',math.nan))
    else:
     # a fixed small push beyond the ring avoids sampling the ring boundary.
     q=q+normal*sign*forward;frac=sample_fraction(polyp,q,radius)
     side.append(('P' if frac>=POLYP_FRACTION else 'B',frac))
   labels=[x[0] for x in side]
   if labels==['P','P']:kind='both_polyp_crossing'; ps=0
   elif labels.count('P')==1 and labels.count('B')==1:kind='one_polyp_one_background';ps=1 if labels[0]=='P' else -1
   else:kind='ambiguous';ps=0
   points.append(p);normals.append(normal);kinds.append(kind);poly_sign.append(ps);contour_ids.append(cid)
 if not points:return {'points':np.empty((0,2)), 'normals':np.empty((0,2)), 'kind':np.array([],object),'poly_sign':np.array([],int),'contour_id':np.array([],int)}
 return {'points':np.asarray(points),'normals':np.asarray(normals),'kind':np.asarray(kinds,object),'poly_sign':np.asarray(poly_sign,int),'contour_id':np.asarray(contour_ids,int)}
def contiguous_segments(samples,stem,group,capture,bin_):
 """Consecutive boundary samples of same identity: diagnostic segments only."""
 out=[]
 for cid in np.unique(samples['contour_id']):
  ii=np.where(samples['contour_id']==cid)[0];labs=samples['kind'][ii]
  if not len(ii):continue
  # ring contours are cyclic; split every run and merge first/last if needed.
  starts=np.where(labs!=np.roll(labs,1))[0]
  if len(starts)==0:starts=np.array([0])
  runs=[]
  for start in starts:
   k=[];j=int(start)
   while not k or labs[j]==labs[start]:
    k.append(j);j=(j+1)%len(ii)
    if j==start:break
   runs.append(np.asarray(k))
  # unique runs after cyclic handling
  seen=set();sid=0
  for run in runs:
   key=tuple(sorted(ii[run].tolist()))
   if key in seen:continue
   seen.add(key);sid+=1;ids=ii[run];p=samples['points'][ids]
   length=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()) if len(p)>1 else 0.
   out.append({'stem':stem,'group':group,'capture':capture,'bin':bin_,'ring_contour_id':int(cid),'segment_id':f'R{cid}_{sid}','adjacency_type':samples['kind'][ids[0]],'sample_count':len(ids),'length_px':length,'centroid_x':float(p[:,0].mean()),'centroid_y':float(p[:,1].mean())})
 return out
def classify_row(r, completion):
 edge=load_edge();d=edge.prepare_original(r);contours=edge.external_contours(d['polyp']);pts=np.concatenate(contours) if contours else np.empty((0,2),np.float32)
 n=len(pts);samples=ring_boundary_samples(d['ring'],d['polyp'],float(d['rr']))
 cls=np.full(n,'high_confidence',object);kind=np.full(n,'far_from_ring',object);near=np.zeros(n,bool);side_match=np.zeros(n,bool)
 if len(samples['points']) and n:
  tree=cKDTree(samples['points']);dist,idx=tree.query(pts,k=1);limit=max(3,round(NEAR_FACTOR*d['rr']));near=dist<=limit
  kind[near]=samples['kind'][idx[near]]
  vec=pts-samples['points'][idx];sign=np.sign(np.sum(vec*samples['normals'][idx],axis=1));expected=samples['poly_sign'][idx]
  side_match=sign==expected
  cls[near]='uncertain'
  cls[near&(samples['kind'][idx]=='both_polyp_crossing')]='ring_cut'
  cls[near&(samples['kind'][idx]=='one_polyp_one_background')&side_match]='high_confidence'
 comp=completion.set_index('point_index').reindex(np.arange(n))
 # Strong identity assertion: comparison is exact original-contour point pairing.
 if len(comp)!=n or (n and not np.allclose(comp.x.to_numpy(float),pts[:,0])):raise AssertionError('completion point order mismatch')
 compcls=comp.edge_class.to_numpy(object)
 agree=cls==compcls
 row={'stem':r.stem,'group':r.group,'capture':r.capture,'bin':int(r.segment_id),'frame_index':r.frame_index,'status':'ok','ring_px':r.ring_px,'ring_radius_diagnostic_px':float(d['rr']),'original_contour_points':n,'ring_adjacent_contour_fraction':float(near.mean()) if n else math.nan,'adj_high_confidence_fraction':float((cls=='high_confidence').mean()) if n else math.nan,'adj_uncertain_fraction':float((cls=='uncertain').mean()) if n else math.nan,'adj_ring_cut_fraction':float((cls=='ring_cut').mean()) if n else math.nan,'completion_high_confidence_fraction':float((compcls=='high_confidence').mean()) if n else math.nan,'completion_ring_cut_fraction':float((compcls=='ring_cut').mean()) if n else math.nan,'method_disagreement_fraction':float((~agree).mean()) if n else math.nan,'completion_high_to_adjacency_cut_fraction':float(((compcls=='high_confidence')&(cls=='ring_cut')).mean()) if n else math.nan,'completion_cut_to_adjacency_high_fraction':float(((compcls=='ring_cut')&(cls=='high_confidence')).mean()) if n else math.nan,'ring_boundary_both_polyp_fraction':float((samples['kind']=='both_polyp_crossing').mean()) if len(samples['kind']) else math.nan,'ring_boundary_one_polyp_fraction':float((samples['kind']=='one_polyp_one_background').mean()) if len(samples['kind']) else math.nan,'ring_boundary_ambiguous_fraction':float((samples['kind']=='ambiguous').mean()) if len(samples['kind']) else math.nan}
 point=pd.DataFrame({'point_index':np.arange(n),'x':pts[:,0] if n else [],'y':pts[:,1] if n else [],'adjacency_class':cls,'nearest_ring_adjacency':kind,'ring_adjacent':near,'one_polyp_side_match':side_match,'completion_class':compcls,'methods_agree':agree})
 for k,v in row.items():point[k]=v
 segments=contiguous_segments(samples,r.stem,r.group,r.capture,int(r.segment_id))
 return row,point,segments,{'d':d,'pts':pts,'adj':cls,'completion':compcls,'near':near,'samples':samples}
def blend(im,m,c,a=.35):
 out=im.copy();m=m.astype(bool);out[m]=(out[m]*(1-a)+np.asarray(c)*a).astype(np.uint8);return out
def dots(im,pts,sel,c,rad=1):
 out=im.copy()
 for x,y in pts[sel]:cv2.circle(out,(round(x),round(y)),rad,c,-1,lineType=cv2.LINE_AA)
 return out
def card(row,aux):
 d,pts,adj,comp,near=aux['d'],aux['pts'],aux['adj'],aux['completion'],aux['near'];x0,y0,x1,y1=d['crop'];im=cv2.imread(str(row.image_path));im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)[y0:y1,x0:x1]
 raw=blend(blend(blend(im,d['polyp'],(20,210,60),.30),d['ring'],(255,210,0),.55),d['band'],(190,40,220),.18)
 original=dots(im,pts,np.ones(len(pts),bool),(0,230,255),1)
 def classified(base,cl):
  o=blend(blend(blend(base,d['polyp'],(20,210,60),.16),d['ring'],(255,210,0),.48),d['band'],(190,40,220),.15)
  o=dots(o,pts,cl=='high_confidence',(30,230,60),2);o=dots(o,pts,cl=='uncertain',(255,170,0),2);o=dots(o,pts,cl=='ring_cut',(235,45,45),2);return o
 panels=[('Raw mask: green polyp / gold ring / purple unknown',raw),('Original complete contour: cyan',original),('Completion classification',classified(im,comp)),('Ring-adjacency classification',classified(im,adj))]
 h=max(x.shape[0] for _,x in panels);out=[]
 for title,p in panels:
  if p.shape[0]<h:p=cv2.copyMakeBorder(p,0,h-p.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
  cv2.putText(p,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),2,cv2.LINE_AA);cv2.putText(p,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA);out.append(p)
 whole=np.hstack(out);maxw=1800
 if whole.shape[1]>maxw:whole=cv2.resize(whole,(maxw,round(whole.shape[0]*maxw/whole.shape[1])),interpolation=cv2.INTER_AREA)
 p=OUT/'gallery/ring_adjacency_assets'/f'{row.stem}.jpg';p.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(p),cv2.cvtColor(whole,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])
def picks(df):
 specs=[('两种方法分歧最大','method_disagreement_fraction',False),('completion 高可信、邻接判 cut','completion_high_to_adjacency_cut_fraction',False),('completion cut、邻接高可信','completion_cut_to_adjacency_high_fraction',False),('ring 两侧都接 polyp 最多','ring_boundary_both_polyp_fraction',False),('单侧 polyp/背景 最多','ring_boundary_one_polyp_fraction',False)]
 out=[];used=set()
 for label,col,asc in specs:
  for stem in df.sort_values(col,ascending=asc).stem:
   if stem not in used:out.append((label,stem));used.add(stem);break
 return out
def html_page(df,priority):
 rec=df[['stem','group','capture','bin','adj_high_confidence_fraction','adj_uncertain_fraction','adj_ring_cut_fraction','method_disagreement_fraction']].to_dict('records')
 links=''.join(f'<li><a href="#{html.escape(s)}">{html.escape(label)}：{html.escape(s)}</a></li>' for label,s in priority)
 cards=[]
 for r in df.itertuples(index=False):
  cards.append(f'''<details id="{html.escape(r.stem)}"><summary><b>{html.escape(r.group)}</b> · {html.escape(r.capture)} · bin {r.bin} | adjacency: high {r.adj_high_confidence_fraction:.1%}, uncertain {r.adj_uncertain_fraction:.1%}, cut {r.adj_ring_cut_fraction:.1%}; disagreement {r.method_disagreement_fraction:.1%}</summary><img loading="lazy" src="ring_adjacency_assets/{html.escape(r.stem)}.jpg"><div class="review" data-stem="{html.escape(r.stem)}"><button data-label="分类正确">分类正确</button><button data-label="邻接法恢复过多">邻接法恢复过多</button><button data-label="邻接法删除过多">邻接法删除过多</button><button data-label="两法都不可信">两法都不可信</button><button data-label="不确定">不确定</button><input placeholder="一句备注"><span>未标注</span></div></details>''')
 page=f'''<!doctype html><meta charset="utf-8"><title>Ring adjacency edge review</title><style>body{{font:14px system-ui;margin:22px;max-width:1850px}}img{{width:100%;background:#111;margin:8px 0}}details{{border:1px solid #ccd;margin:8px 0;padding:8px}}summary{{cursor:pointer;line-height:1.6}}button{{margin:3px;padding:5px 8px}}input{{width:280px;padding:5px}}.chosen{{background:#d9f4d8}}.legend{{background:#f4f4f4;padding:10px}}</style><h1>Ring 邻接规则 vs completion：原始 polyp contour 人工审查</h1><p>沿现有 ring mask 边界的局部法向，穿过 ring 后检查两侧小邻域：两侧皆 polyp→红色 cut；仅一侧 polyp、且 contour 位于该侧→绿色可信；其余→橙色 uncertain。所有彩色点均是原始 mask contour；没有拟合、GT 或新生成边缘。</p><div class="legend">绿色=可信外缘；红色=不可信/ring-cut；橙色=uncertain；金=ring；紫=unknown。第三栏为上一版 completion，第四栏为本轮 ring-adjacency。</div><p><a href="../tables/manual_ring_adjacency_review.csv" download>下载可编辑审查 CSV 模板</a> · 按钮仅暂存浏览器，下载后保存。</p><button id="download">下载当前人工标签 CSV</button><h2>优先看两法不同的 case</h2><ol>{links}</ol><h2>全部 {len(df)} 个 selected frames</h2>{''.join(cards)}<script>const rows={json.dumps(rec,ensure_ascii=False)},labels={{}};document.querySelectorAll('.review').forEach(b=>{{const s=b.dataset.stem,span=b.querySelector('span'),inp=b.querySelector('input');b.querySelectorAll('button').forEach(x=>x.onclick=()=>{{labels[s]={{label:x.dataset.label,comment:inp.value||''}};b.querySelectorAll('button').forEach(q=>q.classList.remove('chosen'));x.classList.add('chosen');span.textContent=x.dataset.label;}});inp.oninput=()=>{{if(labels[s])labels[s].comment=inp.value;}};}});document.querySelector('#download').onclick=()=>{{const head=['group','capture','bin','stem','adj_high_confidence_fraction','adj_uncertain_fraction','adj_ring_cut_fraction','method_disagreement_fraction','manual_label','comment'],q=x=>'"'+String(x??'').replaceAll('"','""')+'"';let a=[head.join(',')];rows.forEach(r=>{{let z=labels[r.stem]||{{}};a.push([r.group,r.capture,r.bin,r.stem,r.adj_high_confidence_fraction,r.adj_uncertain_fraction,r.adj_ring_cut_fraction,r.method_disagreement_fraction,z.label||'',z.comment||''].map(q).join(','));}});let e=document.createElement('a');e.href=URL.createObjectURL(new Blob([a.join('\\n')],{{type:'text/csv;charset=utf-8'}}));e.download='manual_ring_adjacency_review.csv';e.click();}});</script>'''
 (OUT/'gallery/ring_adjacency_review.html').write_text(page,encoding='utf8')
def main():
 for d in ('tables','gallery','gallery/ring_adjacency_assets','inputs'):(OUT/d).mkdir(parents=True,exist_ok=True)
 protocol={'purpose':'review-only simplified ring-side adjacency classification of original polyp contour','GT_used':False,'fitting_used':False,'original_contour_only':True,'rule':{'both_ring_sides_polyp':'ring_cut','one_side_polyp_one_background':'high_confidence only for original contour on polyp side','other':'uncertain'},'fixed_ring_normalized_constants':{'contour_near_factor':NEAR_FACTOR,'sample_disk_radius_factor':SAMPLE_RADIUS_FACTOR,'sample_forward_factor':SAMPLE_FORWARD_FACTOR,'max_cross_ring_factor':MAX_RING_CROSS_FACTOR,'polyp_fraction_threshold':POLYP_FRACTION}}
 (OUT/'inputs/ring_adjacency_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 edge=load_edge();sel=edge.selected();completion=pd.read_csv(OUT/'tables/original_contour_point_classification.csv')
 rows=[];points=[];segments=[]
 for i,r in enumerate(sel.itertuples(index=False),1):
  c=completion[completion.stem.eq(r.stem)]
  row,p,s,aux=classify_row(r,c);rows.append(row);points.append(p);segments.extend(s);card(r,aux)
  if i%25==0:print(f'{i}/395',flush=True)
 frame=pd.DataFrame(rows).sort_values(['group','capture','bin']);pt=pd.concat(points,ignore_index=True);seg=pd.DataFrame(segments)
 save(frame,'tables/ring_adjacency_classification.csv');save(pt,'tables/original_contour_point_ring_adjacency.csv');save(seg,'tables/ring_adjacency_segments.csv')
 review=frame[['group','capture','bin','stem','adj_high_confidence_fraction','adj_uncertain_fraction','adj_ring_cut_fraction','method_disagreement_fraction']].copy();review.columns=['group','capture','bin','stem','auto_high_confidence_fraction','auto_uncertain_fraction','auto_ring_cut_fraction','method_disagreement_fraction'];review['manual_label']='';review['comment']='';save(review,'tables/manual_ring_adjacency_review.csv')
 pri=picks(frame);html_page(frame,pri)
 files=['inputs/ring_adjacency_protocol.json','tables/ring_adjacency_classification.csv','tables/original_contour_point_ring_adjacency.csv','tables/ring_adjacency_segments.csv','tables/manual_ring_adjacency_review.csv']
 (OUT/'inputs/ring_adjacency_sha256.txt').write_text(''.join(f'{sha(OUT/f)}  {f}\n' for f in files),encoding='utf8')
 print('complete',len(frame),len(pt),len(seg))
if __name__=='__main__':main()
