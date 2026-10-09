#!/usr/bin/env python3
"""GT-free, original-contour-only ring-edge identity audit.

Completion is diagnostic only: every classified/arc point is an unmodified
pixel on the original INIT polyp-mask external contour.  This program never
calls V2 circle or ellipse fitting.
"""
from __future__ import annotations
import csv
import hashlib
import html
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .layout import EDGE_OUT as OUT, RAW, S6, bind_asset_paths
SCALES = (.05, .10, .15, .20, .25)  # exactly the pre-existing diagnostic set
BOUNDARY_TOL = 1.5
INTERIOR_TOL = 2.0
MAJORITY = 3  # fixed majority of five scales; no GT is read anywhere
SHORT_COMPONENT_POINTS = 12

V2 = None
REC = None

def load_tools():
    global V2, REC
    if V2 is None:
        from . import boundary_recovery as REC_MODULE
        from . import frozen_v2 as V2_MODULE
        V2, REC = V2_MODULE, REC_MODULE
    return V2, REC

def truth(x): return str(x).lower() == 'true' if isinstance(x, str) else bool(x)
def point_values(im, pts):
    x = np.clip(np.rint(pts[:, 0]).astype(int), 0, im.shape[1]-1)
    y = np.clip(np.rint(pts[:, 1]).astype(int), 0, im.shape[0]-1)
    return im[y, x]
def sha(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def save_csv(x, rel):
    p = OUT / rel; p.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(x).to_csv(p, index=False)

def selected() -> pd.DataFrame:
    s = pd.read_csv(S6 / 'tables/selected_bins_blind.csv')
    s = s[s.method.eq('S6_NDG_FCP39')][['stem', 'group', 'capture', 'segment_id', 'selection_outcome']]
    raw = bind_asset_paths(pd.read_csv(RAW / 'tables/raw_candidate_features_blind.csv'))
    cols = ['stem', 'frame_index', 'sharpness', 'ring_px', 'image_path', 'mask_path', 'ring_json_path',
            'chosen_model_candidate', 'chosen_shape', 'valid']
    out = s.merge(raw[cols], on='stem', how='left', validate='one_to_one')
    # Some frozen selected stems were V2-invalid and therefore have null paths
    # in the old feature row.  Their already-created raw candidate assets still
    # exist; filling only these path pointers does not run segmentation/ring fit.
    cache = RAW / 'candidates'
    for col, sub, ext in [('image_path','images','.jpg'), ('mask_path','masks','.png'), ('ring_json_path','ring_fit/json','.json')]:
        fallback = out.stem.map(lambda stem: str(cache / sub / f'{stem}{ext}') if (cache / sub / f'{stem}{ext}').exists() else np.nan)
        out[col] = out[col].where(out[col].notna(), fallback)
    assert len(out) == 395 and out.image_path.notna().all() and out.mask_path.notna().all() and out.ring_json_path.notna().all()
    return out

def prepare_original(r):
    """Reproduce only V2's non-fitting preparation, from raw mask/ring JSON."""
    v2, _ = load_tools()
    original = cv2.imread(str(r.mask_path), cv2.IMREAD_UNCHANGED)
    if original is None: raise FileNotFoundError(r.mask_path)
    with open(r.ring_json_path) as f: js = json.load(f)
    rings = js.get('rings', [])
    yy, xx = np.nonzero((original == 1) | (original == 2))
    if not len(xx): raise ValueError('empty_raw_polyp_and_ring_mask')
    pad = 40; x0 = max(0, int(xx.min())-pad); x1 = min(original.shape[1], int(xx.max())+pad+1)
    y0 = max(0, int(yy.min())-pad); y1 = min(original.shape[0], int(yy.max())+pad+1)
    polyp = (original[y0:y1, x0:x1] == 1).astype(np.uint8)
    ring = (original[y0:y1, x0:x1] == 2).astype(np.uint8)
    if not rings:
        return {'polyp': polyp, 'ring': ring, 'crop': (x0,y0,x1,y1), 'has_ring': False}
    # Some selected frames have a stored ring segmentation but no stored inner
    # ellipse (they were V2-invalid).  Do not refit a ring: use the existing
    # ring-mask area only to normalize the already-fixed diagnostic scales.
    # These rows remain explicitly marked as mask-only fallback in the table.
    if not isinstance(rings[0].get('inner_ellipse'), dict):
        ys, xs = np.nonzero(ring)
        if not len(xs):
            return {'polyp': polyp, 'ring': ring, 'crop': (x0,y0,x1,y1), 'has_ring': False}
        rr = math.sqrt(len(xs)/math.pi)
        center = np.array([float(xs.mean()), float(ys.mean())])
        band = cv2.dilate(ring, v2.disk(max(3,.08*rr))).astype(bool)
        guard = cv2.dilate(band.astype(np.uint8), v2.disk(max(2,.06*rr))).astype(bool)
        return {'polyp':polyp, 'ring':ring, 'band':band, 'guard':guard, 'rr':rr, 'center':center,
                'A':np.zeros_like(polyp,bool), 'B':np.zeros_like(polyp,bool), 'crop':(x0,y0,x1,y1),
                'has_ring':True, 'ring_geometry_mode':'mask_equivalent_radius_fallback_no_inner_ellipse'}
    def local(e): return {**e, 'cx': e['cx']-x0, 'cy': e['cy']-y0}
    inner, outer = local(rings[0]['inner_ellipse']), local(rings[0]['outer_ellipse'])
    rr = math.sqrt(inner['major_radius'] * inner['minor_radius'])
    inside = v2.ellipse_mask(polyp.shape, inner).astype(np.uint8)
    outerfill = v2.ellipse_mask(polyp.shape, outer).astype(np.uint8)
    outside = (1-outerfill).astype(np.uint8)
    band = cv2.dilate(ring, v2.disk(max(3, .08*rr))).astype(np.uint8)
    band = np.maximum(band, ((1-inside)&(1-outside)).astype(np.uint8))
    # Reconstruct A/B only to show precisely what frozen V2 retained/deleted;
    # component_records does not fit a circle or ellipse.
    comps, _ = v2.component_records(polyp, inside, outside, band, np.array([inner['cx'],inner['cy']]), rr)
    broad = [(d, cm) for d, cm, *_ in comps if d['region']=='outside' and d['broad_descriptor'] and d['visible_contour_points']>=12]
    substantial = [(d, cm) for d, cm, *_ in comps if d['visible_contour_points']>=12]
    A = np.zeros_like(polyp)
    for _, cm in (broad if broad else substantial): A = np.maximum(A, cm)
    allmask = np.zeros_like(polyp)
    for _, cm in substantial: allmask = np.maximum(allmask, cm)
    B = cv2.morphologyEx(allmask, cv2.MORPH_OPEN, v2.disk(max(2,.12*rr)))
    guard = cv2.dilate(band, v2.disk(max(2,.06*rr))).astype(bool)
    return {'polyp': polyp, 'ring': ring, 'band': band.astype(bool), 'guard': guard, 'rr': rr,
            'center': np.array([inner['cx'],inner['cy']]), 'inner': inner, 'outer': outer,
            'A': A.astype(bool), 'B': B.astype(bool), 'crop': (x0,y0,x1,y1), 'has_ring': True,
            'ring_geometry_mode':'stored_inner_outer_ellipse'}

def external_contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1,2).astype(np.float32) for c in cs if len(c) >= 1]

def arc_span(points, center):
    if len(points) < 2: return 0.0
    angle = np.mod(np.arctan2(points[:,1]-center[1], points[:,0]-center[0]), 2*np.pi)
    bins = np.unique(np.floor(angle/(2*np.pi)*72).astype(int))
    if len(bins) <= 1: return 0.0
    bins = np.sort(bins); gaps = np.diff(np.r_[bins, bins[0]+72])
    return float(360*(1-gaps.max()/72))

def high_arcs(contours, states, near, center):
    arcs = []; cursor = 0; aid = 0; total_high = int((states == 'high_confidence').sum())
    for contour_id, pts in enumerate(contours):
        n = len(pts); state = states[cursor:cursor+n]; nnear = near[cursor:cursor+n]
        ishigh = state == 'high_confidence'
        if ishigh.all() and n:
            segments = [np.arange(n)]
        elif not ishigh.any():
            segments = []
        else:
            start = np.where(ishigh & ~np.roll(ishigh,1))[0]
            segments=[]
            for st in start:
                ids=[]; j=int(st)
                while ishigh[j]:
                    ids.append(j); j=(j+1)%n
                    if j==st: break
                segments.append(np.array(ids,dtype=int))
        for ids in segments:
            aid += 1; sub=pts[ids]
            if len(sub) > 1:
                length=float(np.linalg.norm(np.diff(sub,axis=0),axis=1).sum())
                if len(ids)==n: length += float(np.linalg.norm(sub[0]-sub[-1]))
            else: length=0.
            arcs.append({'arc_id': f'A{aid}', 'contour_component_id': contour_id, 'point_count':len(ids),
                         'arc_length_px':length, 'centroid_x':float(sub[:,0].mean()), 'centroid_y':float(sub[:,1].mean()),
                         'touches_ring_neighborhood':bool(nnear[ids].any()), 'angular_span_deg':arc_span(sub,center),
                         'high_confidence_contour_fraction':len(ids)/max(total_high,1),
                         '_global_ids': (cursor+ids).tolist()})
        cursor += n
    return arcs

def classify(r):
    d = prepare_original(r); contours = external_contours(d['polyp']); pts = np.concatenate(contours) if contours else np.empty((0,2),np.float32)
    n=len(pts)
    states=np.full(n, 'uncertain', object); cut_votes=np.zeros(n,int); outer_votes=np.zeros(n,int)
    if d.get('has_ring',False) and n:
        # Existing ring-constrained completion, reused unchanged and solely as diagnosis.
        _, rec = load_tools(); runs=rec.completion(d['polyp'].astype(bool), d['band'].astype(bool), float(d['rr']))
        near=point_values(d['guard'],pts).astype(bool)
        for run in runs:
            dt=cv2.distanceTransform(run['completed'].astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            dist=point_values(dt,pts); cut_votes += dist > INTERIOR_TOL; outer_votes += dist <= BOUNDARY_TOL
        # Away from ring, original outer contour is trusted except tiny isolated contour artefacts.
        short=np.zeros(n,bool); k=0
        for c in contours:
            if len(c)<SHORT_COMPONENT_POINTS: short[k:k+len(c)]=True
            k+=len(c)
        states[~near & ~short]='high_confidence'; states[~near & short]='uncertain'
        states[near & (cut_votes>=MAJORITY)]='ring_cut'
        states[near & (outer_votes>=MAJORITY) & ~(cut_votes>=MAJORITY)]='high_confidence'
        # all remaining ring-neighbourhood points stay uncertain.
        old_candidate=str(r.chosen_model_candidate) if pd.notna(r.chosen_model_candidate) else 'none'
        old_mask=d.get(old_candidate, np.zeros_like(d['polyp'],bool)) if old_candidate in ('A','B') else np.zeros_like(d['polyp'],bool)
        old_retained=(point_values(old_mask,pts)>0) & ~near
        stability=np.maximum(cut_votes,outer_votes)/len(runs)
        runs_bridge=sum(x['bridge_px']>0 for x in runs)
    else:
        near=np.zeros(n,bool); old_retained=np.zeros(n,bool); stability=np.zeros(n,float); runs=[];runs_bridge=0
    arcs=high_arcs(contours,states,near,d.get('center',np.array([0.,0.])))
    high=states=='high_confidence';unc=states=='uncertain';cut=states=='ring_cut'; old_deleted=~old_retained
    row={'stem':r.stem,'group':r.group,'capture':r.capture,'bin':int(r.segment_id),'frame_index':r.frame_index,
         'image_path':r.image_path,'mask_path':r.mask_path,'ring_json_path':r.ring_json_path,'ring_px':r.ring_px,
         'status':'ok' if d.get('has_ring',False) else 'missing_ring_geometry',
         'ring_geometry_mode':d.get('ring_geometry_mode','missing'),
         'original_contour_points':n,
         'original_contour_length_px':float(sum(np.linalg.norm(np.diff(c,axis=0),axis=1).sum()+ (np.linalg.norm(c[0]-c[-1]) if len(c)>1 else 0) for c in contours)),
         'old_v2_retained_fraction':float(old_retained.mean()) if n else math.nan,
         'old_v2_deleted_fraction':float(old_deleted.mean()) if n else math.nan,
         'new_high_confidence_fraction':float(high.mean()) if n else math.nan,
         'uncertain_fraction':float(unc.mean()) if n else math.nan,
         'ring_cut_fraction':float(cut.mean()) if n else math.nan,
         'ring_neighborhood_fraction':float(near.mean()) if n else math.nan,
         'high_confidence_arc_count':len(arcs),
         'max_high_confidence_arc_fraction':max([a['high_confidence_contour_fraction'] for a in arcs],default=0.),
         'recovered_from_old_v2_fraction':float((high&old_deleted).mean()) if n else math.nan,
         'new_high_near_ring_fraction':float((high&near).mean()) if n else math.nan,
         'multiscale_unstable_fraction':float((near&(stability<MAJORITY/len(SCALES))).mean()) if n else math.nan,
         'completion_scales_with_bridge':runs_bridge,
         'completion_mean_bridge_px':float(np.mean([x['bridge_px'] for x in runs])) if runs else math.nan}
    point_df=pd.DataFrame({'point_index':np.arange(n),'x':pts[:,0] if n else [],'y':pts[:,1] if n else [],'edge_class':states,
                           'ring_neighborhood':near,'completion_cut_votes':cut_votes,'completion_outer_votes':outer_votes,
                           'old_v2_retained':old_retained})
    for k,v in row.items(): point_df[k]=v
    for a in arcs:
        a.update({'stem':r.stem,'group':r.group,'capture':r.capture,'bin':int(r.segment_id)})
    return row, point_df, arcs, {'d':d,'pts':pts,'states':states,'near':near,'old_retained':old_retained,'arcs':arcs}

def blend(img, mask, color, alpha=.35):
    out=img.copy(); loc=mask.astype(bool); out[loc]=(out[loc]*(1-alpha)+np.array(color)*alpha).astype(np.uint8); return out
def draw_points(img, pts, sel, color, radius=1):
    out=img.copy()
    for x,y in pts[sel]: cv2.circle(out,(int(round(x)),int(round(y))),radius,color,-1,lineType=cv2.LINE_AA)
    return out
def image_card(row, aux):
    d,pts,states,near,old,arcs=aux['d'],aux['pts'],aux['states'],aux['near'],aux['old_retained'],aux['arcs']
    x0,y0,x1,y1=d['crop']; im=cv2.imread(str(row.image_path)); im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)[y0:y1,x0:x1]
    left=blend(im,d['polyp'],(20,210,60),.30); left=blend(left,d['ring'],(255,210,0),.55)
    if d.get('has_ring'): left=blend(left,d['band'],(190,40,220),.20)
    mid=im.copy(); mid=draw_points(mid,pts,np.ones(len(pts),bool),(0,230,255),1)
    right=blend(im,d['polyp'],(20,210,60),.18); right=blend(right,d['ring'],(255,210,0),.48)
    if d.get('has_ring'): right=blend(right,d['band'],(190,40,220),.18)
    # Old V2 is deliberately shown beneath new colours: blue retained, grey deleted.
    right=draw_points(right,pts,~old,(130,130,130),1); right=draw_points(right,pts,old,(30,150,255),1)
    right=draw_points(right,pts,states=='high_confidence',(30,230,60),2)
    right=draw_points(right,pts,states=='uncertain',(255,170,0),2)
    right=draw_points(right,pts,states=='ring_cut',(235,45,45),2)
    for a in arcs:
        cv2.putText(right,a['arc_id'],(int(a['centroid_x']),int(a['centroid_y'])),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(right,a['arc_id'],(int(a['centroid_x']),int(a['centroid_y'])),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA)
    h=max(left.shape[0],mid.shape[0],right.shape[0]); panels=[]
    # cv2's built-in font is ASCII-only; the Chinese legend lives in the HTML.
    for title,p in [('Raw: green=polyp, gold=ring, purple=unknown',left),('Original complete contour: cyan',mid),('Class: green=high / orange=uncertain / red=cut',right)]:
        if p.shape[0] != h: p=cv2.copyMakeBorder(p,0,h-p.shape[0],0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
        cv2.putText(p,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),2,cv2.LINE_AA);cv2.putText(p,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,0),1,cv2.LINE_AA);panels.append(p)
    card=np.hstack(panels); maxw=1500
    if card.shape[1]>maxw:
        card=cv2.resize(card,(maxw,round(card.shape[0]*maxw/card.shape[1])),interpolation=cv2.INTER_AREA)
    path=OUT/'gallery'/'assets'/f'{row.stem}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(path),cv2.cvtColor(card,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,88])

def pick_priority(df):
    specs=[
        ('ring 明显横穿 / 邻域占比高','ring_neighborhood_fraction',False),('ring 很大','ring_px',False),
        ('旧 V2 删除大量边缘','old_v2_deleted_fraction',False),('新方法恢复大量外缘','recovered_from_old_v2_fraction',False),
        ('completion 判断高度一致','multiscale_unstable_fraction',True),('multi-scale 判断不一致','multiscale_unstable_fraction',False),
        ('仅一小段 high-confidence arc','max_high_confidence_arc_fraction',True),('两段或更多空间分离 high-confidence arc','high_confidence_arc_count',False),
        ('待核查：可能误删真实外缘','ring_cut_fraction',False),('待核查：ring 邻域仍判真实外缘','new_high_near_ring_fraction',False),
    ]
    out=[]; used=set()
    for label,col,asc in specs:
        q=df.dropna(subset=[col]).sort_values(col,ascending=asc)
        for stem in q.stem:
            if stem not in used:
                out.append((label,stem));used.add(stem);break
    return out

def gallery(frames, priority):
    data=frames[['stem','group','capture','bin','new_high_confidence_fraction','uncertain_fraction','ring_cut_fraction','high_confidence_arc_count','old_v2_deleted_fraction','recovered_from_old_v2_fraction']].to_dict('records')
    priority_html=''.join(f'<li><a href="#{html.escape(stem)}">{html.escape(label)}：{html.escape(stem)}</a></li>' for label,stem in priority)
    cards=[]
    for r in frames.itertuples(index=False):
        cards.append(f'''<details id="{html.escape(r.stem)}"><summary><b>{html.escape(r.group)}</b> · {html.escape(r.capture)} · bin {r.bin} · {html.escape(r.stem)} | high {r.new_high_confidence_fraction:.1%}, uncertain {r.uncertain_fraction:.1%}, ring-cut {r.ring_cut_fraction:.1%}, arcs {r.high_confidence_arc_count}</summary>
<img loading="lazy" src="assets/{html.escape(r.stem)}.jpg"><div class="review" data-stem="{html.escape(r.stem)}"><button data-label="分类正确">分类正确</button><button data-label="恢复过多">恢复过多</button><button data-label="删除过多">删除过多</button><button data-label="ring-cut识别错误">ring-cut识别错误</button><button data-label="不确定">不确定</button><input placeholder="一句备注（浏览器内临时保存）"><span>未标注</span></div></details>''')
    page=f'''<!doctype html><meta charset="utf-8"><title>Original contour edge review</title>
<style>body{{font:14px system-ui;margin:22px;max-width:1550px;color:#222}}img{{width:100%;background:#111;margin:8px 0}}details{{border:1px solid #ccd;margin:8px 0;padding:8px}}summary{{cursor:pointer;line-height:1.6}}button{{margin:3px;padding:5px 8px}}input{{width:280px;padding:5px}}.chosen{{background:#d9f4d8}}.review span{{margin-left:8px;font-weight:600}}.legend{{padding:10px;background:#f4f4f4}}</style>
<h1>原始息肉 contour：ring 截断边三分类（仅人工审查）</h1>
<p>所有点均来自原始 INIT polyp mask 的外 contour。completion 只判断身份，绝不产生绘制/测量点；未运行 circle/ellipse fitting，也未读取 GT。</p>
<div class="legend">右图颜色：<b style="color:#0a9d30">绿色=高可信真实外缘</b>；<b style="color:#d88300">橙色=不确定</b>；<b style="color:#d00000">红色=ring 截断人工边</b>。金=ring，紫=unknown；蓝/灰分别为旧 V2 原保留/删除的原 contour。A1... 是连续 high-confidence arc。</div>
<p><a href="../tables/manual_edge_review.csv" download>下载可编辑人工审查模板 CSV</a> · 在本页面点击按钮/填写备注后，点击下方按钮下载带标签版本（静态页面无法直接写回磁盘）。</p>
<button id="download">下载当前人工标签 CSV</button><h2>优先审查（GT-free 自动挑选）</h2><ol>{priority_html}</ol><h2>全部 {len(frames)} 个 selected frames</h2>{''.join(cards)}
<script>const rows={json.dumps(data,ensure_ascii=False)};const labels={{}};document.querySelectorAll('.review').forEach(box=>{{const stem=box.dataset.stem,span=box.querySelector('span'),inp=box.querySelector('input');box.querySelectorAll('button').forEach(b=>b.onclick=()=>{{labels[stem]={{label:b.dataset.label,comment:inp.value||''}};box.querySelectorAll('button').forEach(x=>x.classList.remove('chosen'));b.classList.add('chosen');span.textContent=b.dataset.label;}});inp.oninput=()=>{{if(labels[stem])labels[stem].comment=inp.value;}};}});document.querySelector('#download').onclick=()=>{{const head=['group','capture','bin','stem','auto_high_confidence_fraction','auto_uncertain_fraction','auto_ring_cut_fraction','arc_count','manual_label','comment'];const esc=x=>'"'+String(x??'').replaceAll('"','""')+'"';const lines=[head.join(',')];rows.forEach(r=>{{const z=labels[r.stem]||{{}};lines.push([r.group,r.capture,r.bin,r.stem,r.new_high_confidence_fraction,r.uncertain_fraction,r.ring_cut_fraction,r.high_confidence_arc_count,z.label||'',z.comment||''].map(esc).join(','));}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{{type:'text/csv;charset=utf-8'}}));a.download='manual_edge_review.csv';a.click();}});</script>'''
    (OUT/'gallery/edge_review.html').write_text(page,encoding='utf-8')

def main():
    for d in ('tables','gallery','gallery/assets','inputs','scripts'): (OUT/d).mkdir(parents=True,exist_ok=True)
    protocol={'purpose':'GT-free original segmentation contour edge identity classification only','baseline':'S7-FMD39 selected stems; geometry/fusion untouched','forbidden':['GT/reference diameter','circle fitting','ellipse fitting','new contour generation for measurement'],'original_contour_only':True,'diagnostic_completion_source':'ring_occlusion_boundary_recovery_v1 completion()','scales_over_ring_radius':list(SCALES),'boundary_tol_px':BOUNDARY_TOL,'interior_tol_px':INTERIOR_TOL,'majority_required':MAJORITY,'classes':{'high_confidence':'away from ring, or majority completion runs retain original point near completed outer boundary','uncertain':'inconsistent/topologically insufficient completion evidence, or tiny remote contour artefact','ring_cut':'ring-neighborhood original point is interior to completed mask in >=3/5 diagnostic scales'}}
    (OUT/'inputs/frozen_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    sel=selected(); save_csv(sel,'tables/frozen_s7_selected_stems_blind.csv')
    rows=[]; points=[]; arcs=[]
    for i,r in enumerate(sel.itertuples(index=False),1):
        try:
            row, p, a, aux = classify(r); rows.append(row); points.append(p); arcs.extend(a); image_card(r,aux)
        except Exception as e:
            rows.append({'stem':r.stem,'group':r.group,'capture':r.capture,'bin':int(r.segment_id),'frame_index':r.frame_index,'status':f'error:{type(e).__name__}:{e}', 'original_contour_points':math.nan})
        if i%25==0: print(f'{i}/395',flush=True)
    frame=pd.DataFrame(rows).sort_values(['group','capture','bin']); ptab=pd.concat(points,ignore_index=True) if points else pd.DataFrame(); atab=pd.DataFrame(arcs)
    save_csv(frame,'tables/edge_classification.csv');save_csv(ptab,'tables/original_contour_point_classification.csv');save_csv(atab,'tables/high_confidence_arcs.csv')
    template=frame[['group','capture','bin','stem','new_high_confidence_fraction','uncertain_fraction','ring_cut_fraction','high_confidence_arc_count']].copy()
    template.columns=['group','capture','bin','stem','auto_high_confidence_fraction','auto_uncertain_fraction','auto_ring_cut_fraction','arc_count']
    template['manual_label']='';template['comment']='';save_csv(template,'tables/manual_edge_review.csv')
    priority=pick_priority(frame[frame.status.eq('ok')]); gallery(frame,priority)
    good=frame[frame.status.eq('ok')]
    report=f'''# 原始 contour 边缘三分类（待人工审查）\n\n自动分类 **{len(frame)}** 个 S7 fixed selected frames（成功 `{len(good)}`）。按原始 contour 点汇总：high-confidence **{good.new_high_confidence_fraction.mean():.1%}**，uncertain **{good.uncertain_fraction.mean():.1%}**，ring-cut **{good.ring_cut_fraction.mean():.1%}**。旧 V2 平均删除 **{good.old_v2_deleted_fraction.mean():.1%}**；新 high-confidence 中、原先被旧 V2 删除的平均比例为 **{good.recovered_from_old_v2_fraction.mean():.1%}**。\n\n优先审查：\n'''+''.join(f'- {label}: `{stem}`\n' for label,stem in priority)+'''\n\n只做原始边缘身份标记；没有 ellipse/circle fitting、Dxy 比较或 GT。详见 `gallery/edge_review.html`；人工标签请编辑 `tables/manual_edge_review.csv` 或在网页中下载已标注版本。\n'''
    (OUT/'HUMAN_REPORT.md').write_text(report,encoding='utf-8')
    files=['inputs/frozen_protocol.json','tables/frozen_s7_selected_stems_blind.csv','tables/edge_classification.csv','tables/original_contour_point_classification.csv','tables/high_confidence_arcs.csv','tables/manual_edge_review.csv']
    (OUT/'inputs/blind_sha256.txt').write_text(''.join(f'{sha(OUT/f)}  {f}\n' for f in files),encoding='utf-8')
    print('complete',len(frame),'frames',len(ptab),'points',len(atab),'arcs')

if __name__=='__main__': main()
