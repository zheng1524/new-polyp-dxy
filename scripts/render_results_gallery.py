"""Collaborator result gallery from actual frozen Measurement objects; no fits."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dxy_mid import _engine
from dxy_mid.diagnostics import ellipse_points, candidate_points

ROOT=Path(__file__).resolve().parents[1]
plt.rcParams['font.sans-serif']=['Noto Sans CJK JP','Noto Sans CJK SC','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def band(error):
    return '0–2%' if error<=2 else '2–5%' if error<=5 else '5–10%' if error<=10 else '10–15%' if error<=15 else '>15%'


def background(ax,image,labels,zoom=True):
    ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));ax.axis('off')
    if zoom:
        y,x=np.nonzero(labels>0)
        if len(x):
            ax.set_xlim(max(0,x.min()-25),min(labels.shape[1],x.max()+25))
            ax.set_ylim(min(labels.shape[0],y.max()+25),max(0,y.min()-25))


def overlay_mask(ax,labels):
    rgba=np.zeros((*labels.shape,4));rgba[labels==1]=[.2,.9,.4,.4];rgba[labels==2]=[1.,.65,.1,.5];ax.imshow(rgba)


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=lambda z:z.tolist() if isinstance(z,np.ndarray) else z.item()))


def render_ranking(groups,selection,dest):
    d=groups.reset_index();fig,ax=plt.subplots(figsize=(11,13));y=np.arange(len(d))
    ax.barh(y,d.absolute_error_pct,color=d.ring.map({'R5':'#287da8','R10':'#ec9653'}));ax.invert_yaxis();labels=[]
    for i,r in enumerate(d.itertuples()):
        name=r.display_group.split('_closed')[0].replace('GP_','').replace('_',' ');labels.append(f'{name} {r.ring}')
        ax.text(r.absolute_error_pct+.1,i,f'{r.absolute_error_pct:.2f}%',va='center',fontsize=8)
        if r.group_id in selection:ax.scatter(r.absolute_error_pct+.9,i,marker='*',c='#b22664',s=55)
    ax.set_yticks(y,labels,fontsize=8);ax.set_xlim(0,max(d.absolute_error_pct)+2.5)
    ax.set_xlabel(f'绝对相对误差（%）；星号标出{len(selection)}组结果图集案例')
    ax.set_title(f'DXY-MID：39/39组，757/790帧\n中位绝对相对误差{d.absolute_error_pct.median():.6f}%；MAE {d.absolute_error_mm.mean():.6f}mm\nR5蓝／R10橙；原图Mid尺度，GT为人工确认基准')
    ax.axvline(5,c='gray',ls=':',lw=1);ax.axvline(10,c='gray',ls=':',lw=1)
    fig.tight_layout();dest.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dest,dpi=170);plt.close(fig)


def load_actual(row,inputs,args):
    source=inputs.loc[row.source_id]
    image_path=args.input_root/source.image_path;mask_path=args.input_root/source.mask_path
    assert sha(image_path)==source.image_sha256 and sha(mask_path)==source.mask_sha256
    with (args.cache_run/'cache'/(row.cache_key+'.pkl')).open('rb') as f:cache=pickle.load(f)
    assert cache['key']==row.cache_key
    m=cache['measurement'];a=m.audit
    assert a['source_id']==row.source_id and a['image_sha256']==source.image_sha256 and a['mask_sha256']==source.mask_sha256
    image=cv2.imread(str(image_path));labels=cv2.imread(str(mask_path),cv2.IMREAD_UNCHANGED)
    assert labels.dtype==np.uint8 and set(np.unique(labels))<={0,1,2} and image.shape[:2]==labels.shape
    assert np.isclose(a['D_frame_mm'],row.D_mm,atol=1e-8,equal_nan=True)
    for key in ['measurement_eligible','normal_Gate_pass','Q_conflict_recovery','strict_stable','ring_valid']:
        assert a[key]==getattr(row,key)
    return m,image,labels,source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-run',type=Path)
    p.add_argument('--input-root',type=Path)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,default=ROOT/'validation/frozen39/inputs.csv')
    p.add_argument('--output',type=Path,default=ROOT/'docs/results_gallery')
    p.add_argument('--ranking-only',action='store_true')
    p.add_argument('--ranking-output',type=Path,default=ROOT/'mainline/figures/group_abs_error_descending.png')
    args=p.parse_args();out=args.output
    groups=pd.read_csv(ROOT/'mainline/tables/group_results.csv').set_index('group_id')
    selection=json.loads(args.selection.read_text())
    if args.ranking_only:
        render_ranking(groups,selection,args.ranking_output);return
    if args.cache_run is None or args.input_root is None:p.error('Gallery needs --cache-run and --input-root; --ranking-only needs neither')
    for d in ['figures','evidence']:(out/d).mkdir(parents=True,exist_ok=True)
    inputs=pd.read_csv(args.input_manifest).set_index('source_id')
    frames=pd.read_csv(args.cache_run/'frames.csv');videos=pd.read_csv(args.cache_run/'videos.csv')
    index=[];failure_rows=[]
    for number,group in enumerate(selection,1):
        g=groups.loc[group];part=frames[frames.group.eq(group)];good=part[part.measurement_eligible]
        representatives=[]
        for video,z in good.groupby('video'):
            median=videos.loc[videos.group.eq(group)&videos.video.eq(video),'median_mm'].item()
            rep=z.assign(distance=(z.D_mm-median).abs()).sort_values(['distance','frame_index','source_id']).iloc[0].copy()
            rep['video_median_mm']=median;rep['video_deviation']=abs(median-g.DXY_MID_mm);representatives.append(rep)
        row=pd.DataFrame(representatives).sort_values(['video_deviation','video','frame_index']).iloc[0]
        m,image,labels,source=load_actual(row,inputs,args);a=m.audit;c=m.candidate
        assert c is not None and m.ring.valid
        _,_,axis,ends=_engine.axes(c)
        np.testing.assert_allclose(ends,[[row.axis_x1,row.axis_y1],[row.axis_x2,row.axis_y2]],atol=1e-7)
        assert np.isclose(np.linalg.norm(ends[1]-ends[0]),a['L_px'],atol=1e-7)
        mid=m.ring.geometry['centerline'];assert np.isclose(2*mid['major_radius'],a['L_mid_px'])
        assert np.isclose(a['D_frame_mm'],a['L_px']*a['D_mid_mm']/a['L_mid_px'])
        contours,_=cv2.findContours((labels==1).astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        Q=m.evidence['Q'];P=m.evidence['P']
        for q in Q:
            assert np.array_equal(q['q'],contours[q['contour_id']].reshape(-1,2)[q['contour_index']])
            assert labels[int(q['q'][1]),int(q['q'][0])]==1
        assert len(P)==a['P_count'] and len(Q)==a['Q_count']
        fig,axs=plt.subplots(1,5,figsize=(19,5.2),layout='constrained')
        for i,ax in enumerate(axs):background(ax,image,labels,zoom=i>0)
        axs[0].set_title('① 原始内镜图像',fontsize=11)
        overlay_mask(axs[1],labels);axs[1].set_title('② 原始分割：息肉绿／Ring橙',fontsize=11)
        e=ellipse_points(mid);axs[2].plot(*e.T,c='#ffe45c',lw=2)
        theta=np.deg2rad(mid['angle_deg']);v=mid['major_radius']*np.array([np.cos(theta),np.sin(theta)]);center=np.array([mid['cx'],mid['cy']]);midends=np.array([center-v,center+v])
        axs[2].plot(*midends.T,'o-',c='#ffe45c',lw=2,ms=3)
        axs[2].set_title(f"③ 实际Mid尺度：{a['D_mid_mm']:.1f}mm/{a['L_mid_px']:.2f}px",fontsize=11)
        if len(P):axs[3].scatter(*P.T,s=4,c='#35dc70')
        if Q:axs[3].scatter(*np.array([q['q'] for q in Q]).T,s=4,c='#00d8ff')
        curve=candidate_points(c)
        axs[3].plot(*curve.T,c='#ff408e',lw=1.8);axs[3].set_title(f"④ P绿／Q青＋实际主体（{c.shape}）",fontsize=11)
        axs[4].plot(*curve.T,c='#ff408e',lw=1.5);axs[4].plot(*ends.T,'o-',c='#ff408e',lw=2,ms=4)
        axs[4].set_title(f"⑤ 展示帧Dxy={a['D_frame_mm']:.3f}mm",fontsize=11)
        display=g.display_group.split('_closed')[0].replace('GP_','').replace('_',' ')
        fig.suptitle(f"{display} / 实际Paris={g.paris} / {g.ring}｜组级GT={g.GT_mm:.3f}mm → DXY-MID={g.DXY_MID_mm:.3f}mm\n组级绝对误差={g.absolute_error_mm:.3f}mm；有符号相对误差={g.signed_error_pct:+.2f}%（负值低估，正值高估）",fontsize=13)
        fig.supxlabel(f"展示帧：Normal={a['normal_Gate_pass']} / Q恢复={a['Q_conflict_recovery']} / Strict={a['strict_stable']}；source_id={row.source_id}\n全组：有效{g.n_valid_frames}/{g.n_input_frames}帧，Ring失败{g.n_ring_failed_frames}帧，Q恢复{g.n_Q_recovery_frames}帧，Strict {g.n_strict_frames}帧；有效视频{g.n_valid_videos}/{g.n_input_videos}。单帧值不等于组级值。",fontsize=9)
        filename=f'{number:02d}_{group}.png';fig.savefig(out/'figures'/filename,dpi=145);plt.close(fig)
        evidence=dict(group_id=group,source_id=row.source_id,video=row.video,frame_index=int(row.frame_index),timestamp_sec=float(row.timestamp_sec),image_sha256=source.image_sha256,mask_sha256=source.mask_sha256,
            selected_from_actual_aggregation=True,representative_rule='Closest frame to each video median, then video median closest to group; chronological/video-ID tie-break; no GT',
            frame_Dxy_mm=a['D_frame_mm'],L_px=a['L_px'],L_mid_px=a['L_mid_px'],scale_mm_per_px=a['scale_mm_per_px'],
            endpoints=ends,chosen=dict(shape=c.shape,center=c.center_native,basis=c.basis_native),ring=m.ring.geometry,
            P_count=len(P),Q_count=len(Q),Q_original_contour_indices_verified=True,normal=a['normal_Gate_pass'],Q_recovery=a['Q_conflict_recovery'],strict=a['strict_stable'],
            group_prediction_mm=g.DXY_MID_mm,GT_mm=g.GT_mm,signed_error_pct=g.signed_error_pct,checks_passed=True,
            Q_indices=[dict(contour_id=q['contour_id'],contour_index=q['contour_index'],q=q['q']) for q in Q],figure_sha256=sha(out/'figures'/filename),geometry_refit_for_plot=False)
        dump(out/'evidence'/(group+'.json'),evidence)
        index.append(dict(group_id=group,display_group=g.display_group,paris=g.paris,ring=g.ring,GT_mm=g.GT_mm,DXY_MID_mm=g.DXY_MID_mm,
            signed_error_pct=g.signed_error_pct,absolute_error_pct=g.absolute_error_pct,error_band=band(g.absolute_error_pct),source_id=row.source_id,frame_Dxy_mm=a['D_frame_mm'],
            video=row.video,video_median_mm=row.video_median_mm,frame_index=row.frame_index,figure='figures/'+filename,evidence='evidence/'+group+'.json',checks_passed=True))
        # Extra failure contrast for the illustrated group with the most Ring failures.
        if g.n_ring_failed_frames and g.crosses_10mm:
            bad=part[~part.ring_valid].sort_values(['video','frame_index']).iloc[0]
            fm,im,lab,src=load_actual(bad,inputs,args);assert fm.candidate is None and np.isnan(fm.audit['D_frame_mm'])
            ff,aa=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
            for ax in aa:background(ax,im,lab)
            overlay_mask(aa[1],lab);aa[0].set_title('同组真实Ring失败帧');aa[1].set_title('Dxy=NaN；不参与聚合、不补值')
            ff.suptitle(f"{display} / {g.ring}：参考环恢复未通过\n{fm.ring.reason}\nsource_id={bad.source_id}",fontsize=10)
            fn=f'{number:02d}_{group}_Ring_failure.png';ff.savefig(out/'figures'/fn,dpi=145);plt.close(ff)
            failure_rows.append(dict(group_id=group,source_id=bad.source_id,figure='figures/'+fn,reason=fm.ring.reason,Dxy_mm=np.nan,checks_passed=True))
        print(number,group,'verified',flush=True)
    pd.DataFrame(index).to_csv(out/'index.csv',index=False)
    pd.DataFrame(failure_rows).to_csv(out/'failure_index.csv',index=False)
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    import shutil
    shutil.copyfile(ROOT/'mainline/figures/group_abs_error_descending.png',out/'figures/39_group_error_ranking.png')
    print('Gallery verified:',len(index),'groups;',len(failure_rows),'extra hard-failure frame')


if __name__=='__main__':main()
