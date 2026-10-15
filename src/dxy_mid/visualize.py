"""One PNG per group: actual mm distribution and median representative frames."""
from pathlib import Path
import pickle
import cv2
import numpy as np

def render_groups(frames,groups,videos,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Noto Sans CJK JP','Noto Sans CJK SC','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    output=Path(output);dest=output/'png';dest.mkdir(exist_ok=True)
    for group,part in frames.groupby('group',sort=True):
        good=part[part.measurement_eligible.astype(bool)&np.isfinite(part.D_mm)]
        vv=videos[videos.group==group];valid_v=vv[np.isfinite(vv.median_mm)]
        n=max(1,len(valid_v));fig=plt.figure(figsize=(7+3.5*n,5),layout='constrained')
        grid=fig.add_gridspec(1,n+1,width_ratios=[2]+[1]*n);ax=fig.add_subplot(grid[0,0])
        for i,(video,z) in enumerate(good.groupby('video',sort=True)):
            stable=z.stability_review_pass.astype(bool).to_numpy();vals=z.D_mm.to_numpy(float)
            jitter=np.linspace(-.12,.12,len(z))
            ax.scatter(i+jitter[stable],vals[stable],s=18,c='#277da8')
            ax.scatter(i+jitter[~stable],vals[~stable],s=22,c='#e59031',marker='x')
            m=float(np.median(vals));ax.plot([i-.3,i+.3],[m,m],c='black',lw=2)
        pred=groups.loc[groups.group==group,'prediction_mm'].iloc[0]
        if np.isfinite(pred):ax.axhline(pred,c='#cc3054',lw=2,label=f'组级中位数 {pred:.3f} mm');ax.legend()
        ax.set_xticks(range(len(valid_v)),[Path(str(x)).stem.rsplit('_',1)[-1] for x in valid_v.video],rotation=25)
        ax.set(xlabel='视频（每段等权）',ylabel='直径（mm）',title='逐帧毫米测量（×：未通过严格稳定性）');ax.grid(alpha=.2)
        for col,v in enumerate(valid_v.itertuples(),1):
            z=good[good.video==v.video].copy();z['distance']=(z.D_mm-v.median_mm).abs()
            order=['distance']+(['frame_index'] if 'frame_index' in z else [])+['source_id'];row=z.sort_values(order).iloc[0]
            with (output/'cache'/(row.cache_key+'.pkl')).open('rb') as f:measurement=pickle.load(f)['measurement']
            im=cv2.cvtColor(cv2.imread(row.image_path),cv2.COLOR_BGR2RGB);ay=fig.add_subplot(grid[0,col]);ay.imshow(im)
            ring=measurement.ring.geometry['centerline'];t=np.linspace(0,2*np.pi,361);angle=np.deg2rad(ring['angle_deg']);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            pts=np.array([ring['cx'],ring['cy']])+np.c_[ring['major_radius']*np.cos(t),ring['minor_radius']*np.sin(t)]@R.T;ay.plot(pts[:,0],pts[:,1],c='yellow',lw=1)
            candidate=measurement.candidate
            if candidate is not None:
                pts=candidate.center_native+np.c_[np.cos(t),np.sin(t)]@candidate.basis_native.T;ay.plot(pts[:,0],pts[:,1],c='#47e06f',lw=1.5)
                ay.plot([row.axis_x1,row.axis_x2],[row.axis_y1,row.axis_y2],'o-',c='#ff4295',lw=2,ms=4)
            ay.set_title(f'{Path(str(v.video)).stem.rsplit("_",1)[-1]}：{row.D_mm:.3f} mm\n{measurement.audit["quality_class"]}',fontsize=9);ay.axis('off')
        if len(valid_v)==0:fig.add_subplot(grid[0,1]).axis('off')
        fig.suptitle(f'{group}\nDXY-MID / {pred:.3f} mm / 有效帧 {len(good)}/{len(part)} / 有效视频 {len(valid_v)}/{len(vv)}',fontsize=11)
        fig.savefig(dest/(group+'.png'),dpi=140);plt.close(fig)
