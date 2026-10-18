"""PNG-only, read-only visualization of the public DXY-MID cases."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from dxy_mid import _engine as engine
from dxy_mid.batch import load_manifest
from dxy_mid.diagnostics import observe_files, export_observation, ellipse_points, candidate_points, candidate_records

COLORS = dict(P='#38df77', Q='#00d8ff', X='#ff5e54', chosen='#ff408e',
              inner='#54b8ff', outer='#ffae42', centerline='#ffe45c')
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def background(ax, observed, zoom=True):
    ax.imshow(cv2.cvtColor(observed.native_image, cv2.COLOR_BGR2RGB))
    if not zoom:
        h, w = observed.original_labels.shape
        ax.set_xlim(-.5, w-.5)
        ax.set_ylim(h-.5, -.5)
    if zoom:
        ys, xs = np.nonzero(observed.original_labels > 0)
        if len(xs):
            h, w = observed.original_labels.shape
            ax.set_xlim(max(0, xs.min()-30), min(w, xs.max()+30))
            ax.set_ylim(min(h, ys.max()+30), max(0, ys.min()-30))
    ax.set_aspect('equal')
    ax.axis('off')


def line(ax, pts, color, **kwargs):
    ax.plot(pts[:, 0], pts[:, 1], color=color, **kwargs)


def ring_overlay(ax, observed):
    ring = observed.measurement.ring
    for name in ['inner', 'outer', 'centerline']:
        if name in ring.geometry:
            line(ax, ellipse_points(ring.geometry[name]), COLORS[name], lw=1.7, label=name)
    if ring.valid:
        mid = ring.geometry['centerline']
        angle = np.deg2rad(mid['angle_deg'])
        offset = mid['major_radius']*np.array([np.cos(angle), np.sin(angle)])
        ends = np.array([np.array([mid['cx'], mid['cy']])-offset,
                         np.array([mid['cx'], mid['cy']])+offset])
        line(ax, ends, COLORS['centerline'], marker='o', ms=3, lw=2)


def final_overlay(ax, observed):
    c = observed.measurement.candidate
    if c is not None:
        line(ax, candidate_points(c), COLORS['chosen'], lw=2)
        _, _, _, ends = engine.axes(c)
        line(ax, ends, COLORS['chosen'], marker='o', ms=4, lw=2.2)


def plot_evidence(ax, observed, sampled=False):
    records = observed.boundary['original_boundary']['records']
    for label in ['X', 'P', 'Q']:
        p = np.array([z['q'] for z in records if z['class_'] == label]).reshape(-1, 2)
        if len(p):
            ax.scatter(*p.T, s=3, c=COLORS[label], label=label, zorder=4)
    data = observed.selected['data']
    qi = observed.selected['actual_sampled_Q_indices']
    qq = [data['Q'][i] for i in qi] if sampled else data['Q']
    # Drawing sparsely never changes allowed regions or the numerical inputs.
    for q in qq[::max(1, len(qq)//28)]:
        ends = np.array([q['q']+q['lo']*q['n'], q['q']+q['hi']*q['n']])
        line(ax, ends, COLORS['Q'], lw=1.3, alpha=.9)
    if sampled and qi:
        points = np.array([data['Q'][i]['q'] for i in qi])
        ax.scatter(*points.T, s=19, facecolors='none', edgecolors='white', linewidths=.6, zorder=5)


def stages(observed, dest, title):
    a = observed.measurement.audit
    fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for ax in axs.flat:
        background(ax, observed)
    background(axs[0, 0], observed, False)
    axs[0, 0].set_title('① 原始内镜JPEG（未画线原始数据另附）')
    palette = np.array([[35, 38, 43], [56, 223, 119], [255, 174, 66]], np.uint8)
    axs[0, 1].imshow(palette[observed.original_labels], alpha=.62)
    axs[0, 1].set_title('② 原始多类mask：背景灰／息肉绿／Ring橙')
    ring_overlay(axs[0, 2], observed)
    evidence = observed.measurement.ring.evidence
    for name, color in [('inner_points', COLORS['inner']), ('outer_points', COLORS['outer'])]:
        p = np.asarray(evidence.get(name, [])).reshape(-1, 2)
        if len(p):
            axs[0, 2].scatter(*p.T, s=2, c=color, alpha=.7)
    axs[0, 2].set_title(f"③ 真实环候选点＋既有RANSAC结果\nInner蓝／Outer橙／Mid黄；完整Mid轴={a['L_mid_px']:.3f}px")
    plot_evidence(axs[1, 0], observed, True)
    axs[1, 0].set_title(f"④ 原始轮廓P绿／Q青／X红；白圈=抽样Q\n求解P={a['P_count']} Q={a['Q_count']}；短Q弧剔除点={a['Q_removed_short_arc_count']}")
    pool = candidate_records(observed)
    eligible = sorted([z for z in pool if not z['diagnostic_profile']], key=lambda z: z['score'])
    displayed = eligible[:8]
    for rec in displayed:
        e = dict(cx=rec['cx'], cy=rec['cy'], major_radius=rec['radius_1_px'],
                 minor_radius=rec['radius_2_px'], angle_deg=np.rad2deg(rec['angle_rad']))
        line(axs[1, 1], ellipse_points(e), '#dadada', lw=.6, alpha=.65)
    final_overlay(axs[1, 1], observed)
    axs[1, 1].set_title(f"⑤ 实际A/B圆、椭圆候选：显示{len(displayed)}条／池{len(pool)}条\n粉色=真实chosen及测量轴；全池及原拒绝理由另附CSV")
    final_overlay(axs[1, 2], observed)
    ring_overlay(axs[1, 2], observed)
    axs[1, 2].set_title(f"⑥ Dxy={a['L_px']:.3f}px × {a['scale_mm_per_px']:.6f}mm/px\n={a['D_frame_mm']:.3f}mm；{a['quality_class']}；Strict={a['strict_stable']}")
    handles = [Line2D([0], [0], color=COLORS[k], lw=2, label=k) for k in ['P', 'Q', 'X', 'chosen', 'centerline']]
    fig.legend(handles=handles, loc='outside lower center', ncol=5)
    fig.suptitle(f'{title}｜DXY-MID：V9.1＋Mid，原图坐标\nsource_id={a["source_id"]}', fontsize=11)
    fig.savefig(dest, dpi=175)
    plt.close(fig)
    pd.DataFrame(displayed).to_csv(dest.with_suffix('.displayed_candidates.csv'), index=False)


def failure(observed, dest):
    a = observed.measurement.audit
    fig, axs = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for ax in axs:
        background(ax, observed)
    axs[0].set_title('原始图：Ring遮挡及高光')
    overlay = np.zeros((*observed.original_labels.shape, 4))
    overlay[observed.original_labels == 1] = [0.2, .9, .4, .45]
    overlay[observed.original_labels == 2] = [1., .6, .1, .55]
    axs[1].imshow(overlay)
    axs[1].set_title(f'原始mask；小孔修复{a["ring_filled_area_px"]}px仍未通过')
    ring = observed.measurement.ring
    for key, color in [('inner_points', COLORS['inner']), ('outer_points', COLORS['outer']), ('rejected_points', COLORS['X'])]:
        p = np.asarray(ring.evidence.get(key, [])).reshape(-1, 2)
        if len(p):
            axs[2].scatter(*p.T, s=3, c=color)
    details = ring.geometry.get('diagnostics', {})
    reason = details.get('reason', ring.reason)
    axs[2].set_title(f'候选点不等于可靠几何；Dxy=NaN\n{ring.reason}', fontsize=10)
    fig.suptitle(f'Ring硬失败：不进入主体拟合、不补数值\nsource_id={a["source_id"]}\n诊断：{reason}', fontsize=10)
    fig.savefig(dest, dpi=175)
    plt.close(fig)


def holes(observed, dest):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    for ax in axs:
        background(ax, observed)
    for ax, labels in zip(axs, [observed.original_labels, observed.effective_labels]):
        overlay = np.zeros((*labels.shape, 4));overlay[labels == 2] = [1., .65, .1, .5]
        ax.imshow(overlay)
    y, x = np.nonzero(observed.filled)
    axs[1].scatter(x, y, c='#00d8ff', s=3)
    axs[0].set_title('原始Ring mask（文件不变）')
    axs[1].set_title(f'内存小孔填充：{observed.filled.sum()}px（青色）')
    fig.suptitle(f'冻结小孔修复；不连接断环、不修改息肉mask\nsource_id={observed.measurement.audit["source_id"]}', fontsize=10)
    fig.savefig(dest, dpi=175);plt.close(fig)


def aggregation(run, dest, cases):
    frames = pd.read_csv(run/'frames.csv');scopes = pd.read_csv(cases/'scope.csv')
    group = scopes.loc[scopes.complete_frozen_group, 'group'].item()
    z = frames[frames.group == group].copy()
    groups = pd.read_csv(run/'groups.csv');videos = pd.read_csv(run/'videos.csv')
    vv = videos[videos.group == group]
    pred = groups.loc[groups.group == group, 'prediction_mm'].item()
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
    point_rows = []
    for j, (video, part) in enumerate(z.groupby('video', sort=True)):
        plot_x = j+np.linspace(-.12, .12, len(part))
        axs[0].scatter(plot_x, part.D_mm, c='#277da8', s=28)
        for x, row in zip(plot_x, part.itertuples()):
            point_rows.append(dict(source_id=row.source_id, group=row.group, video=row.video,
                                   frame_index=row.frame_index, timestamp_sec=row.timestamp_sec,
                                   D_mm=row.D_mm, plot_x=x, strict_stable=row.strict_stable))
        median = vv.loc[vv.video == video, 'median_mm'].item()
        axs[0].hlines(median, j-.28, j+.28, color='black', lw=2)
        axs[0].text(j+.15, median, f'{median:.4f}mm', fontsize=9)
        axs[1].plot(part.sort_values('timestamp_sec').timestamp_sec, part.sort_values('timestamp_sec').D_mm, 'o-', label=f'视频{j+1}（{len(part)}帧）', ms=4)
    axs[0].axhline(pred, color=COLORS['chosen'], lw=2, label=f'视频等权中位数 {pred:.6f}mm')
    axs[0].axhline(10, color='gray', ls=':', label='10mm阈值（非GT）')
    axs[0].set(xticks=range(len(vv)), xticklabels=['视频1', '视频2'], ylabel='Dxy（mm）', title='全部20帧→两段视频中位→组中位')
    axs[1].set(xlabel='原视频时间戳（s；各视频独立时钟）', ylabel='Dxy（mm）', title='冻结选帧的时间分布；没有重新选帧')
    for ax in axs:
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle(f'{group}\n完整冻结组20/20帧、2/2视频；DXY-MID V9.1＋Mid\n20个source_id及逐点数值：multi_video_points.csv（同目录）', fontsize=10)
    fig.savefig(dest, dpi=175);plt.close(fig)
    pd.DataFrame(point_rows).to_csv(dest.with_name('multi_video_points.csv'), index=False)
    return dict(name='multi_video_aggregation', source_id=';'.join(z.source_id), png=dest.name,
                evidence='multi_video_points.csv', quality_class='complete_group_20_frames',
                D_mm=pred, checks_passed=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases', type=Path, default=Path('examples/reproduction_cases'))
    ap.add_argument('--run', type=Path, required=True, help='Fresh public-case measurement run')
    ap.add_argument('--output', type=Path, default=Path('outputs/public-process'))
    ap.add_argument('--only', nargs='+', help='Redraw only these named diagnostic figures')
    args = ap.parse_args();args.output.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.cases/'mechanisms.csv')
    ids = json.loads((args.cases/'visualization_selection.json').read_text())
    rowmap = {r['source_id']: r for r in rows};index=[]
    for name, specification in ids.items():
        if args.only and name not in args.only:
            continue
        row = rowmap[specification['source_id']]
        observed = observe_files(row['image_path'], row['mask_path'], inner_diameter_mm=row['ring_mm'], source_id=row['source_id'], stem=row['stem'])
        evidence = args.output/'evidence'/name
        export_observation(observed, evidence)
        actual = pd.read_csv(args.run/'frames.csv').set_index('source_id').loc[row['source_id']]
        for key in ['D_frame_mm', 'L_px', 'L_mid_px', 'scale_mm_per_px']:
            if not np.isclose(observed.measurement.audit[key], actual[key], atol=1e-8, rtol=1e-10, equal_nan=True):
                raise AssertionError((name, key, observed.measurement.audit[key], actual[key]))
        target = args.output/(name+'.png')
        kind = specification['kind']
        if kind == 'failure': failure(observed, target)
        elif kind == 'holes': holes(observed, target)
        else: stages(observed, target, specification['title'])
        index.append(dict(name=name, source_id=row['source_id'], png=target.name, evidence=f'evidence/{name}',
                          quality_class=observed.measurement.audit['quality_class'],
                          D_mm=observed.measurement.audit['D_frame_mm'], checks_passed=all(observed.checks.values())))
        print(name, 'verified', len(observed.checks), flush=True)
    index.append(aggregation(args.run, args.output/'multi_video_aggregation.png', args.cases))
    if args.only and (args.output/'index.csv').exists():
        prior = pd.read_csv(args.output/'index.csv')
        prior = prior[~prior.name.isin([z['name'] for z in index])]
        pd.concat([prior, pd.DataFrame(index)], ignore_index=True).to_csv(args.output/'index.csv', index=False)
    else:
        pd.DataFrame(index).to_csv(args.output/'index.csv', index=False)


if __name__ == '__main__':
    main()
