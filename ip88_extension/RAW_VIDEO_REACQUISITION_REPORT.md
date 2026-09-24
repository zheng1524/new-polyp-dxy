# Ip8.8 R10 原始视频重采集与面积支持适配报告

## 结论

这次没有读取、复用或接入旧 V2 R3 的 4.067 mm 结果。直接从 `_004.mp4`、`_005.mp4` 各重采 250 帧，以冻结 INIT 分割后重新运行当前环拟合和未修改 V2。原有失败不是环不可见，而是环带因息肉紧贴内缘没有形成闭合孔，使当前 fitter 缺少 `inner_ellipse`。

对这种情况，新适配只使用当前 fitter 已产出的 `outer_ellipse` 与 `mid_ellipse`：`inner = 2*mid - outer`（中心和两个半轴各自计算）。它是环带中心线对称的几何补全，不是 GT 标定、R3 迁移或新 V2 测量算法。只有正且位于外缘内的反推内环被接受。

## 盲流程冻结与面积 guard

- 原始候选：500；标准 native inner：0；中心线补全 inner：364；物理上无法反推：136。
- 可进入 V2/面积支持：364 帧；每段视频按 5 个时间 bin，以既有 sharpness 取新池内最佳有效帧。
- 面积 guard 完全沿用主线的 formal-S0 p2：C≥0.085031、P≥0.317505。两个 capture 的 selected-frame 中位 C/P 都未落入该极端尾部，故按 `post-capture relaxed area guard v1` 不触发 retry；这是 guard 的正常“放行”，不是绕过 guard。
- B1 不变：`10 × median(L_px) / median(ring_px)`；`_004`=9.053602 mm（5/5 bin），`_005`=8.648740 mm（4/5 bin），组中位=8.851171 mm。

## 冻结后评价

- 该组：参考 8.8 mm，预测 8.851171 mm，绝对相对误差 **0.581489%**。
- 将它作为独立视频扩展接入已冻结 38 组主线后的 39 组范围：coverage 39/39，median absolute error **4.576786%**，MAE 0.613364 mm，p95 19.263320%，max 23.685499%。原 38 组主线文件未更改。

## 文件与复现

- 盲流程 SHA：`inputs/blind_sha256.txt`
- 每帧表：`tables/per_frame_blind.csv`
- 选择及面积证据：`tables/selected_frames_blind.csv`、`tables/capture_area_postcheck_blind.csv`
- 可视化：`gallery/index.html`（黄色=原 ring mask，青色=补全的 inner，紫色=V2 最终形状）
- 命令：`run_reacquisition.py extract` → `measure --device cuda` → `select` → `evaluate` → `report`。
