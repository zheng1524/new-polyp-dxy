# raw sharp5 的 capture 后置宽松面积安全检查

## 结果

本轮按“先完整完成 raw `result_sharp5`，再在每段视频（capture）的5个已选帧完成后检查”的解释实现。它只改动 **1/79** 个 capture、**3/395** 个 bin，保留 raw 的 **4.779%** median absolute error，同时将 raw 的最大尾部从 **68.852%** 降为 **23.685%**，p95 从 **21.706%** 降为 **19.380%**。coverage 仍为38/39，MAE 从0.785降为0.628 mm。

改变完全来自此前已审计的 `GP_Ip_12.0mm_closed_R10_single_refY`：68.852%→19.031%。因此这是一个有解释力的探索性安全 guard，但**不能声称已经证明泛化改善**；它目前仅在一个已知 tail failure 上触发，需独立视频/组验证。

## 与前一版“前置 p5 guard”的差异

前一版 [area_guarded_sharp5](../dxy_area_guarded_sharp5_20260922/AREA_GUARD_AUDIT.md) 对每个 bin 的候选在 sharpness 排名之前就施加 `C/P` p5 门槛（C≥0.600、P≥0.635）。结果有27个 retry、104个 missing，尽管移除了Ip12灾难尾部，median 升至5.067%。

本版不把面积作为日常抓帧门槛：

1. 先直接读取既有 raw `result_sharp5` 的395个原始选帧，未改变其 sharpness 选择。
2. 对每个 capture 的**原始已选、V2-valid且有C/P的帧**计算 `median(C)` 与 `median(P)`。
3. 仅当至少4帧可用，且该 capture 的任一中位支持落入正式S0的极低 p2 尾部时，才打开 repair。
4. repair 内也只处理自身低于 p2 的原始帧；其他帧原样保留。被处理帧从同一冻结 bin 的 V2-valid raw candidates 中按原 sharpness 次序找第一张同时通过 p2 C/P 的帧。
5. 若没有替代帧会记 missing；本次没有发生。随后 B1、`>=4`、ring dynamic、capture replacement、photo 和 group median 完全复用 raw。

这使面积支持成为“**capture 已经显示集体几何塌缩时的低频安全开关**”，而不是所有 frame 的强质量过滤器。

## 无GT阈值冻结

正式 S0/SYNC4 当前有效的 `original_V2_R0` view 中，317个具有冻结 V2 几何并可评估面积（50个R2 contact rescue无冻结ellipse/circle，未伪造面积）。全局统一 p2：

|量|阈值|来源|
|---|---:|---|
|C|0.085031|317个正式 S0/R0 C 的 p2|
|P|0.317505|317个正式 S0/R0 P 的 p2|

`C=area(M∩E)/area(M)` 检测小局部主体/错误component；`P=area(M∩(E\U))/area(E\U)` 检测大外推。面积值完全复用上一轮冻结的 [frame_area_support.csv](../dxy_area_guarded_sharp5_20260922/tables/frame_area_support.csv)，没有重跑 V2 或改变 ellipse。p2 不是按GT挑选：它在最终GT读取前由正式S0低尾分布一次性写入 [relaxed_thresholds_blind.json](inputs/relaxed_thresholds_blind.json)，协议与盲预测随后写入 [blind_sha256.txt](inputs/blind_sha256.txt)。

选择 p2 的目的不是一般性质量筛选，而是仅标记正式 S0 分布最底2%的明显面积不支持；p5 是前一版过强过滤的对照。

## 触发行为

79个 capture 中仅下列一个通过“至少4个已选可支持帧，且 capture median C/P 低于p2”的统一规则：

|group / capture|可支持原始帧|raw median C|raw median P|触发|
|---|---:|---:|---:|---|
|`GP_Ip_12.0mm_closed_R10_single_refY_005`|5|0.079370|0.863117|C < 0.085031|

该 capture 的三个个体低C raw选帧被替换，候选池、V2几何和sharpness均冻结：

|bin|原 raw C/P/Dxy|替代 rank|替代 C/P/Dxy|
|---:|---|---:|---|
|2|0.079 / 0.905 / 3.738mm|7|0.093 / 0.985 / 3.602mm|
|3|0.062 / 0.863 / 3.502mm|2|0.314 / 0.671 / 10.899mm|
|5|0.039 / 0.795 / 3.018mm|4|0.998 / 0.427 / 16.611mm|

Ip4 的单一小component和IIa14的局部大/小ellipse虽可在逐帧观察到低 C/P，但其 capture 级中位支持没有跌入p2，故没有改写 raw 选择。这个行为是刻意的：B1本身对孤立帧有鲁棒性，后置guard只针对**多帧、同方向的面积塌缩**，避免前置p5对正常/混合视频的过度介入。

全部 trigger 与原/新 frame 记录见 [post_selected_frames_blind.csv](tables/post_selected_frames_blind.csv)，capture 检查见 [capture_postcheck_blind.csv](tables/capture_postcheck_blind.csv)。候选可视化可直接查看上一轮同一冻结池的 [area support gallery](../dxy_area_guarded_sharp5_20260922/gallery/index.html)。

## 最终GT评价（仅冻结后）

|method|coverage|median abs %|MAE mm|mean abs rel %|within5|within10|p90 %|p95 %|max %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|S0|38/39|5.403|0.635|6.669|44.74|78.95|13.537|15.548|23.685|
|raw `result_sharp5`|38/39|4.779|0.785|7.971|52.63|76.32|14.898|21.706|68.852|
|post-capture relaxed guard|38/39|**4.779**|**0.628**|6.660|52.63|76.32|14.898|**19.380**|**23.685**|

只有Ip12一个group prediction变化：`3.738 → 9.716 mm`，绝对相对误差 `68.852% → 19.031%`。其余37组的预测逐值保持 raw 相同；53个符合B1资格的capture和26个fallback均不变。因此median、within5/within10和p90也保持 raw 相同；MAE/p95/max改善恰好反映该tail group被修复。

完整指标：[metrics_summary.csv](tables/metrics_summary.csv)。组级配对：[group_paired.csv](tables/group_paired.csv)。盲capture预测：[capture_predictions_blind.csv](tables/capture_predictions_blind.csv)。

## 完整冻结与复现

未修改 INIT segmentation/mask、ring fitting、V2 A/B/circle/ellipse、endpoint、L_px、ring_px、scale、sharpness、B1或任何正式 `result/` 内容。新脚本只读取上一轮已缓存的C/P并实施后置选择：[run_post_capture_guard.py](scripts/run_post_capture_guard.py)。

```bash
/home/liu/miniconda3/envs/息肉检测/bin/python scripts/run_post_capture_guard.py blind
/home/liu/miniconda3/envs/息肉检测/bin/python scripts/run_post_capture_guard.py evaluate
```

脚本 SHA256：`c7fb012601fa325552ba7e7371e2d510aba2370a64079d7256a360544dd53726`。
