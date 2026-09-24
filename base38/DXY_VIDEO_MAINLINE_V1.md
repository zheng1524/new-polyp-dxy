# Dxy 视频测量主线 v1：`result_sharp5 + post-capture relaxed area guard`

**状态：已由用户于2026-09-22确认为新的研究视频测量主线。**

主线标识：`dxy_video_result_sharp5_post_capture_relaxed_area_guard_v1`。唯一机器可读定义是 [MAINLINE_MANIFEST.json](MAINLINE_MANIFEST.json)；唯一执行脚本是 [run_post_capture_guard.py](scripts/run_post_capture_guard.py)。该主线的目的不是修改 V2 测量，而是在冻结 V2、尺度、B1 和正式融合后，为原始视频清晰度选帧加入一个极低频的、capture 级几何安全检查。

> 边界：这是 SYNC4/S0、38/39 视频评估域上的**研究视频主线**。`result/` 中仍保留其历史、不同范围的正式结果包，未被本主线改写，也不能把两者的31/40与38/39指标混合比较。

## 1. 一句话定义

每段视频仍先按 raw `result_sharp5` 选出5张最高 sharpness 的原始候选；只有当这5张中至少4张可测帧的**中位面积支持**落入正式 S0 的最低2%时，才对该视频中自身面积支持极低的帧进行重选。重选仍只从同一冻结 bin、同一候选池中按原 sharpness 顺序找第一张通过宽松面积条件的 V2-valid 帧。其余所有帧、视频、photo 与 group 融合保持原样。

## 2. 固定数据范围

|项目|冻结值|含义|
|---|---:|---|
|正式 Dxy 视频|79|只使用既有正式测量视频，不含 segmentation-training 专用视频|
|时间 bin|5 / video|沿用 raw selector 的均匀5段划分|
|每 bin 原始候选|最多24|总计9,480帧；候选图片、INIT mask、ring JSON均为既有缓存|
|V2-valid 原始候选|6,753|仅这些候选可用于 repair；不重跑 V2|
|raw `result_sharp5` 最初选择|395帧|79×5；其中284帧 V2-valid、111帧 V2-invalid|
|immutable formal rows|367|由旧 immutable manifest 的 current-valid 行组成|
|最终可比较 group|38/39|与S0和raw完全相同的coverage；第39组本来没有正式可测预测|

原始候选来自 [raw_candidate_features_blind.csv](../dxy_raw_video_frame_selection_20260921/tables/raw_candidate_features_blind.csv)，raw 选帧来自 [selected_frames_blind.csv](../dxy_raw_video_frame_selection_20260921/tables/selected_frames_blind.csv)。所有上游哈希列于 [MAINLINE_MANIFEST.json](MAINLINE_MANIFEST.json)。

## 3. 完全冻结、绝不重新计算的部分

下列内容不是本主线的可调参数：

|层|冻结对象|
|---|---|
|图像与分割|原始候选图片、正式 INIT segmentation/mask、label `1=polyp`、`2=ring/occlusion`|
|ring与尺度|ring identity、ring JSON、inner/outer ellipse、ring_px、mm_per_px、scale|
|V2|A/B 候选、circle/ellipse、最终 shape、endpoint、`L_px`、逐帧 Dxy、valid/failure reason|
|原始抓帧|每bin按既有 sharpness 降序、frame index 升序；raw 第一名可以是V2-invalid，保持历史定义|
|视频聚合|B1、至少4有效bin、ring dynamic range门槛|
|正式融合|video capture replacement、photo保持S0、formal row median/group median|
|参照|GT定义、原始数据、旧实验与 `result/`|

本主线**不**训练分割、不运行SAM、不重拟合ring/ellipse、不校正ellipse大小、不平滑时间序列、不改scale、不改Paris routing、不改group fusion。

## 4. raw `result_sharp5` 基础层

原始层的规则先完整执行，不受面积值影响：每个 `(capture, segment_id)` 在最多24个 raw 候选中选择 sharpness 最大者；sharpness 并列按较小 frame index。这个选择发生在 V2 成功/失败之后已经写入缓存的意义上，但不先过滤 V2-invalid 帧；若第一张无效，它按原 raw 规则保留为该 bin 的无效选择。

这是重要的保守点：主线不是新的 selector，也不会因为一个普通几何分数而替换 raw 的优先级。面积检查发生在一个 capture 的5个 raw 选择已经确定之后。

## 5. 双向面积支持：仅作后置安全证据

令：

- `M`：完整、冻结的当前帧 label-1 polyp mask；不是core子集，也不是重新分割的mask。
- `E`：冻结 V2 最终选择的 circle 或 ellipse 内部区域。V2的crop坐标先转换回原mask坐标；没有重拟合。
- `U`：已存在的 ring unknown 定义：`dilate(label-2, disk(max(3, 0.08×sqrt(inner_major_radius×inner_minor_radius))))` 与 fitted `(outer ring fill − inner ring fill)` 的并集。

计算：

```text
C = area(M ∩ E) / area(M)
P = area(M ∩ (E \ U)) / area(E \ U)
```

`C`低表示ellipse只解释了小块mask：典型是小圆、错误component或只包住局部主体。`P`低表示ellipse大部分可观察内部没有mask：典型是过大圆、短弧向背景外推。环遮挡 `U` 不进入 `P` 分母，因此不会把真实未知区域错当背景。

本主线不另写几何优化器：面积定义复用 [core occupancy `area_consistency`](../dxy_core_ellipse_occupancy_20260921/scripts/audit.py)，unknown构造复用 [旧occupancy test](../dxy_ellipse_occupancy_measurement_test_20260921/scripts/evaluate_gate.py)，ellipse/ring raster复用 [V2工具](../dxy_head_fitting_v2_20260918/scripts/v2.py)。逐帧冻结结果见 [frame_area_support.csv](../dxy_area_guarded_sharp5_20260922/tables/frame_area_support.csv)。

## 6. 阈值如何冻结

阈值来源不是视频GT，也不是case经验值。仅从正式 S0/SYNC4 current-valid 的 `original_V2_R0` view 计算面积分布：

|来源|数量|处理|
|---|---:|---|
|有冻结 V2 circle/ellipse 的 R0 valid view|317|用于统一分位数|
|`human_approved_R2_contact_rescue` valid view|50|无相应冻结 geometry；明确不计算、不伪造C/P|

取全体317 view 的低 `p2`：

```text
C_thr = 0.08503077259013787
P_thr = 0.3175049108502854
```

这是统一低尾“极端不支持”阈值，而非上一版 p5 的一般质量门（上一版 `C=0.600030`, `P=0.634953`）。无 R5/R10、Ip/non-Ip、尺寸、Paris 或 group 分支。p2值、分位数来源和`n=317`冻结在 [relaxed_thresholds_blind.json](inputs/relaxed_thresholds_blind.json)。

## 7. 后置 capture 触发规则（主线核心）

先在每个 capture 的5个 raw 选择里取集合：

```text
S = {raw-selected frame | V2-valid 且 C/P已成功计算}
```

只有下列条件**同时**成立，该 capture 才开启repair：

```text
|S| >= 4
AND ( median(C in S) < C_thr OR median(P in S) < P_thr )
```

这一步解释了“为什么是后置且不极端”：单个坏帧、或正负两类坏帧混杂时，B1的中位数往往已经稳定；它们不会让整个capture的中位支持进入最底2%。只有多数已选帧共同体现同方向几何不支持，才把该capture视为安全风险。

对于被触发的 capture，再逐bin判定：

```text
individual_bad = (C_raw < C_thr OR P_raw < P_thr)
```

- `individual_bad=False`：raw frame 值、有效性和bin位置完全保留。
- `individual_bad=True`：在该**同一bin**的冻结 V2-valid candidate中，筛 `C>=C_thr AND P>=P_thr`，按既有 sharpness 降序、frame index 升序取第一张。
- 无通过候选：该bin标记 missing；本冻结运行中为0个。
- capture未触发：即使某一帧本身C/P低，也绝不触碰原raw选择。

因此没有“看最终GT后决定是否修”的分支；唯一输入是 raw 选择、冻结C/P和固定p2。

## 8. B1与正式融合（保持不变）

repair 后，对每个 capture 的 selected-valid bins 应用原B1：

```text
D_capture = R_mm × median(L_px) / median(ring_px)
```

其中 `R_mm` 直接由已有 ring code（R5=5mm，R10=10mm）提供；未估计新的mm/px。capture只有同时满足：

```text
n_selected_valid >= 4
p95(ring_px) / p5(ring_px) - 1 >= 0.05
```

才具有替换资格。否则回退 S0 的正式 video 行。对 immutable manifest 的367个current-valid行，只替换具有资格的 `video_keyframe` capture 行；photo行不变；每个group取全部正式行的中位数。该运行中：

|项目|数值|
|---|---:|
|79个视频capture中符合B1资格|53|
|fallback capture|26|
|实际可替换的正式video capture|48|
|被替换的formal行|226|
|有0/1/2个video capture进入最终group的group数|10 / 8 / 20|

## 9. 冻结运行实际上做了什么

|审计量|值|
|---|---:|
|capture trigger|1 / 79|
|post retry|3 / 395 bin|
|post missing|0|
|未触发、逐bin原样保留|392 / 395|
|最终预测相对raw改变的group|1 / 38|

唯一触发 capture：`GP_Ip_12.0mm_closed_R10_single_refY_005`。

|原始可支持bin数|raw median C|raw median P|触发原因|
|---:|---:|---|
|5|0.079370|0.863117|`median C < 0.085031`|

被重选的三个 bin：

|bin|raw `C/P/Dxy`|新 sharpness rank|新 `C/P/Dxy`|
|---:|---|---:|---|
|2|0.079 / 0.905 / 3.738 mm|7|0.093 / 0.985 / 3.602 mm|
|3|0.062 / 0.863 / 3.502 mm|2|0.314 / 0.671 / 10.899 mm|
|5|0.039 / 0.795 / 3.018 mm|4|0.998 / 0.427 / 16.611 mm|

该capture的B1从3.738 mm变为9.716 mm；ring dynamic仍合格（raw 0.145790，repair后0.144007）。Ip12 group预测同为3.738→9.716 mm。所有其他37个group预测与raw逐值一致。具体可复核行：[post_selected_frames_blind.csv](tables/post_selected_frames_blind.csv)、[capture_postcheck_blind.csv](tables/capture_postcheck_blind.csv)、[capture_predictions_blind.csv](tables/capture_predictions_blind.csv)、[group_predictions_blind.csv](tables/group_predictions_blind.csv)。

## 10. GT隔离与最终评价

在 `blind` 阶段写入协议、阈值、capture判定、选帧、B1和group盲预测后，生成 [blind_sha256.txt](inputs/blind_sha256.txt)。`evaluate` 先验证其中每个盲文件哈希，再从既有表读取GT，GT未进入候选、C/P、p2、capture trigger、repair或融合。

误差公式：

```text
relative error (%) = 100 × (prediction_mm − reference_mm) / reference_mm
absolute relative error = abs(relative error)
```

|方法|coverage|median abs %|MAE mm|mean abs rel %|median signed %|within5|within10|p90 %|p95 %|max %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|S0|38/39|5.403|0.635|6.669|0.896|44.74|78.95|13.537|15.548|23.685|
|raw `result_sharp5`|38/39|4.779|0.785|7.971|0.868|52.63|76.32|14.898|21.706|68.852|
|**主线 v1**|38/39|**4.779**|**0.628**|6.660|0.868|52.63|76.32|14.898|**19.380**|**23.685**|

唯一变化组 Ip12 的绝对相对误差为 `68.852% → 19.031%`。所以median、within5、within10、p90保持raw不变；MAE、p95、max改善均来自该已审计tail group。完整逐组数据：[group_paired.csv](tables/group_paired.csv)，完整指标：[metrics_summary.csv](tables/metrics_summary.csv)。

## 11. 为什么它比前置 area guard 更适合作为主线

|性质|前置 p5 area guard|本主线 capture后置 p2|
|---|---|---|
|何时使用面积|每个候选入选前|raw五帧选完后|
|阈值|C≥0.600、P≥0.635|只识别最低2%：C≥0.085、P≥0.318|
|影响范围|27 retry、104 missing|3 retry、0 missing|
|改变group|5|1|
|median abs %|5.067|4.779|
|max %|40.972|23.685|

前置p5把“面积合理”误当成所有视频帧的普遍质量要求，干扰了raw的低median优势；主线v1只把面积用在足以影响B1的集体几何塌缩上。它依靠的是已有B1的中位数鲁棒性与非常窄的安全触发，而不是叠加新的frame selector。

## 12. 已知限制与维护规则

1. **证据集中于一个已审计组。** 唯一改变预测的是Ip12，尾部改善不是跨多组统计收益；不得把它宣传为已证实的普适精度提升。
2. **阈值不可再用现有GT微调。** 不得为了把Ip12从19.0%继续压低而改变p2、`>=4`或retry规则；任何改变必须创建新的实验版本。
3. **R2 geometry缺失。** 50个R2 contact rescue view没有冻结circle/ellipse，未用于阈值标定；不要补算或猜测其C/P。
4. **面积只识别几何不支持。** 高C/P不证明尺度、ring或视频中的主体语义一定正确；它不能替代V2质量审阅。
5. **不要修改 `result/`。** 本研究主线和历史结果包范围不同。若未来要发布到正式包，必须另行完成scope一致性、独立集验证、可复现打包与用户批准。
6. **主线升级纪律。** 后续改动只能在新 `experiments/dxy_*` 目录开展；不得覆盖本目录的manifest、盲预测或已冻结阈值。

## 13. 复现与完整性校验

首次重建：

```bash
cd /home/liu/polyp_research
PY=/home/liu/miniconda3/envs/息肉检测/bin/python
$PY experiments/dxy_post_capture_relaxed_area_guard_20260922/scripts/run_post_capture_guard.py blind
$PY experiments/dxy_post_capture_relaxed_area_guard_20260922/scripts/run_post_capture_guard.py evaluate
```

只读验证已冻结主线：

```bash
$PY experiments/dxy_post_capture_relaxed_area_guard_20260922/scripts/verify_mainline.py
```

验证器检查主脚本、raw输入、面积缓存、immutable manifest、盲输出哈希；还检查395个bin、1个trigger capture、3个retry、0 missing、未触发capture逐帧未改，以及仅Ip12 group prediction变化。主脚本 SHA256：`c7fb012601fa325552ba7e7371e2d510aba2370a64079d7256a360544dd53726`。
