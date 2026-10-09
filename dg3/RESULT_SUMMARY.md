# D-G3 参考环锚定短时序聚合：修复后结果总结

## 目的

在不改变 D-G3 的分割、ring fitting/尺度、G3 几何、Sharp5 中心帧、capture 中位数和正式 group fusion 的条件下，检验已选关键帧前后 ±3 帧能否改善**同一时间 bin 内**的直径估计。每一邻帧均独立运行冻结的 V2 与 G3；只有 bin 内估计替换，capture 仍要求至少四个有效 bin。

## 本次修复与可复现性

第一版邻帧处理把 `bin` 保存为普通字段，却没有提供既有 G3 调用所需的 `segment_id`，导致所有已通过 V2 的邻帧在 G3 步骤出现 `AttributeError`，不能评价。本次在独立目录 `ring_anchored_temporal_dxy_v2` 修复该数据适配：`segment_id = bin`。

修复版直接复用第一版已解码的 1,946 张原视频邻帧及其 INIT mask；没有重新分割、重选帧或改动旧结果。处理后有 1,931/1,946 帧 V2 有效、1,477/1,946 帧 G3 有效。无 `AttributeError`。其余无效主要是冻结 G3 的 `insufficient_multiarc_or_occlusion_endpoints`，属于原有几何可观测性拒绝，不会被时序聚合补成数值。

在读入 reference 前，所有盲测输入、邻帧测量、bin/capture/group 聚合结果已写入 `inputs/blind_sha256.txt`。B0 刻意不经邻帧流程重算，而是直接读取封存的 `quality_G3_frames_blind.csv`；它与冻结 D-G3 的 39 个 group prediction 最大绝对差为 **0 mm**。

## 方法

|方法|每个时间 bin 的估计|
|---|---|
|B0|冻结 D-G3 中心帧的单帧 Dxy。|
|B1|有效邻帧的 `median(R × p/r)`。|
|B2|有效邻帧上以 Huber 损失拟合过原点关系 `p = k r`，输出 `Rk`；`p` 和 `r` 的尺度由重复观测的稳健差分估计。|
|B3|与 B2 使用完全相同的有效邻帧；仅以相邻 `log(p/r)` 变化残差作固定 Huber 软降权。|

其中 `p` 为冻结 G3 的息肉像素直径，`r` 为同一校正坐标定义下的 ring 等效内径，`R` 为 ring 物理内径。所有方法随后均按冻结规则取 capture 内五个 bin 的中位数（至少四个有效 bin），并使用原正式 group fusion。

## 数据量与有效性

- 278 个 D-G3 中心 bin；每个 bin 最多 7 帧（中心帧 ±3）。
- B0 有 212 个有限 bin；B1 为 220；B2/B3 为 216。
- 61 个 video capture：B0 有 40 个 eligible；B1 因邻帧使两个 capture 达到四个有效 bin，计 42；B2/B3 均为 40。
- 所有 B0–B3 最终均有 39/39 group prediction；新增可用 capture 并未改变 coverage。

## 最终 39 组评价

|方法|coverage|Median absolute error|MAE|Mean absolute error|within 5%|within 10%|p90|p95|max|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|B0|39/39|4.981%|0.631 mm|6.848%|51.28%|74.36%|14.830%|20.220%|23.685%|
|B1|39/39|5.153%|0.637 mm|6.871%|48.72%|74.36%|14.848%|20.314%|23.685%|
|B2|39/39|4.981%|0.630 mm|6.776%|51.28%|74.36%|14.079%|20.416%|23.685%|
|B3|39/39|4.981%|0.630 mm|6.777%|51.28%|74.36%|14.079%|20.416%|23.685%|

## 配对结论

- **B1 不值得保留。** 相对 B0，5 组改善、7 组恶化、27 组不变；主指标从 4.981% 退化到 5.153%，MAE 也增加。
- **B2/B3 没有改善主指标。** 两者均为 6 组改善、11 组恶化、22 组不变。B2 虽将 MAE 降低约 0.0016 mm、p90 降低约 0.75 个百分点，但 median 不变，p95 从 20.220% 升至 20.416%。
- B3 和 B2 的整体结果几乎相同，说明这套邻帧 `log(p/r)` 软权重没有提供可辨认的额外物理信息。
- 最大单组改善出现在 B2/B3 的 `GP_Ip_2.9mm_closed_R5_single_refY`（绝对误差降低 3.863 个百分点）；这不足以构成全局规则的依据，且没有带来 median 改善。

## 科学解释与结论

邻近帧增加了重复观测，能在少数 bin 中恢复有效几何，也使 B2 的平均误差与 p90 略有下降；但在当前 D-G3 数据中，ring 与 polyp 的短时共同尺度关系不足以稳定地提供超越中心帧的普遍收益。可能原因包括 ring/息肉深度和姿态不完全一致，以及连续帧共享同一分割或 G3 几何偏差，使“更多帧”并非独立且无偏的重复测量。

因此本实验的负结果是明确的：**不建议把 B1、B2 或 B3 晋升到 D-G3 或正式主线。** 后续若继续研究时序，应先证明在局部片段中 `p/r` 的可预测稳定性，并独立验证；不应仅凭本轮 39 组探索性评价增加经验权重或复杂时序规则。

## 对应数据

- `tables/metrics_summary.csv`：总体和 Paris 分层指标。
- `tables/paired_group_comparison.csv`：B1/B2/B3 相对 B0 的逐组配对结果。
- `tables/neighbor_measurements_blind.csv`：全部邻帧盲测测量及有效性原因。
- `tables/bin_estimates_blind.csv`：每个 bin 的 B0–B3 数值。
- `Temporal_Dxy_results.xlsx`：group、capture、bin、邻帧和指标的汇总工作簿。
