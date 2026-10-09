# S5-CPAG-39：统一规则适用性审计

## 结论

**是。基础 38 个测量组由同一条 S5-CPAG 规则处理；没有 R5/R10、Paris 类型或 group-specific 分支。**

每段 video capture 均按同一流程：首尾去除 5% → 五个时间 bin → 每 bin
冻结 Sharp5 原始候选 → 当前 V2 产生 `L_px`/`ring_px` → 仅在 capture 满足
统一 C/P p2 触发条件时同 bin retry → 若有至少四个有效 bin 且
`p95(ring_px)/p5(ring_px)-1 >= 0.05`，以

`D_capture = R_mm * median(L_px) / median(ring_px)`

替换该 capture 的五条历史 video-keyframe 行；否则这些行不替换。最后一律对
所有保留 photo、未替换 keyframe 和已替换 capture 值取 group median。

`R_mm` 只是由环物理内径（R5=5、R10=10 mm）确定的尺度输入，不是不同的
决策规则。C/P 全局固定为 `C >= 0.08503077259013787`、
`P >= 0.3175049108502854` 的 capture-level p2 安全检查。

## 四个常被误认为“例外”的组

它们并没有被特殊修补；两段视频都未获得 B1 资格，因而走上述统一 fallback。

|组|冻结正式输入|S5 video bin|capture replacement|最终预测 mm|
|---|---:|---:|---|---:|
|`GP_IIa_9.5mm_closed_R10_single_refY`|3 photo + 10 keyframe|0/10 valid|无；`fewer_than_4_valid_bins`|10.111486|
|`GP_Is_10.5mm_closed_R10_single_refY`|3 photo + 10 keyframe|0/10 valid|无；`fewer_than_4_valid_bins`|9.898249|
|`GP_Is_4.8mm_closed_R5_single_refY`|1 photo + 10 keyframe|0/10 valid|无；`fewer_than_4_valid_bins`|5.039074|
|`GP_Is_9.79mm_closed_R10_single_refY`|3 photo + 10 keyframe|0/10 valid|无；`fewer_than_4_valid_bins`|9.961314|

四个预测都精确等于各自冻结正式输入行的 median。这不是“放宽门禁让不可信
geometry 进入”，而是 S5 对任何不满足 capture 替换资格的组采取的同一保守
fallback。公开审计包保留了这十个已选 bin 的 invalid 状态；完整 24-frame/bin
raw candidate pool、视频、mask 和模型是受控外部资产，按 README 的外部资产说明
提供后，可由 `scripts/run_raw_selector.py` 重新得到。

## 唯一不属于基础统一规则的组

`GP_Ip_8.8mm_closed_R10_single_refY` 不在原 38 组输入范围，故**不声称**它由
基础规则覆盖。它在 `ip88_extension/` 中作为明确、可审计的扩展：两段原始视频
各重采 250 帧；只有 native `inner_ellipse` 缺失时，以当前 outer 和 midline
ellipse 使用 `inner = 2*mid - outer` 恢复内环，再执行同一 INIT、V2、C/P、B1
和 group median。其独立盲输入 SHA、逐帧表和脚本均保留在该目录。

因此，可用一个准确表述概括主线：**S5 是 38 组统一规则 + 1 个输入范围之外、
显式标注的 Ip8.8 内环恢复扩展，而不是 39 组暗含例外规则。**

## 可复核文件

- `base38/inputs/frozen_protocol.json`：统一 CPAG 规则。
- `base38/scripts/run_post_capture_guard.py`：capture gate、B1 replacement 与
  group fusion 的可运行实现。
- `base38/tables/post_selected_frames_blind.csv`：五 bin 的 selected frame 与
  valid/C/P/L/ring 值。
- `base38/tables/capture_predictions_blind.csv`：每个 capture 的 eligibility 和
  fallback reason。
- `base38/tables/integrated_rows_blind.csv`：最终 photo/keyframe/replacement 行。
- `ip88_extension/RAW_VIDEO_REACQUISITION_REPORT.md`：唯一扩展的范围与拒绝条件。
