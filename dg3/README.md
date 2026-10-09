# D-G3：冻结的并列研究候选

`D-G3` 是 **Quality-aware Sharp5 + G3 遮挡感知椭圆** 的最终冻结实验包；它与 S5-CPAG-39 并列保留，**但没有取代 S5**。本包用于复核负面结果、复现实验算法，以及制作有限的内部汇报素材。

|冻结候选|coverage|median absolute error|MAE|p95|max|
|---|---:|---:|---:|---:|---:|
|S5-CPAG-39|39/39|4.577%|0.613 mm|19.263%|23.685%|
|D-G3 / B0|39/39|4.981%|0.631 mm|20.220%|23.685%|

D-G3 的部分 G3 几何在可视化中更贴合主体，但没有转化为总体中位误差收益；因此它是冻结的对照/研究候选，而非正式主线。

## 算法

1. 在原视频首尾各剔除 5% 后分为 5 个时间 bin；从每 bin 的既有 Sharp5 候选中保留 Top-3。
2. 以息肉 ROI、息肉边缘和 ring 边缘的 Tenengrad/Laplacian 进行局部清晰度筛选；再以 ring 连续性、frontality 和 ring-adjacency 可信外缘进行质量门控与字典序排序。
3. INIT 分割、ring fitting、尺度恢复和原 V2 A/B 主体候选保持冻结。ring-adjacency 只对**原始 polyp contour**标记 `high_confidence`、`uncertain`、`ring_cut`；ring cut 不作为拟合点。
4. G3 对可信多弧段进行确定性多初值椭圆优化，以边缘 Huber loss 为主，并加入遮挡端点切线、ring unknown 与可见背景软约束。多解/观测不足时拒绝，而不是强制输出。
5. 每帧计算 `Dxy_i = ring_mm × L_px_i / ring_px_i`，每 capture 对有效帧取中位数（至少 4 个 bin），之后沿冻结 formal group fusion/fallback 汇总。

## 内容与可复现层级

```text
src/dxy_dg3/                  D-G3 自定义源码闭包
  frozen_v2.py                原 V2 circle/ellipse/A/B 代码的冻结副本
  boundary_recovery.py        completion 诊断（不把新边当观测）
  edge_completion.py          原始 contour 的 completion 审计
  ring_adjacency.py           ring 两侧邻接三分类
  multiarc.py                 多弧段 G2 对照与固定 fusion
  g3.py                       遮挡感知 amodal ellipse
  quality_selection.py        Quality-aware Sharp5
  layout.py                   无机器路径的资产/输出配置
dg3/input_data/               必要的盲表、formal manifest 与 Ip8 扩展表
dg3/tables/                   278 个选择帧、帧/capture/group 盲结果及最终评价
dg3/report_assets/            5 张已授权代表原图和 5 张 S5-vs-D-G3 PNG
```

不提交完整原始视频、全部候选帧/缓存、患者数据集、模型权重或任何凭据。`report_assets/raw_frames/` 仅是获授权的 5 张代表性原图；`report_assets/visualizations/` 是基于冻结结果生成的汇报叠加图。请勿将其再分发到不具备数据治理授权的环境。每个已提交素材的 SHA256 在 `report_assets/SHA256SUMS`。

## 无原始资产的验证

安装仓库依赖后，以下命令只从已提交的最终表重算 D-G3/B0 指标：

```bash
python -m dxy_dg3 verify
```

它应输出 39 组、median `4.980738446606654%`、MAE `0.6310721590553092 mm`、p95 `20.219854095804195%`、max `23.685498698929642%`。

## 用私有资产重跑几何

完整盲选/几何重跑需要数据持有者提供原始视频或已抽取的候选帧、对应 INIT mask 与 ring JSON，以及 README 之外的 INIT 权重。把它们放在独立的资产根目录并设置：

```bash
export DXY_DG3_ASSET_ROOT=/secure/path/to/assets
export DXY_DG3_DATA_ROOT=$PWD/dg3/input_data
export DXY_DG3_OUTPUT_ROOT=/secure/path/to/dg3-reproduced
```

提交表中的相对指针遵循：`dxy_raw_candidates/images/`、`dxy_raw_candidates/masks/`、`dxy_raw_candidates/ring_fit/json/`、`raw/`、`formal_assets/masks/` 与 `formal_assets/ring_json/`。源码只写入 `DXY_DG3_OUTPUT_ROOT`，绝不会覆盖本目录的冻结表。权重缺失时不能重新执行 INIT segmentation；原始视频缺失时不能重新抽帧；但冻结表验证仍可运行。

资产齐全时，按以下顺序重新生成盲选择和几何表；每一步只使用前一步的盲结果：

```bash
python -m dxy_dg3.edge_completion
python -m dxy_dg3.ring_adjacency
python -m dxy_dg3.multiarc blind
python -m dxy_dg3.g3 blind
python -m dxy_dg3.quality_selection blind

# 上述盲结果和 SHA 固定后，才允许离线评价：
python -m dxy_dg3.quality_selection evaluate
```

`src/polypseg/`、`src/dxy_s5cpag/` 和 `model_code/init_segmentation/` 是 D-G3 复用的 INIT、ring/scale、V2 与模型代码；没有任何运行时 import 指向旧本地工程。

## 结果溯源

- `inputs/frozen_protocol.json`、`inputs/quality_original_blind_sha256.txt`：Quality-aware D-G3 的原盲流程证据。后者是旧环境生成时的原始 SHA 清单。
- `inputs/quality_portable_sha256.txt`：本仓库的可移植副本 SHA。`quality_candidate_features_blind.csv` 仅把机器绝对路径替换为相对资产指针，所以它与原 SHA 的一项不同；数值列、选择、几何与最终预测没有被改动。
- `inputs/temporal_protocol.json`：后续短时聚合审计；B0 直接读取冻结 D-G3 结果而没有重算。未提交完整邻帧缓存，因此不把它的旧 SHA 清单冒充为当前可验证证据。
- `tables/group_predictions_evaluated.csv`：B0–B3 的最终 paired 评价；D-G3 即 `method=B0`。
- `RESULT_SUMMARY.md`：B1/B2/B3 没有优于 B0 的负面结果。
