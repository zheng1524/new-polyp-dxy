# 可复现 Dxy 视频测量：S5-CPAG-39 与 D-G3 冻结候选

仓库保存两个可独立核查的冻结候选：正式 S5-CPAG-39，以及未晋升的 D-G3 研究候选。S5 仍是当前冻结主线；D-G3 被并列归档，以便复核其几何收益与总体指标未改善这一负面结果。

冻结结果：coverage **39/39**，median absolute error **4.577%**，MAE **0.613 mm**，p95 **19.263%**，max **23.685%**。结果证据、盲流程 SHA 与关键可视化均被保留；完整原始视频、完整患者图像集、模型权重和大型候选缓存不上传。D-G3 目录只有用户明确授权的极小代表原图子集。

|候选|角色|coverage|median absolute error|MAE|p95|max|
|---|---|---:|---:|---:|---:|---:|
|S5-CPAG-39|正式冻结主线|39/39|4.577%|0.613 mm|19.263%|23.685%|
|D-G3 / B0|冻结研究候选，未晋升|39/39|4.981%|0.631 mm|20.220%|23.685%|

详见 [`dg3/README.md`](dg3/README.md)。D-G3 含 5 张已授权代表原图和 5 张 S5-vs-D-G3 汇报 PNG；它们受数据治理约束，不代表完整数据集，也不得任意再分发。

## 算法流程

1. **INIT segmentation**：冻结的多类别模型把每帧标成背景、polyp、ring。
2. **ring geometry**：清理 ring mask，按 component 找 outer/inner contour，以现有 RANSAC ellipse fitting 得到环几何。
3. **scale recovery**：以已知 R5/R10 内径，将 inner/outer ellipse 做 affine rectification，恢复 `mm_per_px`。
4. **V2 polyp measurement**：在 ring occlusion 下构建 A/B 候选，拟合圆和椭圆；`process_view`、`choose`、`choose_geometry` 以固定几何质量规则选择最终长度。
5. **raw Sharp5**：每个视频剔除首尾 5%，分 5 个时间 bin；每 bin 取最高既有 sharpness 帧。B1 为 `R_mm × median(L_px) / median(ring_px)`，至少 4 个有效 bin 且 ring 动态范围不少于 5%。
6. **CPAG**：先完成 Sharp5。仅当某 capture 已选帧的 C/P 面积支持中位数落入 formal-S0 p2 极低尾，才在同一 bin 以 sharpness 顺序重试面积支持合格帧。阈值固定为 C≥0.08503077259013787、P≥0.3175049108502854。
7. **formal fusion**：只替换符合 B1 条件的 video capture；photo/S0 保持；按 group median 融合。
8. **Ip8.8 R10**：唯一此前未覆盖的组从两段原始视频密集重采。当 ring 因息肉贴住内缘而没有闭合内孔时，用当前 outer ellipse 和 skeleton-midline ellipse 按 `inner = 2×mid − outer` 补全 inner。V2、C/P、B1 不改变，旧 V2 R3 不使用。

基础 38 组的“同一规则 + 统一 fallback”边界，及四个没有 video replacement
的组为何仍是同一 S5 路径，见 [`base38/S5_UNIFIED_RULE_AUDIT.md`](base38/S5_UNIFIED_RULE_AUDIT.md)。
Ip8.8 R10 是唯一明确标注、可从基础规则范围中排除的输入扩展，而非隐式特例。

## 目录

```text
src/
  polypseg/                 INIT 模型加载、ring fitting、标定与仿射尺度工具
  dxy_s5cpag/               V2、scale、C/P、B1、centerline fallback、指标
  dxy_dg3/                  D-G3 的完整自定义源码闭包（Quality-aware、ring-adjacency、G3）
model_code/init_segmentation/ 冻结 INIT 的 UNet3+ 自定义模型源码（无权重）
scripts/run_raw_selector.py 原始视频候选、Sharp5、B1 与 formal fusion
base38/                     38 组 S5-CPAG 冻结协议、脚本、盲输出和验证
ip88_extension/             Ip8 原视频重采集代码、证据与结果
mainline/                   39 组最终 manifest、误差 CSV/图、构建脚本
dg3/                        D-G3 冻结盲表、指标、报告和有限汇报素材
configs/                    外部数据/模型路径说明
tests/                      无数据 smoke tests
```

## 安装

建议 Python 3.10。版本记录来自实际冻结环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

CUDA/PyTorch 的安装需按你的平台替换 `torch==2.5.1+cu121`；其余版本不要为方便而任意升级。

## 模型与外部资产

需要的模型是原项目的**冻结 INIT multiclass segmentation model**，通过 `polypseg.common.load_pretrained_model`、Transformers `AutoModel(..., trust_remote_code=True)` 载入。其 UNet3+ 自定义源码已提交在 `model_code/init_segmentation/`，并提供了去本机路径的 `config.example.json`；将该目录内容、由数据持有者确认的 `config.json` 和 `model.safetensors` 或 `pytorch_model.bin` 一同放到权重目录。

设置位置：

```bash
source configs/runtime.example.env
export DXY_MODEL_DIR=/path/to/init-segmentation-model
```

缺失该权重时，不能执行新视频的 `measure`/INIT segmentation；已提交的 CSV 仍可重建最终 S5-CPAG-39 表、指标和误差图。原始 mp4 缺失时，不能执行 raw candidate extraction 或 Ip8 的重采集。详细布局见 [`configs/data_layout.md`](configs/data_layout.md)。

## S5-CPAG-39 复现顺序

完整数据与模型已按配置提供时：

```bash
python scripts/run_raw_selector.py extract
python scripts/run_raw_selector.py measure --device cuda
python scripts/run_raw_selector.py select
python scripts/run_raw_selector.py evaluate

python base38/scripts/run_post_capture_guard.py blind
python base38/scripts/run_post_capture_guard.py evaluate

python ip88_extension/scripts/run_reacquisition.py extract
python ip88_extension/scripts/run_reacquisition.py measure --device cuda
python ip88_extension/scripts/run_reacquisition.py select
python ip88_extension/scripts/run_reacquisition.py evaluate

python mainline/scripts/build_mainline.py --output-root outputs/mainline
```

路径均由 `DXY_*` 环境变量控制，代码中没有机器特定路径。原始实验的盲输出 SHA 在 `base38/inputs/blind_sha256.txt` 和 `ip88_extension/inputs/blind_sha256.txt`；由于公开仓库只保留必要审计包，完整原始输入应由有权限的数据持有者提供。

## 无原始数据验证

```bash
python -m unittest discover -s tests -v
python mainline/scripts/build_mainline.py --output-root /tmp/dxy-mainline-check
python -m dxy_dg3 verify
```

前者测试核心几何、C/P、B1 和中心线重建；后者只用已提交的表重建最终 group 排序。最终结果见 [`mainline/ERROR_RANKING.md`](mainline/ERROR_RANKING.md)。

## 冻结范围

不要修改 `base38/`、`ip88_extension/inputs/*sha*` 或 `mainline/` 内的冻结结果以“改善”指标。新的方法必须在新的实验目录进行，并明确报告与 S5-CPAG-39 的 paired 比较。
