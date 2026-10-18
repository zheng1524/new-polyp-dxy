# 真实公开复现案例：6组27帧

这里保存原始JPEG与同帧单通道0/1/2 mask，均来自已冻结790帧池，没有重新压缩、改分辨率、修改分割或重选帧。约3.74 MiB的原始资产由维护者本轮授权增量公开，仅为实验室硅胶模型，不是患者影像。来源与检查见[DATA_PROVENANCE.json](DATA_PROVENANCE.json)，逐文件哈希见[SHA256SUMS.csv](SHA256SUMS.csv)。代码及这些文件已经在仓库中，不需要额外数据下载或权重。

## 复现范围

| 案例 | 帧数 | 用途与统计范围 |
| --- | ---: | --- |
| Is 10.5 R5 | 20 | 两段视频各10个冻结选帧，完整原组；普通测量及10mm附近 |
| Ip 6.4 R10 | 1 | Ip主体及实际测量方向，机制子集 |
| IIa 14 R10 | 1 | 扁平主体、P/Q共存，机制子集 |
| IIa 9.5 R10 | 1 | P=0、环内遮挡推断，机制子集 |
| Ip 6.2 R5 | 2 | Q冲突恢复、短Q弧过滤，机制子集 |
| Ip 4.7 R5 | 2 | 一帧小孔修复后可用、一帧Ring硬失败，机制子集 |

没有按GT误差筛选。除完整Is组外，按预先指定机制、质量身份筛选后的时间顺序选取首帧。完整Is组是全部20/20帧，不是只保留优质帧。其他5组的子集聚合不是它们原完整视频/组的结果。

既有单帧示例仍在[`../mask_demo`](../mask_demo/README.md)。这里的27帧与既有样例是不同输入。

## 运行（从仓库根目录）

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[plot,test]'

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/reproduce_public.py \
  --case all --output outputs/public --workers 2 --render
```

可替换`--case`：`mechanisms`只跑8个机制帧，`multivideo`跑完整20帧，`failure`只跑真实失败帧。脚本先执行相同正式`dxy_mid.batch.run_manifest`，之后才读取对应expected CSV；不读取GT，不从expected生成预测。

预期全部27帧结果：26帧有效、1帧Ring失败；24 Normal、2 Q恢复、24 Strict。6/6子集组有数值，但只有1/1组是完整冻结组。Ring失败案例单独运行时为0/1数值组，Dxy为NaN、`ring_recovery_unreliable`；这是预期失败，不是脚本错误或补值。

完整Is组的预期结果：

| 视频 | 有效帧 | 视频中位数（mm） |
| --- | ---: | ---: |
| `GP_Is_10.5mm_closed_R5_single_refY_004.mp4` | 10/10 | 10.274870 |
| `GP_Is_10.5mm_closed_R5_single_refY_005.mp4` | 10/10 | 10.143904 |
| 视频等权组级中位数 | 20/20 | **10.2093871774** |

视频是采集标识，目录内没有完整MP4。我们复现的是原视频已冻结的全部选帧测量及聚合，而非从MP4重新解码、选帧和分割。

输出包含`frames.csv`、`videos.csv`、`groups.csv`、Strict诊断对照、`regression.json`及`reproduction_scope.json`。`--render`输出每组一张简洁分布PNG。

## 可审计文件

- `frames.csv`及三个子集manifest：只含正式输入字段，路径相对于本目录。
- `case_metadata.csv`：group/video/source_id、原帧号、时间戳、Ring规格、GT、Paris、预测、质量、失败原因及选例理由；只用于展示/评价。
- `GT_evaluation_only.csv`：复用人工确认映射；不传给预测组件，不重审或还原旧名称。
- `expected_*.csv`：从原39组冻结评价CSV按source_id截取，计算结束后才比较。
- `scope.csv`：每组公开/原冻结帧数、视频数、是否完整组。
- `visualization_selection.json`：过程图的明确source_id，不能被误认为新的选帧算法。

GT误差不用于选择这些帧或参数。这是机制演示与冻结回归，不是新的泛化性能研究，也不代表随机抽样准确率。

## 从真实求解过程重新生成PNG

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/render_public_process.py \
  --cases examples/reproduction_cases --run outputs/public --output outputs/public-process
```

诊断入口通过只读Python返回事件取得冻结函数本次实际产生的中间量，额外执行几帧正式测量；没有可视化专用拟合。每图附source_id和数值CSV，点位、长轴、Mid尺度及毫米值自动核验。已公开的图与说明见[过程图集](../../docs/visualization/README.md)。

完整790帧、完整视频、分割权重及训练数据不包含在此公开授权范围中。待决定资产见[资产清单](../../docs/PUBLIC_ASSETS_PENDING.json)。
