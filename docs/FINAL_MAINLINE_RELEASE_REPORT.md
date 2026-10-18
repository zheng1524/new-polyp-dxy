# DXY-MID正式主线发布报告（2026-10-18）

本报告对应PR #1的正式主线替换：V9.1 Geometry + 原图Mid Scale + 视频等权中位聚合。不是新算法研究。最终合并提交和GitHub CI见[PR #1](https://github.com/zheng1524/new-polyp-dxy/pull/1)及[默认main](https://github.com/zheng1524/new-polyp-dxy/tree/main)；没有把权重/视频未公开作为停留Draft的理由。

## 正式结果已公开

- [39组CSV](../mainline/tables/group_results.csv) / [中文Excel直接下载](../mainline/tables/group_results.xlsx?raw=true)：实际Paris、R5/R10、可信GT、预测、正负毫米/百分比误差、有效与失败帧、Normal/Q恢复/Strict、有效视频及10mm错误跨越。
- [79段视频表](../mainline/tables/video_results.csv)：77段有数值，2段无有效帧仍保留NaN；没有静默删除失败视频。
- [790帧数值与质量](../validation/results/frame_measurements.csv) / [33帧Ring失败明细](../validation/results/ring_failed_frames.csv)。删除的是本机路径列，未删测量数值或质量诊断。
- [总体指标](../mainline/tables/metrics_summary.csv) / [39组误差排序](../mainline/ERROR_RANKING.md)。

输入直接复用此前独立全790帧复算保存的DXY-MID结果，不从S5/G3构建。表格来源与原始文件SHA在`validation/results/PROVENANCE.json`。GT直接采用人工已确认的最终映射，未重审、替换或调节。组ID保留原关联，实际Paris及显示名称使用确认后字段。

CSV与Excel由同一DataFrame生成，39组无重复、全部有预测与GT。Excel全中文表头、首行冻结、误差降序，含视频/指标/阅读说明工作表；读取Excel逐单元格对照CSV与冻结数值通过。`signed_error=prediction-GT`，负值低估、正值高估；百分数已乘100。

## 冻结科研性能

| 项目 | 完整39组结果 |
| --- | --- |
| 独立数值覆盖 | 39/39组；757/790帧 |
| Ring失败 | 33帧，NaN；无失败组 |
| Normal / Q恢复 / Strict | 627 / 130 / 616帧 |
| 中位绝对相对误差 | 4.4113269936% |
| MAE | 0.7092238117 mm |
| P95 / 最大绝对相对误差 | 14.726267% / 16.565640% |
| 误差10%以内 | 30/39组（76.9231%） |
| 10mm分类 | 2组漏判，无误判；GT和预测均以≥10mm为阳性 |

误差按全部39组统计，不从图集中重新计算。39组是硅胶模型回顾性数据，不能宣称临床泛化。

| 审计项目 | 结果 |
| --- | --- |
| 目标组数 | 39 |
| 本算法独立成功 | 39组数值输出，757帧 |
| 本算法失败 | 0组；33帧Ring失败，保留NaN |
| 历史测量值回退 | 0 |
| 失败与原因 | 33帧见Ring失败CSV；无有效帧的2段视频也在视频表保留 |
| 最终统计样本 | 39/39组，帧质量分母790 |
| 是否用GT选择算法 | 否；算法与GT冻结不变。GT仅用于误差评价和选择展示误差档位 |

## 真实结果图集

[合作者图集](results_gallery/README.md)共11组，覆盖0～2%、2～5%、5～10%、10～15%、>15%五档，兼顾Ip/Is/IIa、R5/R10及正负误差。额外提供Ip 10.3 R5真实Ring失败帧（NaN）。完整39组排序图以星号标出所选11组。

每组五面板为原图、原mask、实际Mid、P/Q与实际chosen、真实测量轴。直接读取此前独立Measurement缓存，不运行拟合。原图/mask SHA256、Q原轮廓索引、chosen基矩阵、轴端点、Mid来源、同帧毫米值及组级结果均核对；11组全部通过。选择帧先靠近所属视频中位数，再按视频中位与组值的距离确定展示视频，帧选择不使用GT。组级与帧级值分别标注。

图集诚实保留了Q恢复、严格稳定不足及大误差：Ip 6.2 R5的20帧均为Q恢复、Strict为0；Ip 10.3 R5只剩1段有效视频，并发生10mm漏判；Is 10.5 R10同样漏判。最大相对误差的IIa 6.5 R10以确认后的实际Paris显示，未因标识误写删除或降低其误差。Normal/Strict通过不等于测量正确。

本轮只公开获准的派生图及科研数值，没有新增原始JPEG/mask、完整MP4、训练数据或权重。原27帧公开复现输入仍保留。

## 机器入口彻底替换

| 入口 | 现行为 |
| --- | --- |
| `mainline/CURRENT_POINTER.md` | 唯一正式DXY-MID |
| `mainline/MAINLINE_MANIFEST.json` | `OFFICIAL_MAINLINE`、V9.1/Mid/等权Fusion、冻结指标与表格SHA |
| `mainline/ERROR_RANKING.md` | DXY-MID正式39组结果与排序 |
| `mainline/scripts/build_mainline.py` | 只导出/校验DXY-MID冻结结果，不执行S5聚合 |
| README / AGENTS / 安装脚本 | DXY-MID默认入口与报告依赖一致 |
| S5/G3/DG3 | 历史归档，不参与默认测试或预测回退 |

旧S5的8个mainline文件原样移到`legacy/S5_mainline_20261018`，与前提交逐字节一致；旧S5/DG3测试移到legacy供手动追溯。不是删除历史。底层已验证代码复用与历史预测回退严格区分。

原62个数值源码/冻结输入及GT文件SHA不变。保留`DXY-MID-1.0.0rc1`这个数值实现标识用于逐帧复现，正式主线身份由manifest的`OFFICIAL_MAINLINE`确定。新增openpyxl 3.1.5仅服务Excel，未改变测量依赖或数学行为。

## 验收结果

- 当前默认自动测试20项：19通过，1项私有全量数据测试默认SKIP；完整790帧另行独立运行并通过，未用SKIP冒充全量验收。
- 从完整790帧图像/mask重新计算Ring、Boundary、Geometry、Scale、Gate、Stability及Fusion，39组全部匹配。只复用本次自身内容哈希缓存；4进程起跑后依据16逻辑CPU续为8进程，各BLAS/OpenMP线程为1。中断前241份自身缓存保留，无历史候选或预测输入。
- 全790帧毫米最大差为7.11e-15mm，视频中位最大差3.55e-15mm，组中位最大差3.55e-15mm；Ring参数最大差0。全部质量标记和失败原因一致。
- 既有公开单帧真实mask样例通过；完整Is双视频20/20帧另行重算通过，组级10.209387177421648mm。
- 39组CSV/Excel、11组图像数值及当前README链接校验通过。完整证据在[发布验证目录](../validation/evidence/final_mainline_20261018/)。
- GitHub CI运行相同默认测试、公开mask及真实失败CLI，并单独校验39组表；最终PR/main运行链接见PR #1的当前说明，不能以旧CI替代本次结果。

## 复现命令

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[plot,test,report]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/reproduce_public.py --case multivideo --output outputs/public-videos --workers 2
.venv/bin/python mainline/scripts/build_mainline.py --check
```

不带`--check`可重建正式CSV/Excel；这是冻结结果导出，不是从图像独立测量。

持有完整图像/mask者额外运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/reproduce39.py --data-root datasets/frozen39 --output outputs/fresh39 --workers 4
.venv/bin/python scripts/render_results_gallery.py --cache-run outputs/fresh39 --input-root datasets/frozen39 --selection docs/results_gallery/selection.json --output outputs/gallery
```

只从公开表重绘排序图不需要原始数据：

```bash
.venv/bin/python scripts/render_results_gallery.py --ranking-only --selection docs/results_gallery/selection.json --ranking-output outputs/ranking.png
```

## 分发限制与发布状态

正式主线可以合并到默认main，完整结果表、图集及公开Level 1案例均可访问。完整39组原图/mask与Level 2视频/精确权重仍有分发限制；没有借此伪称790帧已对外从原图完全复现。没有擅自上传额外原始资产，也没有创建宣称完整Level 2公开的Release。

本轮按现有迁移分支增量提交，更新原PR #1，经CI后设为Ready并正常合并main；不force push、不重写历史。最终main提交SHA、PR合并状态及CI回执由远端再次读取核对，见PR #1及本轮最终交付信息。完成后停止，不继续算法优化。
