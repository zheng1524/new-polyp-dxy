> 这是公开案例阶段的保留记录，其Draft/停止条件已由后续正式发布任务取代。当前状态见[正式发布报告](FINAL_MAINLINE_RELEASE_REPORT.md)。原数据来源与验证证据保持不变。

# DXY-MID公开复现案例与真实过程图（2026-10-18）

## 本轮结果与范围

在既有`dxy-mid-migration`分支及PR #1增量交付，没有重新迁移主线。开始时远端分支为`991ea2017ef522decf76b4521b4ce7ea8d132121`，main为`518c83f2695cbee4c73078aac194bc85027decbb`，PR仍是OPEN/Draft。本轮不合并main，不创建Release。

新增6组27帧原始JPEG与对应0/1/2多类mask，54个原始文件共3,921,199字节（3.74 MiB），支持无需模型权重的Level 1独立测量。Is 10.5 R5完整包含原冻结组20/20帧、2/2视频；其余5组只公开机制子集。原既有单帧示例保留。

正式版本继续是V9.1 Geometry＋原图Mid Scale＋视频等权中位聚合。原61个正式源码/冻结数据文件SHA256保持一致，分割、选帧、Ring、P/Q/X、短弧、Geometry、Gate、Q恢复、Scale和Fusion没有修改。GT沿用人工确认映射，没有重新审计GT、恢复旧映射或按误差选例。证据见[`numerical_freeze_unchanged.json`](../validation/evidence/public_cases_20261018/numerical_freeze_unchanged.json)。

## 数据公开与安全检查

所选数据是维护者明确允许本轮增量公开的实验室自制硅胶模型影像。27幅原图逐一通过图集检查，仅见模型内镜场景，无患者信息、人像或身份叠字，JPEG EXIF为空；mask为同帧单通道uint8原始类别文件。逐文件SHA256与冻结输入完全相同，未重压缩、缩放或改mask。完整视频、全部790帧与训练数据不随本轮上传。

输入身份来自`validation/frozen39/inputs.csv`，expected只按source_id截取`expected_frames.csv`及`expected_ring.json`。`case_metadata.csv`另存GT、Paris、帧号、视频时间戳、预测、质量及失败原因。预测manifest仅含正式输入字段；测量完成后才读取expected，GT不进入算法。

选择首先按机制和质量身份筛选，再取该筛选内时间顺序首帧，未使用误差或GT排序。完整Is组保留全部20帧。所选6组及各自分母见[scope.csv](../examples/reproduction_cases/scope.csv)，逐文件哈希、来源及授权范围见[案例目录](../examples/reproduction_cases/README.md)。

| 审计项目 | 本轮结果 |
| --- | --- |
| 目标组数 | 6个机制组；其中仅1个完整冻结组，其他5个为子集 |
| 本算法独立成功 | 6/6机制子集组有数值；完整冻结组1/1 |
| 本算法失败 | 1/27帧Ring失败；无失败子集聚合组。单独failure入口为0/1数值组、NaN |
| 历史测量值回退 | 0；expected仅用于计算后的比较 |
| 失败组及原因 | Ip 4.7 R5公开失败帧：`ring_recovery_unreliable`，内外中心一致性诊断未通过；[证据图](visualization/ring_failure.png) |
| 最终统计样本 | 数值帧26/27；完整组与机制子集分开报告，不据此估计39组性能 |
| 是否用GT选择算法 | 否；未改算法、参数或预测，GT映射原样冻结 |

## 独立计算与回归

从公开JPEG/mask独立计算27帧，得到26帧数值有效、1帧Ring硬失败、24帧Normal、2帧Q_conflict_recovery、24帧Strict。全部质量身份、拒绝原因、P/Q数量及选中形状与冻结结果一致。

相对冻结评价表：L_px最大差约8.53×10⁻¹⁴px，毫米直径最大差约3.55×10⁻¹⁵mm，视频/子集组中位最大差约1.78×10⁻¹⁵mm；Mid像素轴及mm/px差为0。Ring内、外、Mid参数另行逐一比较。新旧差异均为浮点舍入，非算法改变。

正式主线仍为：

\[
s_i=\frac{D_{\mathrm{inner}}+1.2}{L_{\mathrm{mid},i}},\quad
D_i=L_{\mathrm{polyp},i}s_i,
\qquad D_{video}=\operatorname{median}_i(D_i),\quad
D_{group}=\operatorname{median}_{video}(D_{video}).
\]

长度均为原图完整直径像素；R5/R10中心线物理直径分别6.2/11.2mm。不对息肉仿射拉伸，不分开取像素中位数与尺度中位数。Q恢复即使Strict=False仍按冻结规则进入正式聚合；Ring失败保留NaN。

完整Is组两视频各10帧，中位数分别10.274870mm、10.143904mm，组级等权中位数**10.209387177421648mm**，与原完整组一致。其他5组的子集预测不得描述为原完整组复现。

本轮没有重跑39组全部Geometry；公开27帧独立重算并回归，原39组冻结预测及GT文件哈希不变。既有790帧独立复算证据仍在迁移技术报告，不能把本轮样例回归冒称新增39组复算。

## 可视化真实性与实现

新增`src/dxy_mid/diagnostics.py`只读诊断API。它调用现有`measure_frame`，通过线程本地Python返回事件捕获本次`pipeline._measure`及短弧过滤的实际局部变量；不写数值模块，不重拟合，不改变求解器输入，不替换solver hook，并在退出时恢复原profiler。正常API不启用此观察器。

`scripts/render_public_process.py`用这些实际对象绘制7张PNG：两种典型主体六阶段图、Q恢复、P=0遮挡推断、Ring失败、小孔修复及多视频聚合。图中原图面板显示完整图像，其他面板只改变视窗。颜色、source_id及数值源统一，所有轴均来自实际chosen或Ring参数。

真实Ring内外候选点和RANSAC椭圆直接来自`RingResult.evidence/geometry`；没有把全部候选点叫作内点。Mid使用已有参数平均结果。P/Q/X来自原轮廓记录，Q保留contour_id/index。白圈是实际抽样过的Q，法向区间稀疏显示；全量约束与求解身份写入CSV。

候选图只显示8个实际非profile候选，完整A/B池、原拒绝理由和诊断profile身份均保留。没有构造不存在的“更好候选”。粉色测量端点用实际chosen的基矩阵SVD得到，端点距离必须与L_px一致；黄色轴长度必须与Mid尺度来源一致。

六帧诊断图共98项自动真实性检查通过：同帧尺寸、息肉mask不变、原轮廓顺序及完整分类、Q真实像素和索引、法向区间、短弧完全退出求解、chosen轴与记录一致、Mid尺度与同帧毫米公式，以及Ring失败无chosen/NaN。多帧冻结回归进一步检查数值与Normal/Strict/Q恢复身份。逐项记录及逐点CSV见[图集索引](visualization/README.md)。

### 图中保留的问题

- Q恢复案例原始冲突为`P_p90_residual_too_large`，仍按冻结Q距离恢复分支输出；没有将它改称Normal或Strict，也未为了展示改善其拟合。
- P=0案例虽通过原门控/稳定性，仍属于遮挡约束推断；窄区间稳定不等于真实隐藏边界已被验证。
- Ring失败案例填充7px小孔后仍不可靠，保留NaN；另一帧填充340px后可测，不能据此声称整组或整段视频全部恢复。

## 安装与复现命令

从全新克隆的本分支根目录：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[plot,test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/reproduce_public.py \
  --case all --output outputs/public --workers 2 --render
```

只复现完整组用`--case multivideo`，只验证失败用`--case failure`，快速机制集合用`--case mechanisms`。脚本输出独立重算的逐帧/视频/组表、逐帧和Ring参数回归报告及明确复现分母。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/render_public_process.py \
  --cases examples/reproduction_cases --run outputs/public --output outputs/public-process
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -v
```

增加公开数据完整性、普通Ip、Q恢复、P=0、真实Ring失败、图点真实性、正式文件哈希冻结及视频等权测试。完整多视频20帧通过上述独立命令验收，不要求每次CI全跑。CI只计算少量代表帧，另执行公开failure CLI；完整39组数据不可用时保持明确SKIP。

独立failure CLI验收发现旧评价脚本假定CSV必有候选轴列：当所有帧均硬失败时，测量器从未生成候选，因此合法地没有轴列。只修改`verify_regression.py`的评价规范化：无任何数值有效帧时允许缺轴列并按NaN比较；任何有效测量缺轴列仍报错。新增真实失败CLI自动测试。未修改测量器、预测表或失败身份。

本地最终完整20项测试：19通过、1项私有全量测试SKIP；随后新增的真实失败CLI测试单独通过，共20个不同测试通过，1项全量测试SKIP。最终自动化与命令验收记录在`validation/evidence/public_cases_20261018/`。远端检查统一执行21项，使用现有[DXY-MID workflow](https://github.com/zheng1524/new-polyp-dxy/actions/workflows/dxy-mid.yml)；本次具体提交的状态和运行链接记在[PR #1](https://github.com/zheng1524/new-polyp-dxy/pull/1)的Checks与说明中，不以历史CI冒充本次CI。

## 外部目前能复现什么

无需额外下载即可从所列原图和mask独立完成Ring、小孔修复、Boundary、Geometry、Gate、Stability、Mid尺度和聚合；可完整复现一个冻结多视频组，并真实重现失败/遮挡/Q恢复机制。不能从这些JPEG逆向宣称完成了MP4解码、选帧或神经分割。

未发布完整MP4、精确INIT权重、全790帧或训练数据。新增完整视频和权重均等待人工决定，真实大小、来源、用途及建议分发方案见[`PUBLIC_ASSETS_PENDING.json`](PUBLIC_ASSETS_PENDING.json)。其中两段公开组的原视频共18,889,999字节，权重52,246,340字节。可考虑经审查后作为独立Release/数据资产发布，当前没有虚构下载地址。

没有新算法、GT优化、历史回退或自动晋升main。完成增量推送及PR更新后停止，等待人工审查。
