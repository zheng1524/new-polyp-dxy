# DXY-MID主线迁移技术报告

## 范围及当前发布状态

本次只做工程迁移、独立重算及发布准备，不调整任何几何、P/Q、Ring、Gate、Scale或Fusion参数，不开展新算法搜索。目标仓库为`zheng1524/new-polyp-dxy`，默认分支main，迁移前提交`518c83f2695cbee4c73078aac194bc85027decbb`。实际研究工作区的origin指向另一个三维项目`polyp-resarch`；没有修改该仓库的三维代码或用户未提交工作。本次在独立干净Dxy checkout的`dxy-mid-migration`分支操作。

完整权重、790帧及原视频的公开分发来源/许可尚待明确，因此不把迁移分支标成全数据集两级外部可复现的正式Release。已有公开真实样例可独立运行；本地全量验证情况见下文最终回归证据。

## 正式算法及来源

新的唯一推荐流程为DXY-MID，Geometry=V9.1、Scale=Mid。实际研究代码使用多代实验目录导入和动态hook，V9.1入口对57帧重新求解、其余733帧引用V9结果。本次没有把这个增量脚本直接复制为正式入口，而是抽取其实际可达数值函数，建立普通包内相对导入、明确当前输入API与新批量入口。

|实际步骤|冻结实现来源|迁移位置与处理|
|---|---|---|
|分割|INIT第13轮检查点；历史`run_old_method_v1.run_segmentation`|`segmentation.py`，同架构、预处理、logits恢复及PIL PNG；严格核对权重SHA|
|视频候选与选帧|原始均匀候选及已批准`roi_sharpness_frame_selection_v2/roi_quality.py`|`candidates.py`、`selection.py`；不读旧筛选结果替代新视频选择，冻结实验则重放输入manifest|
|Ring小孔|V9 `ring_holes.run`|`_engine/ring_holes.py`；只修复可信封闭背景小孔|
|Ring拟合|U2 `ring_component`及V8.1 `existing_ring_check`|同名可达函数；可见内缘RANSAC/中心线推断/闭孔有限恢复；无按组分支|
|原图P|V8.4.7 `observable_boundary`|冻结3×3实际mask邻接定义；不修改P规则|
|跨环Q|V8.5实验B `cross_ring_boundary`|原始息肉轮廓索引锚点、法向区间，Ring仅提供邻接与厚度|
|短Q弧|V9.1 `q_arc_filter`|`max(3px,0.02×原轮廓周长)`；整弧删除；不改P|
|A/B及数值几何|原V2 A/B构造、圆/椭圆数值函数；V7-B方向互补|保留实际算法；移除运行时读取实验文件的AST前缀裁剪，用等价已展开函数承载|
|评分、Gate及恢复|V8.4.4 `q_evidence_policy`、`q_distance_selection`及原V7五变体审查|保留0.5锚点/区间混合评分、原冲突恢复、默认与严格标记，不擅自改成V8.5 C/D|
|Mid|V3已核验`scale.run(...,'mid_major')`|只调用原图标量分支；中心线仍来自内外椭圆参数平均|
|Fusion|V9 `aggregation_interface.aggregate`|同函数源码：同帧毫米→视频中位→视频等权组中位|

冻结数学定义：`s=(D_inner+1.2)/(2*a_mid)`，`D_frame=L_native*s`。R5中心线物理直径6.2mm，R10为11.2mm。不仿射拉伸息肉，不使用分开取中位后相乘。只要Ring有效且几何有限正值即可参与正式Fusion；Strict-only单独诊断。

Ring辅助函数为历史返回契约仍计算环平面诊断矩阵；Mid分支不使用它，也不将它作用于息肉。正式Scale矩阵为单位矩阵。

圆/椭圆用同一原图坐标，长轴是完整直径；PNG绘制同一候选的真实测量轴端点。零P明确记为遮挡推断；Q恢复保留原冲突原因，不冒充正常Gate或严格稳定。

冻结关键参数为16扇区、P充分扇区≥4点、空/稀疏P方向分别选2/1个Q、P抽样上限128、Q有效权重有P时0.35/零P时1.0、RANSAC 64次、每形状5个优化初值、max_nfev=70、椭圆轴比上限3.5。Q容差维持10%/20%/35%局部环宽、默认20%；五变体维持默认、10%、35%、seed+1、Q池上限24。种子仍由`SHA256(source_id + "|0.2")`前8位十六进制得到。没有进行阈值搜索。

## 模块结构与复用边界

正式包为`src/dxy_mid/`：`pipeline.py`高层API、`batch.py`内容哈希批处理、`segmentation.py`/`video.py`视频与模型入口、`selection.py`冻结选帧、`visualize.py`简洁PNG、`_engine/`实际可达数值闭包。配置在`configs/dxy_mid.json`。所有输入路径来自CLI/manifest，图像/mask路径可相对解析；不硬编码研究者家目录。

`tools/migration/extract_verified.py`仅是维护者迁移工具，需要显式指定原研究工作区，不是运行依赖。`SOURCE_PROVENANCE.csv`记录原始文件SHA、实际函数体及迁移函数体SHA、是否展开动态前缀/优化hook。展开只改变运行时IO/导入机制及溯源常量，不改变数值表达式；回归专门验证这一点。

历史S5/G3/DG3文件不搬迁不删除，以保留引用与研究记录；原首页保存为`legacy/S5_README.md`，各历史首页标明退役。它们不在正式入口调用图中。保留V2拟合源码和历史Ring邻接辅助函数是算法源码复用，不是历史测量回退。没有调用G3 amodal拟合、S5 formal Fusion、旧照片直径或S0/V2保存预测。

## 输入与缓存审计

冻结输入为38基础组C_joint 770帧，加第39组已冻结20帧，共790帧。正式回归从当前JPEG与0/1/2 mask重新执行Ring及全部Geometry；不读取V9/V9.1几何pickle作为预测来源。发布输入清单将原视频绝对路径标签一对一映射为文件名，原source_id/stem不变，保持随机序列。重复视频键/重复source_id/哈希不一致直接拒绝。

输入清单仅投影身份、环尺寸、当前图像/mask与哈希字段；多余GT、旧预测列不进入算法。`expected_frames.csv`只在计算结束后的独立比较读取；GT仅在`evaluate39.py`读取。本路线的自有缓存按当前图像、mask、环尺寸、种子身份、版本与源码指纹建立，其他路线缓存不能由manifest传入。

`validation/evidence/pure_frame_call_trace.json`提供实际函数调用与IO审计：纯单帧计算拒绝额外CSV/JSON/pickle读取，测量输入为图像/mask本身。研究原冻结源文件保持原样。

## 独立安装与复现验收

采用新建venv（不共享site-packages），安装锁定依赖，CPU mask层未安装Torch；完整观察到的依赖清单在`environment.cpu.lock.txt`。从仓库外工作目录调用已安装CLI，重算已公开真实图像/mask，与冻结样例逐帧数值及质量一致。默认自动测试包括真实样例；完整数据测试在未配置数据时明确SKIP。原有归档测试继续保留。

本地Level 2从真实原视频按冻结帧号解码，JPEG质量95，使用SHA核验的INIT权重重新分割。发现最初OpenCV与PIL的PNG编码字节不同而类别像素一致，已恢复历史PIL编码，图像/mask SHA与冻结值完全一致，最终Dxy及质量回归通过。另外对一段真实原视频执行完整C_joint：重新解码120个候选、预测120张mask、按冻结质量规则选出10帧；所选帧号及图像/mask哈希与原冻结C_joint完全一致（`raw_video_C_joint_test.json`）。该测试使用本地受控依赖，不等于已解决对外权重/视频获取。

可复现命令见README与`DATA_MODEL.md`，API及六组件契约见`API.md`。完整790帧回归结果、自动测试及环境实际证据在验收完成后记录于本报告的“最终验证”部分。

## 迁移回归发现并修正的接口问题

第一套790帧重算的所有Ring内/中/外椭圆参数、逐帧像素长度、毫米值、测量轴、P/Q数量、候选形状、视频/组中位及质量布尔值均一致。计数为757数值、33 Ring失败、627正常Gate、130 Q恢复、616 Strict。但比较器发现33个Ring失败帧的`stability_flags`为空，而原冻结为`ring_failure`。

原因是高层API的初始审计字典漏写该失败标签；`ring_valid=False`、`reason=ring_failure`、NaN及所有测量数值原本均正确。修复统一初始化字段为`ring_failure`，有效Ring帧仍由原Gate审计覆盖此字段。没有按组修改、降低阈值或复制冻结预测。新增无Ring单元测试断言该标签。初次失败审计保留在`validation/evidence/research_environment/`，初次干净环境未完成的运行也保留为工程诊断，未标为通过；最终在新目录`clean_full39_final`使用修复后的干净安装从头执行790帧，不复用该批缓存。

另外，长视频分割后直接fork进入CPU拟合出现原生线程锁等待。视频入口改为用新Python解释器调用已有CPU CLI，隔离CUDA/OpenCV状态；没有改变任何测量函数或参数。模型/视频阶段和数值测量阶段仍通过同一当前输入manifest对接。

## GT与误差口径

预测回归以原冻结每帧Dxy为最高比较基准。人工已更正两组Ip2.9→Ip2.8、Is6.5→IIa6.5，GT为可信实测基准；没有用文件名名义尺寸替代缺失GT。预测不变时，更正GT后的中位绝对相对误差4.4113%、MAE0.709224mm；更正前旧报告为4.1915%、0.7041mm。不能改回错误映射来追求旧指标。39组是回顾性硅胶模型数据，不能声称临床泛化。

## 尚存限制

本地79段原视频均存在，但完整输入、冻结权重及原视频尚未建立已确认的公开分发方式；已核查目标GitHub Release资产清单为空（`github_release_inventory.json`），详见`DISTRIBUTION_REVIEW.md`；没有虚构下载地址、拿另一个权重替代或用预测CSV冒充独立复现。零P推断与Q恢复的几何正确性不因数值稳定而得到证明；深度差、透视、模糊、Ring遮挡和单视频证据不足保留原有局限。算法层未自动修复这些问题。

## 固定审计表

|审计项目|汇报|
|---|---|
|目标组数|39（基础38+明确扩展1），790帧|
|本算法独立成功|39/39组、757/790帧；干净安装完整独立重算通过|
|本算法失败|33帧NaN、无组级NaN；逐帧Ring原因完整保留|
|历史测量值回退|0|
|失败组及具体原因|五组含失败帧；见下文及Ring_failed_frames.csv，不将其称整组失败|
|最终统计样本|全部39个独立数值组，同时报告790输入帧与33硬失败|
|是否使用GT选择算法|否；数学配置冻结不变，GT只用于单独评价；没有新参数搜索|

## 最终验证（2026-10-15）

最终从仓库外工作目录调用干净venv内已安装的CLI，输入仅为当前790张图像及其原始类别mask，没有复用之前的Geometry缓存。`validation/evidence/clean_environment/prediction_freeze.json`记录计算完成后的输出SHA；其后才运行冻结比较及独立GT评价。

|检查|结果|
|---|---|
|逐帧身份与输入映射|790/790一致|
|逐帧像素长度/尺度/毫米值/实际测量轴|PASS；毫米最大差`7.11e-15 mm`，仅CSV浮点序列化级差异|
|Ring内缘/外缘/中心线及失败状态|790/790一致；参数最大差0|
|视频中位数|77段有效视频一致，最大差`3.55e-15 mm`；总输入79段|
|组级视频等权中位数|39/39一致，最大差`3.55e-15 mm`|
|Normal / Q恢复 / Strict及失败原因|逐帧完全一致|
|目标/独立数值覆盖|39/39组；757/790帧|
|质量构成|627 Normal-Gate、130 Q恢复、616 Strict；33 Ring失败|
|基础38组|770帧，737数值、33 Ring失败、612 Normal、125 Q恢复、601 Strict|
|第39组|20帧均数值有效、15 Normal、5 Q恢复、15 Strict；输入来源单独标识|
|源码/模型/原输入保护|原69份冻结源码SHA未变；权重与12份架构源码SHA核验通过|
|历史测量回退 / GT算法选择|0 / 无|

详细证据：

- [完整逐帧/视频/组级比较](../validation/evidence/clean_environment/regression.json)、[Ring参数比较](../validation/evidence/clean_environment/ring_regression.json)。
- [790帧独立重算紧凑表](../validation/evidence/clean_environment/recomputed_frames.csv)、[39组质量统计](../validation/evidence/clean_environment/group_quality_39.csv)、[逐帧数值差](../validation/evidence/clean_environment/frame_regression.csv)。
- [全部Ring诊断](../validation/evidence/clean_environment/Ring_audit_790.csv)、[33个失败帧的实际门控指标及尝试](../validation/evidence/clean_environment/Ring_failed_frames.csv)。
- [CPU干净安装](../validation/evidence/clean_install.json)、[公开样例回归](../validation/evidence/clean_demo_regression.json)、[真实视频冻结帧回归](../validation/evidence/raw_frozen_frame_regression.json)、[完整C_joint视频小样本](../validation/evidence/raw_video_C_joint_test.json)。
- [发布与完整重算源码的AST逐文件等价核验](../validation/evidence/release_vs_tested_numerical_source.json)。发布前只规范化生成源码末尾空行；44个数值实现文件AST一致，没有改公式或参数。

干净安装的自动测试为14项：13 PASS，1项完整私有数据测试按环境配置默认SKIP；该完整测试对应的790帧计算已通过上述独立运行完成，不将SKIP冒充PASS。最终安装/测试日志摘要另有证据文件。

可信GT更正后的固定39组指标：中位绝对相对误差 **4.411327%**、MAE **0.709224 mm**、p95 **14.726267%**、最大 **16.565640%**；10%以内 **30/39**。仅在这39个独立数值组上评价，同时保留完整39组/790帧分母。未改预测，未按GT挑结果。

五组含有Ring失败帧，但均非整组失败：

|组|失败帧|实际原因|
|---|---:|---|
|Ip10.3 R5|10|内外环厚度与1.2mm线材不一致|
|Ip4.0 R5|2|内外环中心不一致|
|Ip4.7 R5|17|内外环中心不一致；仅剩3帧数值证据|
|Ip8.8 R5|3|内外环厚度与1.2mm线材不一致|
|Is8.8 R10|1|内外环中心不一致|

Ip4.7 R5、Ip10.3 R5各仅一段有效视频，均明确标为重复性证据不足；没有用历史照片或S5补齐另一段视频。所有33个失败值保留NaN。

本地完整结果位于 `outputs/dxy_mid_release/clean_full39_final/`，其中`png/`有39张组级图，`frames.csv`为完整质量逐帧表，`videos.csv/groups.csv`为正式聚合结果。完整原图派生图集没有在分发权限未明时上传；Git仅提交原已公开样例的演示PNG，以及可审查的数值/哈希证据。

## GitHub交付与停止点

迁移分支为`dxy-mid-migration`，以审查PR交付，不重写main，不force push。默认README的DXY-MID替换已在该分支完成。由于完整帧、权重和原视频公开分发尚未确认，main暂不合并，也不创建宣称外部两级完整可复现的正式Release。最终提交SHA及PR链接随人工交付提供。

最简单的三条命令（先切到迁移分支仓库根目录）：

```bash
python3 -m venv .venv && .venv/bin/python -m pip install '.[plot,test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/dxy-mid measure --manifest examples/mask_demo/frames.csv --output outputs/demo --workers 1 --render
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/reproduce39.py --data-root datasets/frozen39 --output outputs/regression39 --workers 4
```

第三条需要完整790帧图像/mask包；现阶段没有确认的公开下载地址。L2另需冻结INIT权重和对应原视频。数据持有者可立即按契约运行，其他研究者当前只能完整运行已公开样例。这是分发阻碍，不能用已有结果CSV代替。
