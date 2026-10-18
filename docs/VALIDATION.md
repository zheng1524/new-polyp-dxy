# 当前算法与结果验证

正式入口为`dxy-mid`及`dxy_mid`。`src/dxy_mid/_engine/`是当前正式实现，模块名不等于可删除的旧算法；分割架构位于`model_code/init_segmentation/`。当前源码SHA256见[SOURCE_PROVENANCE.csv](SOURCE_PROVENANCE.csv)。

[最近一次干净安装与790帧回归](../validation/evidence/current_validation.json)确认逐帧、Ring、视频与组级结果保持一致；公开样例及图集检查通过。

## 冻结基准

- [790帧输入与哈希](../validation/frozen39/inputs.csv)，同帧原图/mask独立计算。
- [逐帧期望值](../validation/frozen39/expected_frames.csv)及[Ring几何](../validation/frozen39/expected_ring.json)只在完成计算后比较。
- [最终GT映射](../validation/frozen39/GT_mapping.csv)只用于评价。
- [39组CSV/Excel和79视频表](../mainline/README.md)来自同一套冻结数据。
- [完整790帧数值回归](../validation/evidence/final_mainline_20261018/full39/regression.json)、[Ring回归](../validation/evidence/final_mainline_20261018/full39/ring_regression.json)及[公开真实案例验证](../validation/evidence/public_cases_20261018/regression.json)保留逐项证据。

## 运行验证

根README提供安装、公开mask、完整双视频组、硬失败及全39组命令。CI从干净环境安装，运行Python API、CLI、真实图像/mask、失败身份、表格一致性、图集实际几何和链接测试。完整790帧回归需要额外图像/mask，不在轻量CI中虚报通过。

测试比较逐帧长度/尺度/Dxy、Ring参数与质量标记，再比较视频中位数及组级视频等权中位数。失败NaN、原因及Q恢复身份也必须一致。允许浮点容差，不允许调整冻结期望值掩盖算法变化。

|审计项目|正式结果|
|---|---|
|目标组数|39|
|本算法独立数值成功|39组；757/790帧|
|本算法失败|0个组；33帧Ring失败且Dxy为NaN|
|历史测量值回退|0|
|失败组及具体原因|无全组失败；[33帧原因记录](../validation/results/ring_failed_frames.csv)|
|最终统计样本|39/39组，79段视频中77段有效|
|是否使用GT选择算法|否；冻结预测后独立评价，属于回顾性研究|

中位绝对相对误差4.4113269936%，MAE 0.7092238117mm，p95 14.7262673154%，最大16.5656402776%。数值成功不等于测量正确。全部原始数据和权重的公开范围见[分发说明](DISTRIBUTION_REVIEW.md)。
