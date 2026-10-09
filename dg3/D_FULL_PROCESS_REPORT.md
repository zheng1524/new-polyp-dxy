# D：Quality-aware Sharp5 + frozen G3 全流程报告

## 固定范围

D 仅替换 Sharp5 在既有缓存候选中的选帧。视频首尾裁剪、五段分箱、INIT segmentation、ring fitting/identity、G3 amodal ellipse、ring scale、单帧 `Dxy_i`、至少4有效帧的 capture median 及 formal group fusion/fallback 均不变；没有新网络、GT 调参或重训。

## GT-blind 算法

每个原 Sharp5 bin 先按既有全图 `sharpness` 取前 **3** 张缓存候选。对冻结 mask 的 polyp interior、polyp 边缘邻域和 ring 边缘邻域分别计算 Tenengrad（Sobel 梯度能量均值）与 Laplacian 辅助值，背景不参与。候选须满足：ROI Tenengrad ≥同 bin 最大值的 **75%**、Laplacian ≥最大值的 **50%**、当前 V2 valid、ring 最大连通成分 ≥**0.80**、frontality ≥**0.35**、ring-adjacency 可信原始外缘 ≥**0.35**且至少一段可信 arc。通过者按可信外缘占比→可信弧长→ROI Tenengrad→原 sharpness 字典序排序。无候选即 missing，不补帧。

选帧后复用冻结 G3：只有 trusted 原始 contour 拟合；ring 是 unknown；G3 的切线、遮挡与可见背景软约束及多初值参数保持不变。D 的变化既可能是真实几何改善，也可能是 capture 不足4帧后不再替换 formal view 的 fallback；不能只凭椭圆外观归因。

## 覆盖与指标

- 395 bins：通过固定 quality gate 并产生一个选帧 **278**，质量 gate missing **117**。
- D eligible captures **40**；原 C/G3 为 **56**，这正是“门控偏严”的主要量化证据。
- coverage 39/39；median abs **4.981%**；MAE **0.631 mm**；mean abs **6.848%**；mean signed **0.622%**；mean error **-0.144 mm**；p90/p95/max **14.830/20.220/23.685%**。

相较 C，median 5.004%→4.981%，但 p95=20.220%，说明当前规则尚不能晋升主线。每个组的完整 D 证据均在 `visualizations/D_all_groups/`：行=capture，列=5个时间段；绿色=冻结G3，金色/紫色=ring/unknown；tile 显示 Dxy、ROI Tenengrad、trusted比例，MISSING 明确标出。`tables/D_all_groups_index.csv` 给出预测、误差、有效帧与 eligible capture 索引。
