# D-G3 相对 S5 的恶化案例：只读可视化（已提交子集）

所有图仅渲染已冻结的 S5 CPAG-V2 与 D-G3 G3 结果；没有重跑选择、分割、ring、V2 或 G3。每个 capture 两行各五个时间 bin：上行是 S5，青色线是其原 V2 选中的 circle/ellipse；下行是 D-G3，绿色线是其 G3 ellipse。

## 最大六个恶化组

|group|S5 abs %|D-G3 abs %|D-G3 − S5 (pp)|本包|
|---|---:|---:|---:|---|
|GP_Is_14.0mm_closed_R10_single_refY|0.224|13.941|+13.717|[PNG](GP_Is_14.0mm_closed_R10_single_refY.png)|
|GP_IIa_14.0mm_closed_R10_single_refY|0.487|12.483|+11.996|[PNG](GP_IIa_14.0mm_closed_R10_single_refY.png)|
|GP_Ip_4.0mm_closed_R5_single_refY|11.780|20.094|+8.314|[PNG](GP_Ip_4.0mm_closed_R5_single_refY.png)|
|GP_Ip_2.9mm_closed_R5_single_refY|14.522|18.386|+3.863|[PNG](GP_Ip_2.9mm_closed_R5_single_refY.png)|

这里只保留 5 组代表图；完整配对数值见 `tables_paired_degradations.csv`。所有图仅用于获授权的内部汇报与方法审查。
