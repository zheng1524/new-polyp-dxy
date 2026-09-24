# 当前 Dxy 视频测量研究主线

当前冻结主线为：

`S5-CPAG-39`（Sharp5 + Capture-level Post-area Guard，39组范围）

入口与不可变定义：

- [39组机器可读 manifest](dxy_mainline_s5_cpag39_20260923/MAINLINE_MANIFEST.json)
- [39组误差排序与图](dxy_mainline_s5_cpag39_20260923/ERROR_RANKING.md)
- [39组排序 CSV](dxy_mainline_s5_cpag39_20260923/tables/group_errors_descending.csv)
- [基础38组规则与验证](dxy_post_capture_relaxed_area_guard_20260922/DXY_VIDEO_MAINLINE_V1.md)

38 个基础组规则：先按冻结 raw `result_sharp5` 选择每个bin的最高 sharpness 帧；仅当一个capture的原始已选帧中位面积支持落入正式S0 p2极低尾时，才对该capture内自身低支持帧做同bin、同候选池的宽松重选。V2、scale、B1和正式融合不变。

Ip8 R10 为此前唯一未覆盖组：它以两段原始视频重采集、冻结 INIT 和 outer+mid 中心线内环补全获得独立视频预测；旧 V2 R3 未使用。该扩展的完整定义和 SHA 位于 39组 manifest。

该研究主线不覆盖历史 `result/` 包；二者范围不同。
