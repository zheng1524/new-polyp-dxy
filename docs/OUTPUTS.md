# 输出与质量标记

每次新运行使用独立目录：`frames.csv`、`videos.csv`、`groups.csv`、严格帧诊断表、`run.json`、自身哈希缓存及可选`png/`。

- `L_px`：实际用于测量的原图坐标完整长轴/圆直径，px；不是半径。
- `L_mid_px`：同帧Ring中心线椭圆完整长轴，px；中心线沿用内外椭圆参数平均及双角方向平均，无新增第三拟合器。
- `scale_mm_per_px`：`(ring_mm+1.2)/L_mid_px`。
- `D_mm` / `D_frame_mm`：同帧`L_px × scale_mm_per_px`。
- `axis_x1/y1/x2/y2`：真正参与测量的长轴端点，原图坐标，与PNG一致。
- `ring_valid` / `ring_reason`：Ring恢复结果。使用明确布尔值；不依赖历史全为空的`Ring_valid`别名，更不将NaN转换为True。
- `default_Gate_pass` / `normal_Gate_pass`：正常Gate通过，不包括Q冲突恢复。
- `Q_distance_recovery` / `Q_conflict_recovery`：当当前A/B候选均被正常质量检查拒绝时，冻结算法从当前有界候选中按Q锚点欧氏距离恢复；保留原拒绝原因，不标为严格稳定。
- `stability_review_pass` / `strict_stable`：原五变体综合稳定性；正式Fusion不要求此项通过。
- `quality_class`：`normal_Gate_pass`、`Q_conflict_recovery`、`numeric_only`、`ring_failure`或`hard_no_selected_numeric`，不能合并称同一种可靠测量。
- `measurement_eligible`：Ring有效、所选Geometry像素长度有限且为正；软约束推断与不稳定数值可以聚合，硬失败维持NaN。
- `P_zero_occlusion_inference`：零P，仅由遮挡弱约束推断，不能等同于直接观察边界。
- `P_count/Q_count/X_count`、采样/评分数量、P/Q残差、多解直径区间、容差/种子敏感性和`original_conflict_reasons`：保留逐帧诊断。
- `ring_filled_area_px`、处理后mask哈希、原图/mask哈希、源代码指纹：核验输入和算法。

视频表报告有效帧数、中位数、IQR及稳定/Q恢复构成；组表报告有效视频数、等权中位数及失败原因。严格稳定帧结果单独诊断，不替代主结果。单帧/单视频也允许数值输出，但不能视为充分重复性证据。

常见硬失败包括无参考环材料、可见内缘不足、内外中心不一致、厚度比异常、外缘不稳定、没有有限几何候选。模糊或遮挡信息不足时不填补数字。稳定性未通过可能来自直径多解、容差边界饱和、随机种子/Q池变化和A/B不一致，应逐项保留，不能等同于Ring失败。
