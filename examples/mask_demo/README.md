# 已有公开真实图像的Level 1示例

图像复用仓库历史授权素材中的 `GP_IIa_22.0mm_closed_R5_single_refY_005-raw05-07.jpg`，SHA与V9.1冻结帧一致。mask为同帧原始分割类别：0背景、1息肉、2参考环。

```bash
dxy-mid measure --manifest examples/mask_demo/frames.csv --output outputs/demo --workers 1 --render
python scripts/verify_regression.py --actual outputs/demo/frames.csv --expected examples/mask_demo/expected.csv --output outputs/demo/regression.json
```

运行重新计算Ring、P/Q、Geometry及Mid，`expected.csv`只在第二条评价命令读取，不作为测量输入。该单帧示例的组中位数仅等于自身直径，不冒充完整组级结果。图像数据的原有使用/分发限制继续适用；不授予完整数据集许可。
