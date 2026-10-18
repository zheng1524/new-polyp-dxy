# DXY-MID正式测量主线

唯一正式算法是V9.1 Geometry + Mid Scale + 视频稳健等权聚合。机器入口见[CURRENT_POINTER](CURRENT_POINTER.md)和[MAINLINE_MANIFEST](MAINLINE_MANIFEST.json)。

[39组Excel](tables/group_results.xlsx)含中文表头、冻结首行和误差降序；[CSV](tables/group_results.csv)、[视频表](tables/video_results.csv)、[总体指标](tables/metrics_summary.csv)来自同一套冻结数据。`signed_error=prediction-GT`，负值低估、正值高估。

```bash
python -m pip install ".[report]"
python mainline/scripts/build_mainline.py --check
```

不加`--check`可从仓库随附DXY-MID冻结数值表重建CSV/Excel；这是科研结果导出，不是从图像重新测量。独立图像测量使用根README中的`dxy-mid`命令；全39组独立复算另需未公开的原图/mask输入。

S5/G3/DG3仅供历史追溯。旧S5完整mainline目录保存在[归档](../legacy/S5_mainline_20261018/)，默认导出脚本不再读取其数值或执行其聚合。
