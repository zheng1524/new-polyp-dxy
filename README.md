# S5-CPAG-39

冻结的 Dxy 视频测量研究主线：**Sharp5 + Capture-level Post-area Guard，39 组范围**。

最终指标：coverage 39/39，median absolute error 4.577%，MAE 0.613 mm，p95 19.263%，max 23.685%。

## 内容

- `mainline/`：39 组主线定义、按误差排序的 CSV/图和生成脚本。
- `base38/`：原 38 组 S5-CPAG 的冻结规则、校验脚本和盲输出。
- `ip88_extension/`：此前缺失的 Ip8.8 R10 从原始视频重采集并以中心线内环补全接入的审计、脚本、选帧证据与可视化。

## 重要范围说明

其余 38 组遵循原 S5-CPAG 流程。Ip8.8 R10 是唯一的独立视频扩展：旧 V2 R3 未使用；环带因息肉紧贴内缘而没有闭合孔时，仅以当前 fitter 已有的 outer ellipse 和 skeleton-midline ellipse 按中心线对称恢复 inner ellipse，再使用未修改 V2、S0-p2 post-capture area guard 和 B1。

本仓库有意不包含原始内镜视频、500 帧缓存、模型权重或患者数据。脚本保留原项目绝对路径假设；在完整项目环境中执行即可复现，所需输入 SHA 已写入各实验的 manifest。

## 复现顺序

1. 在完整 `polyp_research` 工作区运行 `base38/scripts/verify_mainline.py`。
2. 按 `ip88_extension/RAW_VIDEO_REACQUISITION_REPORT.md` 执行原始视频重采集流程。
3. 运行 `mainline/scripts/build_mainline.py` 生成 39 组清单和误差排序。

详见 [`mainline/ERROR_RANKING.md`](mainline/ERROR_RANKING.md)。
