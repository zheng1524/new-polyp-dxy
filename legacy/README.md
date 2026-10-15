# 历史研究归档身份

正式推荐入口只有 `dxy-mid` / `python -m dxy_mid`。S5、G3、DG3及历史Fusion均已退役，不是自动回退路径。

为保留历史引用和科研追踪，本次不移动原目录、不改写历史实验或结果：

| 历史内容 | 原位置 | 当前角色 |
|---|---|---|
| S5/CPAG与V2旧整套测量 | `src/dxy_s5cpag`, `base38`, `ip88_extension`, `mainline`, `scripts/run_raw_selector.py` | 仅归档，旧S5包含历史融合，不能称DXY-MID独立覆盖 |
| D-G3研究 | `src/dxy_dg3`, `dg3` | 未晋升的研究记录与既有授权素材 |
| 共享Ring/图像工具 | `src/polypseg` | 保留旧API；DXY-MID数值入口不依赖其实验IO |
| 原首页与原源码闭包审计 | `legacy/S5_README.md`, `SOURCE_CLOSURE_AUDIT.md` | 只描述历史S5版本 |
| 当前数值函数中的历史算法来源 | `src/dxy_mid/_engine`, `docs/SOURCE_PROVENANCE.csv` | 迁移的可维护源码子集；没有读取旧预测 |

V2 A/B构造、圆/椭圆数值拟合和Ring邻接辅助函数仍是DXY-MID自身实现的一部分；“历史方法退役”不等于删除这些经过核验的底层算法。没有调用历史G3 amodal拟合器、S5正式融合、R2/S0最终预测。
