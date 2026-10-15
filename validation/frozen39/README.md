# 冻结39组回归证据

- `inputs.csv`：790帧输入身份、相对图像/mask路径与SHA；38基础组770帧、扩展组20帧。
- `expected_frames.csv`：原V9.1逐帧评价基准。只供计算完成后比较，不能作为测量缓存。
- `PROVENANCE.json`：原冻结表和冻结说明的SHA256，计数基准及元数据键的可移植转换。
- `GT_mapping.csv`：仅供独立评价，人工确认的GT及标识更正。
- `videos.example.csv`：79段原视频索引格式；视频本体不在Git中。

重算入口：`python scripts/reproduce39.py --data-root <解包后的图像/mask目录> --output outputs/regression39 --workers 4`。

当前提交中的回归实际结论及安装证据由 `docs/RELEASE_TECHNICAL_REPORT.md` 记录。完整数据、权重尚未公开时，不应将缺少输入的全量测试标为PASS；CI只可运行真实公开样例。

视频键由原绝对路径一对一投影为文件名，聚合分组关系不变；source_id、stem和随机种子未改。检测到重名视频会拒绝，不凭重复文件名错配输入。
