# 环境、模型与数据契约

## 运行环境

参考环境为Ubuntu、Python 3.10、NumPy 1.26.4、SciPy 1.15.3、OpenCV 4.10.0.84、scikit-image 0.25.2、pandas 2.2.3、Pillow 10.4.0。`pip install '.[plot,test]'`安装锁定的核心版本与绘图/测试依赖。测量和790帧mask复算不需要GPU或PyTorch。推荐4个进程、BLAS/OpenMP各1线程；单进程约150 MiB，运行时受候选拟合复杂度影响。

新视频分割另装 `pip install '.[video]'`。已核验历史栈为PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121、Transformers 4.46.3、safetensors 0.4.5。CPU可运行；CUDA版需匹配显卡驱动。PyTorch包的平台安装源应使用官方安装说明，不从本仓库下载不明二进制。

Ubuntu若缺少venv/OpenCV系统库，安装 `python3-venv libgl1 libglib2.0-0`。中文PNG可安装Noto CJK字体。未安装中文字体时测量值不受影响，图中文字可能缺字。

## Level 1：图像与mask

`frames.csv`必须包含：

|字段|约定|
|---|---|
|source_id|唯一、稳定的帧标识；冻结实验保留原值，用于随机种子|
|group|组标识，只用于聚合，不控制算法|
|video|唯一视频标识，只用于聚合；发布索引使用文件名，不推断文件路径|
|stem|稳定的帧种子标识；同一冻结输入不要任意重命名|
|ring_mm|参考环内径，只允许5或10；不是目标GT|
|image_path|JPEG/PNG图像，BGR读入，原图分辨率|
|mask_path|单通道uint8 PNG，0背景、1息肉、2参考环，与原图同尺寸|
|image_sha256 / mask_sha256|可选但强烈建议；声明后必须匹配，错误直接拒绝|
|frame_index / timestamp_sec / cohort|可选追溯字段|

相对图像路径默认相对于manifest，或由`--data-root`指定。额外CSV列不进入数值算法。所有原始mask保持不变；Ring小孔修复只作用于内存副本。

```
dataset/
  frames.csv
  images/<source_id>.jpg
  masks/<source_id>.png
```

全39组发布索引在 `validation/frozen39/inputs.csv`，只有输入身份、采集标识及哈希。`expected_frames.csv`是隔离的评价基准，测量入口不会读取它。原始视频、GT源模型、分割权重、mask、输入清单与测量输出是不同资产，不能相互替代。

## Level 2：视频及冻结INIT

实际模型是本项目INIT训练导出的UNet3Plus + EfficientNet-B0、inter_ch=64、out_channels=3。不是同名公开Kvasir二分类预训练模型。权重SHA256：

`981aa596e8906edc4457219581c606d4a5a22bfc6f7e4924d3d9a4b9aff64464`

`model_code/init_segmentation/`的12个架构文件与实际冻结模型目录一致。模型目录必须包含这些源码、`config.json`和`model.safetensors`。初始化不下载新的ImageNet权重，严格加载全部参数；不容忍少加载或替换二分类权重。

预处理为RGB→256×256 PIL双线性→float32除255，无均值方差标准化。推理logits搬回CPU后双线性恢复原分辨率（align_corners=False），最后argmax得到0/1/2。PNG使用PIL保存，保持冻结缓存的字节编码。

视频索引：`group,video,path,ring_mm`；`video`是唯一键，`path`才是真正路径，相对video索引所在目录解析。新视频默认使用已批准C_joint：首尾各去5%，5时间段、每段最多24均匀候选、每段最多2帧、间距≥0.1s。分割使用质量95缓存JPEG，质量评分使用原解码帧。没有候选则记录失败，不用旧结果补足。

复现固定790帧时使用`--frame-manifest validation/frozen39/inputs.csv`，直接重放冻结帧号；保留原source_id/stem种子并核验图像、mask哈希。基础38组C_joint 770帧与扩展组既有20帧是明确的不同输入选择来源，不在数值测量中按组切换算法。

## 当前分发状态

已提交既有单帧样例及`examples/reproduction_cases/`中的6组27帧真实JPEG/mask（约3.74 MiB）。其中Is 10.5 R5完整提供两视频全部20个冻结选帧；其余是明确标记的机制子集。可用`scripts/reproduce_public.py`独立重算，详见该目录README。

完整790帧、冻结权重与原视频的分发状态见 `DISTRIBUTION_REVIEW.md`，待决定资产的真实大小与方案见`PUBLIC_ASSETS_PENDING.json`。不提供虚构下载链接。持有原数据者可立即运行本地全39组复算；目前外部可直接完成公开Level 1案例，不能宣称整个数据集的两级外部复现。

GT映射在 `validation/frozen39/GT_mapping.csv`，只供单独评价。人工确认GT为可信基准；Ip2.9两组更正为Ip2.8，Is6.5更正为IIa6.5。原组ID不改，以维护冻结帧关联。没有权威R5/R10实体对应表，不宣称已确认14对独立实体。
