# 正式Python API

```python
from dxy_mid import MeasurementConfig, measure_frame, measure_files
from dxy_mid.batch import run_manifest

one = measure_files('frame.jpg', 'labels.png', inner_diameter_mm=5.0)
print(one.audit['D_frame_mm'])  # mm；硬失败NaN
# one.ring / one.candidate / one.evidence / one.variants 保留几何与诊断

frames, groups, videos = run_manifest('frames.csv', 'outputs/run', workers=4)
```

`measure_frame`接收BGR uint8图像和同尺寸uint8类别mask；必须显式传入`inner_diameter_mm`和稳定`source_id`，可传`stem`。API不会读取组名、GT或外部直径。`MeasurementConfig`只确认冻结V9.1/Mid配置；本次发布不提供算法调参入口。

## 六组件与调用关系

|组件|当前调用|输入 → 输出|坐标/单位|
|---|---|---|---|
|Ring|`_engine.holes` → `_engine.ring`|当前mask/图像/参考尺寸 → 处理后mask、RingResult、失败原因|原图坐标，椭圆半轴px|
|Boundary|`_engine.boundary` → `cross_ring` → `short_Q`|当前帧与Ring → BoundaryResult、索引化P/Q/X与法向区间|原图像素；Q带contour_id/index|
|Geometry|`_engine.baseline` → `configure` → `choose`|当前P/Q、A/B主体 → 当前圆/椭圆、多变体候选|GeometryCandidate完整像素直径|
|Gate/稳定性|`_engine.audit`及冻结求解器质量hook|当前候选/残差 → Gate及五变体稳定性标记|质量不改变物理尺度|
|Scale|`_engine.scale(frame,ring,'mid_major')`|同帧Ring → ScaleResult|原图完整长轴px；mm/px|
|Fusion|`_engine.fusion`|本次单帧毫米值 → 视频中位/组中位|mm，不接受旧方法补值|

`_engine`为经过溯源的内部实现，不承诺历史模块名为稳定公共API；对外推荐上面的`dxy_mid`和`dxy_mid.batch`。迁移保留函数边界与数据契约，未新建另一套几何算法。数值函数按原实际定义模块整理，消除星号导入、运行时读取实验源码及实验目录IO。函数级来源见 `SOURCE_PROVENANCE.csv`。

原拟合器用临时模块级hook保留冻结质量/优化行为。单进程API通过可重入锁保护，批处理使用进程隔离；不要在同一解释器手工修改内部hook。缓存只由本次路线产生，包含当前图像/mask、参数及源码指纹；读取自身缓存前验证键。不要从不可信来源放入pickle缓存。

`row.group/video`只在API之后加入输出并参与聚合；`source_id/stem`保留历史随机数标识，重命名会改变RANSAC随机序列，不能声称与冻结实验逐帧等价。不同输入源仍使用相同算法。

## 单张图像需要预测mask时

安装可选视频/模型依赖并获得冻结INIT后，可直接复用同一个模型适配器：

```python
import cv2
from dxy_mid.segmentation import INITSegmentation
from dxy_mid import measure_frame
model = INITSegmentation('models/INIT', device='cpu')
labels = model.predict('frame.jpg')
result = measure_frame(cv2.imread('frame.jpg'), labels,
                       inner_diameter_mm=5.0, source_id='my-stable-frame-id')
```

现成mask用户无需加载该模型。模型和完整原视频未公开时，此入口应明确依赖缺失，不提供替代模型或旧结果补数。
