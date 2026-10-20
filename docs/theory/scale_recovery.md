# 参考环尺度恢复：标量、面积与仿射

[返回理论入口](README.md) · [误差传播](error_propagation.md) · [固定九场景](experiment.md)

本页给出四种**分析公式**。正式CLI/API仍只使用A（Mid-Major）；B/C/D不会成为可选正式测量分支。四者共享同一帧、同一冻结chosen主体，不重新拟合轮廓。

## 1. 符号与坐标

| 符号 | 定义与单位 |
| --- | --- |
| $D_{\mathrm{mid}}$ | Ring中心线物理直径；R5为6.2 mm，R10为11.2 mm，圆截面线材直径1.2 mm |
| $a\ge b>0$ | 原图Mid椭圆长、短**半轴**，px；完整长轴为 $2a$ |
| $\rho=a/b\ge1$ | Ring轴比的倒数；表中的 $q=b/a=1/\rho$ |
| $L=2c$ | 当前chosen的原图最大像素直径，px；沿用冻结值与真实端点方向 |
| $c\ge d>0$ | 当前chosen圆/椭圆的长、短半轴，px |
| $\theta_r,\theta_p$ | Ring长轴、chosen实际测量轴在原图的方向；无向轴按 $\pi$ 周期处理 |
| $\phi=\theta_p-\theta_r$ | 两轴相对方向；推导使用弧度，CSV展示角度 |
| $B_{\mathrm{body}}$ | 原图chosen半轴矩阵，列向量将单位圆映为主体椭圆；$\sigma_{\max}(B_{\mathrm{body}})=c$ |
| $T$ | Ring平面归一化仿射矩阵；作用于图像坐标列向量 |

原图横向为$x$，纵向为$y$（向下）。设 $R(\theta)$ 为相应二维正交旋转，则可取

$$
T=\operatorname{diag}(1,\rho)R(\theta_r)^\mathsf{T}.
$$

Ring半轴矩阵为 $R(\theta_r)\operatorname{diag}(a,b)$，因此

$$
T R(\theta_r)\operatorname{diag}(a,b)=aI.
$$

矫正后的Ring是半径$a$的圆，完整像素直径仍为 $2a$。长度换算尺度为 $D_{\mathrm{mid}}/(2a)$，**不能再乘一次仿射尺度**。官方rectifier可在左侧多一个正交旋转；向量范数与奇异值不变，故以下长度等价。平移只影响中心，不影响直径。

## 2. A：Mid-Major（唯一正式尺度）

$$
D_A=L\frac{D_{\mathrm{mid}}}{2a}.
$$

A用Ring完整长轴建立单一mm/px标量，不拉伸息肉。物理解释依赖目标与参考环近似共享放大率。正交/弱透视下，理想平面圆的长轴不受倾斜压缩，因而可用来估计该放大率；任意透视、不同深度和畸变下则未必成立。A不是无条件正确的三维尺度恢复器。

## 3. B：Mid-Area

$$
D_B=D_A\sqrt{\rho}=L\frac{D_{\mathrm{mid}}}{2\sqrt{ab}}.
$$

Mid投影椭圆面积为 $\pi ab$，面积等效圆的半径为 $\sqrt{ab}$。另一方面，$\det T=\rho$；各向同性变换 $\sqrt{\rho}I$ 与 $T$ 具有相同的面积倍率。因此B把面积等效线性倍率施加于原chosen长度。

这是**面积等效的标量近似**，不是任意方向长度的精确逆投影，也不是把息肉本身的面积等效直径 $2\sqrt{cd}$ 当成测量目标。它仍然使用原来的最大长度$L$。

## 4. C：Chosen-Axis Affine

令 $e=(\cos\theta_p,\sin\theta_p)^\mathsf{T}$ 为冻结端点给出的单位方向，则

$$
D_C=D_A\lVert Te\rVert
=D_A\sqrt{\cos^2\phi+\rho^2\sin^2\phi}.
$$

C只变换**已经选定的轴**。沿Ring长轴时 $D_C=D_A$；沿短轴时 $D_C/D_A=\rho$。不使用mask PCA方向替代chosen方向，也不从端点舍入长度重新定义$L$。

若chosen为圆，原图没有唯一的最大轴方向；C仍使用API实际保存的方向。这种方向约定会影响C，但不能被解释为圆的真实生长方向。

## 5. D：Whole-Shape Affine

$$
B_{\mathrm{new}}=TB_{\mathrm{body}},\qquad
D_D=2\sigma_{\max}(B_{\mathrm{new}})\frac{D_{\mathrm{mid}}}{2a}.
$$

D变换完整chosen形状，再寻找最长直径，不重新拟合原始息肉轮廓。与C不同，D允许最大轴方向改变。它具有更完整的**二维椭圆仿射表达能力**，但这种表达能力不自动带来对三维主体更准确的物理测量。

### 可独立核验的解析式

主体最大轴相对Ring长轴的方向为$\alpha$；当前真实最大轴满足$\alpha=\phi$（圆的方向任意）。令

$$
\begin{aligned}
\tau&=c^2(\cos^2\alpha+\rho^2\sin^2\alpha)
+d^2(\sin^2\alpha+\rho^2\cos^2\alpha),\\
\kappa&=\rho^2c^2d^2,\\
\lambda_+&=\frac{\tau+\sqrt{\tau^2-4\kappa}}{2}.
\end{aligned}
$$

则 $\sigma_{\max}(TB_{\mathrm{body}})=\sqrt{\lambda_+}$，与SVD结果等价。这来自 $(TB_{\mathrm{body}})^\mathsf{T}TB_{\mathrm{body}}$ 的迹与行列式。

## 6. 严格大小关系与换轴

因为$T$的最小、最大奇异值为1、$\rho$，故 $1\le\lVert Te\rVert\le\rho$。存在单位向量$v$使 $B_{\mathrm{body}}v=ce$，因此

$$
\sigma_{\max}(TB_{\mathrm{body}})\ge\lVert TB_{\mathrm{body}}v\rVert=c\lVert Te\rVert.
$$

由此得到严格代数关系：

$$
D_D\ge D_C\ge D_A,\qquad D_B\ge D_A.
$$

同时 $D_C,D_D\le\rho D_A$。**B与C/D没有普遍排序**：不能把B固定理解成A与D的中间值。$a=b$时，四者完全一致。

对于圆形chosen，$B_{\mathrm{body}}=cU$，$U$正交，所以

$$
\frac{D_D}{D_A}=\rho=\frac ab.
$$

对于长轴与Ring长轴平行的椭圆，变换后半轴为$c$和$\rho d$，因此

$$
\frac{D_D}{D_A}=\max\!\left(1,\rho\frac dc\right),\qquad D_C=D_A.
$$

原长轴没有被拉伸，但当 $d/c>b/a$ 时，原短轴会成为新的最大轴。Is14场景中C接近A而D明显增大，正对应这种机制；具体数值见[实验说明](experiment.md)。

## 7. 二维恢复与三维物理假设

以下是**数学示意（Illustrative Geometry）**，不是新增Blender场景或新实验样本。

**共面薄片。** 在理想正交投影、近似等距像素下，Ring与同一平面薄片共享一个映射 $mR\operatorname{diag}(1,\cos\iota)V^\mathsf{T}$。理想圆环投影满足 $b/a=|\cos\iota|$；$T$消除平面压缩，D可恢复该平面内形状的最大径。C恢复原选定线段的长度，未必等于整个平面形状的最大径；B只提供面积等效倍率。一般透视下，椭圆轴比不能直接当成真实倾角。

**三维球体。** 正交投影下，球体轮廓始终为圆，而倾斜Ring为椭圆。共享放大率$m$时，A已能给出球直径；D仍会将圆放大$\rho$倍。这是把平面逆变换施加于三维轮廓造成的过度矫正。三维椭球及真实隆起主体更复杂，不能据球的示意断言D对所有三维主体都不合理。

当Ring与主体不共深度时，透视放大率也不同；A同样可能偏差。仿射变换不能从一幅轮廓恢复未知深度。B/C/D相对A均只增大或保持预测：可能抵消低估，也可能过度补偿或加剧高估。接近GT不等于其物理假设已被验证。

## 8. 可检查的证据层级

| 类型 | 本文结论 |
| --- | --- |
| 严格恒等式 | 四种倍率、SVD解析式、大小关系、圆形与平行轴特例 |
| 几何假设下成立 | 正交圆投影轴比、近似共享放大率、共面平面恢复 |
| 本实验观察 | 九场景的误差排序、Is14换轴及Mask扰动现象 |
| 不能据此证明 | A在全部真实息肉最准确；D对全部三维主体不合理；九场景代表真实39组 |

数值检查：`python docs/theory/verify_math.py`。脚本只使用NumPy、公开公式和CSV，不调用正式Geometry。
