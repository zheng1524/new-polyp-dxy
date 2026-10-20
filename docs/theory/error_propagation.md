# Mask变化如何传播到毫米直径

[返回理论入口](README.md) · [尺度推导](scale_recovery.md) · [实验与数据](experiment.md)

## 1. 有限扰动的精确分解

固定同一场景的物理参考直径，统一写作

$$
D_X=L\frac{D_{\mathrm{mid}}}{2a}h_X,
$$

其中 $h_A=1$，$h_B=\sqrt{a/b}$，$h_C=\lVert Te\rVert$，$h_D=2\sigma_{\max}(TB_{\mathrm{body}})/L$。

前后预测均为有限正值时，

$$
\frac{D'_X}{D_X}=\frac{L'}L\frac a{a'}\frac{h'_X}{h_X},\qquad
\Delta\log D_X=\Delta\log L-\Delta\log a+\Delta\log h_X.
$$

这是**有限变化的严格恒等式**，不是小扰动近似。NaN或数值硬失败不能代入比值，应单独报告。本实验始终保留这些状态，实际927条记录均有数值。

A和B分别满足

$$
\begin{aligned}
\Delta\log D_A&=\Delta\log L-\Delta\log a,\\
\Delta\log D_B&=\Delta\log L-\tfrac12\Delta\log a-\tfrac12\Delta\log b.
\end{aligned}
$$

## 2. C的局部敏感性

令 $\rho=a/b$，$\phi=\theta_p-\theta_r$，并定义

$$
g=\cos^2\phi+\rho^2\sin^2\phi,\quad
w=\frac{\rho^2\sin^2\phi}{g},\quad
u=\frac{(\rho^2-1)\sin\phi\cos\phi}{g}.
$$

在方向可局部连续表示时，

$$
\mathrm d\log D_C=\mathrm d\log L+(-1+w)\mathrm d\log a-w\mathrm d\log b
+\nu(\mathrm d\theta_p-\mathrm d\theta_r).
$$

这是**局部微分关系**；有限变化的线性近似会有余项。方向导数使用弧度，不能直接对CSV中的度数求导。Ring与chosen旋转通过相对角进入，不能重复计入一个额外尺度。

$w\in[0,1]$：原轴平行Ring长轴时依赖$a$，平行短轴时依赖$b$。角度敏感度在 $\tan|\phi|=1/\rho$ 处达到 $(\rho-1/\rho)/2$；0°和90°端点的一阶角度项为0。圆的方向约定和无向轴的角度折返应另作标记，不能解释成真实主体旋转。

## 3. D的整个形状敏感性

D由主体两个半轴、主体方向与Ring轴比共同决定，不能只研究原长轴方向。使用[尺度推导中的](scale_recovery.md) $\tau,\kappa,\lambda_+$，设 $s=\sqrt{\tau^2-4\kappa}>0$。对任意局部参数$z$，

$$
\frac{\partial\log\sigma_{\max}}{\partial z}
=\frac{\partial_z\tau+(\tau\partial_z\tau-2\partial_z\kappa)/s}{4\lambda_+}.
$$

取 $z=(\log c,\log d,\log\rho,\alpha)$，相应的 $\partial_z\tau$ 为

$$
\begin{pmatrix}
2c^2(\cos^2\alpha+\rho^2\sin^2\alpha)\\
2d^2(\sin^2\alpha+\rho^2\cos^2\alpha)\\
2\rho^2(c^2\sin^2\alpha+d^2\cos^2\alpha)\\
2(\rho^2-1)(c^2-d^2)\sin\alpha\cos\alpha
\end{pmatrix},
\qquad
\partial_z\kappa=\begin{pmatrix}2\kappa\\2\kappa\\2\kappa\\0\end{pmatrix}.
$$

再加 $-\mathrm d\log a$ 即得到 $\mathrm d\log D_D$；$D_{\mathrm{mid}}$固定。因$h_D$中的$L$与外部$L$相消，D最终取决于完整主体矩阵，而非单独的原轴长度。

若最大奇异值重复（$s=0$），不能强套唯一方向梯度；可使用

$$
|\sigma_{\max}(M+E)-\sigma_{\max}(M)|\le\lVert E\rVert_2
$$

的稳定界，并直接检查多方向。原生圆的半轴方向表示也不唯一，不能把坐标表示的变化当成真实几何变化。若扰动前后chosen均为圆，则有更简单的严格关系：$D_D=L D_{\mathrm{mid}}/(2b)$，所以 $\Delta\log D_D=\Delta\log L-\Delta\log b$。

## 4. 理想Mask同比模型与它的边界

只有在拟合输出确实满足 $L'=(1+\delta_p)L$、$a'=(1+\delta_r)a$ 时，才能写为

$$
\frac{D'_A}{D_A}=\frac{1+\delta_p}{1+\delta_r},\qquad
\frac{\Delta D_A}{D_A}\approx\delta_p-\delta_r.
$$

后一个式子才是一阶近似；将**输入Mask倍率**直接替代**拟合输出倍率**则额外需要前述假设。

| 条件 | 输入变化 | 理想A趋势 |
| --- | --- | --- |
| N0 | 原始Mask | 基准 |
| N1 | 仅polyp扩大 | 增大约 $\delta_p$ |
| N2 | 仅Ring扩大 | 减小约 $\delta_r$ |
| N3 | 两者分别独立扩大 | 由倍率比决定 |
| N4 | 两者同倍率扩大 | 理想抵消 |

实际V2扰动围绕两类**各自原始可见Mask质心**作最近邻缩放，Ring优先合并；Ring内孔与外缘一起放大，不是增加材料厚度。RGB、相机和三维GT不变。它不是整幅图像共同缩放：类别覆盖、局部接触、遮挡、P/Q分类、chosen参数和Gate/Q恢复都可能变化。

因此输入倍率相等，不保证正式Geometry输出等比例。九场景中7个场景近似抵消，两个Is14场景却显著变化；见[逐场景结果](experiment.md)。这也不证明真实分割误差必然服从5%–10%均匀放大。

## 5. 波动中的相关性

在同一条件的有限数值预测集合上，严格成立

$$
\operatorname{Var}(\log D_A)=\operatorname{Var}(\log L)+\operatorname{Var}(\log a)
-2\operatorname{Cov}(\log L,\log a).
$$

一般地令 $x=\Delta\log L$，$y=-\Delta\log a$，$z=\Delta\log h_X$，则

$$
\begin{aligned}
\operatorname{Var}(\log D_X)={}&\operatorname{Var}(x)+\operatorname{Var}(y)+\operatorname{Var}(z)\\
&+2\operatorname{Cov}(x,y)+2\operatorname{Cov}(x,z)+2\operatorname{Cov}(y,z).
\end{aligned}
$$

共同变化可能通过协方差降低比值波动；但共同输入倍率不保证拟合输出共同变化。即使N3输入倍率独立，遮挡合并和P/Q过程仍可能使输出相关。

公开[144组方差分解](data/log_variance_decomposition.csv)采用每场景/条件20次重复的样本协方差（ddof=1）。协方差项可以为负；这是代数分解，**不是独立因果贡献百分比**。不能用只保留数值成功后的方差掩盖失败频率。

## 6. 连续传播、跳变与两种评价

记录圆/椭圆、A/B主体身份切换，同时检查半轴、中心和方向的实际变化。相同标签不保证参数稳定，标签切换也不必然导致明显直径跳变。

本实验的参数跳变诊断固定为：长度或主体短半轴相对理想polyp倍率偏差>20%，或中心移动>0.15$L_0$，或前后均为椭圆且无向长轴改变>30°。它只是描述标签，不修改Gate、不剔除样本。候选切换、参数跳变、Gate改变可以重叠。

- **Accuracy**：相对固定三维GT的差异。
- **Sensitivity**：相对该场景该方法N0的变化。

稳定的高估仍可能不准确，偶然接近GT的跳变也未必稳定。20次重复的经验5%–95%分位不是临床置信区间，不能精确估计罕见尾部，更不能将720次Mask扰动视为720个独立息肉。
