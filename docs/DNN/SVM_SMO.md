# SMO 算法的推导

本文为《医学人工智能与机器学习》课程的作业之一。

笔者不想为这篇文章单独开一个“机器学习”的博客分类了，暂且还是扔在这个“深度神经网络”板块吧，虽然它和神经网络关系不大……不过多储备一些算法知识，没有任何坏处。

## SVM

这里速通一下 SVM 的知识，权当做 notation。SVM 的思想是找一个超平面能够让数据分得“最开”，我们使用几何间隔对这个分开程度进行量化的度量。

在最简单的线性场景下，我们可以使用

$$
w^\top x+b=0
$$

来描述一个超平面，由于这个式子是一个点积的形式，$w$ 就是这个超平面的法向量。我们可以依据数据在超平面的上方或者下方来进行分类，也就是 $f(x_i)=\mathrm{sign}(w^\top x_i+b)$。这个就是我们的模型，需要优化的参数就是 $w$ 和 $b$。

接下来定义损失，刚刚提到需要用几何间隔来量化超平面分割得好不好。几何间隔其实就是高维场景的点到直线距离公式，然后乘以正负类的符号：

$$
\gamma_i=y_i\dfrac{w^\top x_i+b}{\|w\|}
$$

我们要最大化几何间隔，其实就是最大化几何间隔的**下界**，因为优化上界或者其他东西一点用都没有。这个下界就是离我们的超平面最近的正样本点 $x_s$，我们叫它**支持向量**。考虑到几何间隔对线性缩放的 $w$ 和 $b$ 不变，我们就可以固定 $|w^\top x_s+b|=1$，这样其实最大化的就是 $\dfrac{1}{\|w\|}$，取一个倒数再平方，我们就可以转换成一个常见的优化问题了：

$$
\mathrm{arg}\max_{w,b\quad} \dfrac 12 \|w\|^2 \mathrm{\quad s.t.\quad}y_i(w^\top x_i+b)\ge 1
$$

但是，数据不可能完全能够满足线性可分，在决策边界处可能会存在噪音，因此我们需要引入一个容差来滤除噪音，也就是从刚刚的**硬间隔** SVM 变成**软间隔** SVM。引入间隔带宽 $\xi$ 为一个向量，条件改成 $y_i(w^\top x_i+b)\ge 1-\xi_i$，同时我们希望这种容差尽可能小（也就是精确度尽可能高），因此把它也引进损失里面，得到：

$$
\mathrm{arg}\max_{w,b,\xi\quad} \dfrac 12 \|w\|^2+C\sum \xi_i \mathrm{\quad s.t.\quad}y_i(w^\top x_i+b)\ge 1-\xi_i,\xi_i\ge 0\tag 1
$$

下面就是对其进行求解了。

## 拉格朗日乘数法

我习惯于深度学习那一套，一个 `optim.Adam` 就能解决。不过在 SVM 的年代，连 SGD 尚且是襁褓中的孩子。于是我们只能试图通过拉格朗日乘数法解决。这个约束优化问题。$(1)$ 式有两组约束，因此引入对应的两组参数 $\alpha_i$ 和 $\mu_i$，写出拉格朗日乘子：

$$
\mathcal{L}=\dfrac 12 \|w\|^2+C\sum \xi_i-\sum\alpha_i[y_i(w^\top x_i+b)-1+\xi_i]-\sum\mu_i\xi_i
$$

对优化变量求偏导：

$$
\begin{align*}
    \dfrac{\partial \mathcal{L}}{\partial w}&=w-\sum \alpha_iy_ix_i=0&\implies w=\sum \alpha_iy_ix_i\\
    \dfrac{\partial \mathcal{L}}{\partial b}&=-\alpha_iy_i=0&\implies \alpha_iy_i=0\\
    \dfrac{\partial \mathcal{L}}{\partial \xi_i}&=C-\alpha_i-\mu_i=0&\implies C=\alpha_i+\mu_i
\end{align*}
$$

带入乘子得到**对偶问题**：

$$
\mathrm{arg}\max_{\alpha\quad}W(\alpha)=\mathrm{arg}\max_{\alpha\quad}\sum \alpha_i - \dfrac 12 \sum\sum \alpha_i \alpha_j y_i y_j (x_i^\top x_j)\mathrm{\quad s.t.\quad}\sum \alpha_iy_i=0,\alpha_i\in [0,C]
$$

将以上需要的所有约束汇总就得到了 **KKT条件**：

$$
\begin{cases}
    \alpha_i,\mu_i,\xi_i\ge 0\\
    y_i(w^\top x_i+b)-1+\xi_i\ge 0\\
    \alpha_i[y_i(w^\top x_i+b)-1+\xi_i]=0\\
    \mu_i\xi_i=0
\end{cases}
$$

由于 $C=\alpha_i+\mu_i$，对 $\alpha_i$ 讨论：

- $\alpha_i=0$ 则 $\mu_i=C,\xi_i=0$，是可以被硬分类的样本点。
- $\alpha_i\in(0,C)$ 也是可以被硬分类的样本点，但是 $y_i(w^\top x_i+b)-1=0$，为之前所述的**支持向量**。
- $\alpha_i=C$，落在软间隔内，**也是支持向量**。

由于 $w=\sum \alpha_iy_ix_i$，按上面的定义，$w$ **仅由支持向量所决定**，下面的问题就是需要高效求解 $\alpha_i$ 了。

也就是说最后得到的分类器是：

$$
f(x)=\mathrm{sign}\left(b+\sum \alpha_iy_ix_i^\top x\right)
$$

当然 $x_i^\top x$ 这个内积可以替换成核函数的内积 $K(x_i,x)=\phi(x_i)^\top \phi(x)$ 以缓解维度灾难，但 Kernel trick 不是我们推导 SMO 算法的主题。

## SMO 求解

在我看来 SMO 算法有点类似 EM 算法，面对一系列相互制衡的优化变量，它们都选择各个击破。

SMO 算法的思想是每次选两个变量，固定其他不动，然后根据各种等式得到一个二次方程，再把解约束到条件内，不断迭代求解。

具体而言，我们先选择 $\alpha_1$ 和 $\alpha_2$，固定其他的，然后由 $\sum \alpha_i y_i=0$ 可得

$$
\alpha_1 y_1 + \alpha_2 y_2 = \zeta\mathrm{(const.)}\implies \alpha_1=(\zeta-\alpha_2 y_2)y_1
$$

这样，由于 $\alpha_1$ 不是自由的，其他 $\alpha$ 又是人为固定的，则

$$
W(\alpha)=W(\alpha_2)=c\alpha^2_2+b\alpha+a
$$

为一个二次方程——我们可以解析地求出其极值 $\hat\alpha_2$！

但是这个极值不一定满足约束，因此我们需要对其进行裁剪。也就是基于 $\alpha_1 y_1 + \alpha_2 y_2 = \zeta$ 和 $\alpha_1,\alpha_2\in[0,C]$，再计算 $\alpha_1$。

根据 $\alpha_1$ 和 $\alpha_2$ 的值，基于 KKT 条件就能计算出 $b$ 了。如果出现裁剪算出来导致 $b$ 的值不一样，就取一个平均。

每一次不断重复选择一对 $\alpha$ 进行优化，直到满足 KKT 条件的容差，就可以停止算法了。就初始变量的选择而言，也可以选择那些违反 KKT 条件最严重的样本，这样收敛更快。

## 评述

SMO 算法的训练复杂度至少是样本量 $N$ 的平方级，导致大规模的 SVM 应用相当困难。不过基于 RBF 的 SVM 在使用 SMO 优化后，也能够获得相当高的准确率，并且能够抵抗如 FGSM 等方法的攻击，因为它能够给垃圾样本一个很低的置信度。（参考 [FGSM 的论文](https://arxiv.org/abs/1412.6572)）
