# 扩散模型理论篇: 从多阶段变分自编码器到概率流常微分方程采样器系列

观前提示：本文 $\alpha$ 和 $\beta$ 的定义和原论文差了一个平方的阶，以及 $q$ 和 $p$ 的定义和原论文相反。

## VAE's revenge

让我们回顾一下 VAE 的建模过程：

为了拟合目标分布 $p(x)$，我们引入一个隐变量 $z$，这样对其的建模就变成了 $p(x,z)=p(x|z)p(z)$，而反过来，我们也需要对原变量编码进隐变量中，也就是建模 $q(x,z)=q(z|x)q(x)$。然后我们求这两个联合分布的 KL 散度，也就是 $KL(q(x,z)||p(x,z))$ 来衡量拟合分布和原分布的相似性。然后我们引入强先验的正态性假设，把这个 KL 散度拆出常数得到 $ELBO$，再拆成 MSE 和 KLD 两项。

在对 VAE 的讨论中，我们也详细介绍了由于其强制引入的正态性假设，导致压缩率过高，生成的图像很糊。

这就引入了我们介绍 DDPM 的动机——从纯噪声的 $p(z)$ 一步迈到多样的真实分布 $p(x)$，这一步多少迈得有点大了。但是如果我们使用从 $x_0, x_1, \cdots, x_T$ 的多步解码来代替 VAE 的单步解码呢？

也就是，引入联合分布：$p(x_0, x_1, \cdots, x_T) = p(x_T | x_{T-1}) p(x_{T-1} | x_{T-2}) \cdots p(x_1 | x_0) p(x_0)$ 为我们的“编码器”，负责将 $x_0$ 逐步映射到纯噪声分布 $x_T\sim\mathcal N(0,I)$，然后反过来是“解码器” $q(x_0, \cdots, x_T) = q(x_0 | x_1) q(x_1 | x_2) \cdots q(x_{T-1} | x_T) q(x_T)$ 负责将噪声逐步还原到原图像。

下面，让我们开始进行变分推断吧。首先是 KL 散度：

$$
\begin{align*}
    KL(p \Vert q) &= \int p \log \frac{p}{q} \mathrm dx_T \cdots \mathrm dx_0\\&= \int p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0) \log \frac{p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0)}{q(x_0 | x_1) \cdots q(x_{T-1} | x_T) q(x_T)} \mathrm dx_T \cdots \mathrm dx_0
\end{align*}
$$

现在我们仍然需要对“编码过程” $p$ 引入归纳偏置。由于我们是将图像转化为纯噪声，所以我们可以把每一步 $p$ 看作是一个**逐步加噪**的过程：

$$
x_i=\alpha_i x_{i-1}+\beta_i\varepsilon_i,\quad \varepsilon_i\sim\mathcal{N}(0,I)
$$

这里的 $\alpha_i$ 和 $\beta_i$ 是事前给定的参数（其实 $\dfrac{\alpha_i}{\beta_i}$ 可以理解成信噪比），需要满足 $\alpha_i^2+\beta_i^2=1$。为什么要满足这个条件呢？让我们考虑把 $x_i$ 一直展开到 $x_0$：

$$
\begin{align*}
    x_i&=\alpha_i x_{i-1}+\beta_i\varepsilon_i\\
    &=\alpha_i (\alpha_{i-1} x_{i-2}+\beta_{i-1}\varepsilon_{i-1})+\beta_i\varepsilon_i\\
    &=(\alpha_i\alpha_{i-1}\cdots\alpha_1)x_0+\beta_i\varepsilon_i+\alpha_i\beta_{i-1}\varepsilon_{i-1}+\cdots+(\alpha_i\alpha_{i-1}\cdots\alpha_2)\beta_1\varepsilon_1
\end{align*}
$$

由于诸 $\varepsilon$ 是独立的正态分布，可以叠加：

$$
\beta_i\varepsilon_i+\alpha_i\beta_{i-1}\varepsilon_{i-1}+\cdots+(\alpha_i\alpha_{i-1}\cdots\alpha_2)\beta_1\varepsilon_1=\hat\beta_i\hat\varepsilon_i,\quad\hat\varepsilon_i\sim\mathcal{N}(0,I)
$$

如果我们取 $\hat\alpha_i^2=(\alpha_i\alpha_{i-1}\cdots\alpha_1)^2$，且由正态分布方差的叠加得到 $\hat\beta_i^2=\beta_i^2+\alpha_i^2\beta_{i-1}^2+\cdots+(\alpha_i\alpha_{i-1}\cdots\alpha_2)^2\beta_1^2$，然后把它们加起来：

$$
\begin{align*}
    \hat\alpha_i^2+\hat\beta_i^2&=\beta_i^2+\alpha_i^2\beta_{i-1}^2+\cdots+(\alpha_i\alpha_{i-1}\cdots\alpha_2)^2\beta_1^2+(\alpha_i\alpha_{i-1}\cdots\alpha_1)^2\hat\beta_i^2\\
    &=\beta_i^2+\alpha_i^2\beta_{i-1}^2+\cdots+(\alpha_i\alpha_{i-1}\cdots\alpha_2)^2(\beta_1^2+\alpha_1^2)
\end{align*}
$$

这样我们就发现，如果满足 $\alpha_i^2+\beta_i^2=1$，那么就有点像高中学过的裂项相消“点鞭炮”，从后面一直算到前面，最终推出 $\hat\alpha_i^2+\hat\beta_i^2=1$。

这有什么用呢？刚刚的推导中，我们其实得到了一个非常有用的式子：

$$
x_i=\hat\alpha_i x_0+\hat\beta_i\hat\varepsilon_i,\quad\hat\varepsilon_i\sim\mathcal{N}(0,I),\ \hat\alpha_i^2+\hat\beta_i^2=1
$$

这就意味着，为了获取加噪的中间结果，我们可以一步从 $x_0$ 获得。

现在让我们回过头来，看看单步加噪过程 $x_i=\alpha_i x_{i-1}+\beta_i\varepsilon_i$，我们其实可以把它看作是 $\varepsilon_i$ 这个**正态分布的重参数化**！

也就是，我们可以把加噪过程的递推式写成条件分布的形式：$p(x_i | x_{i-1}) = \mathcal{N}(x_i; \alpha_t x_{i-1}, \beta_i^2 I)，\alpha_i^2 + \beta_i^2 = 1$

基于此，我们初步来整理一下 KL 散度的式子：
$$\int p \log \frac{p}{q} \mathrm dx_T \cdots \mathrm dx_0 = \int p \log p \mathrm dx_T \cdots \mathrm dx_0 - \int p \log q \mathrm dx_T \cdots \mathrm dx_0
$$

注意到对 $p$ 而言，所有参数和分布都是定死的，没有可学习的参数，那么上式的第一部分就是一个常数，可以丢掉。

下面我们着重算第二部分：

$$
\begin{align*}
    ELBO &=- \int \left[ p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0) \right] \left( \sum_{i=1}^T \log q(x_{i-1} | x_i) + \log q(x_T) \right) \mathrm dx_T \cdots \mathrm dx_0\\
    &= - \sum_{i=1}^T \int p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0) \log q(x_{i-1} | x_i) \mathrm dx_T \cdots \mathrm dx_0
\end{align*}
$$

这里把 $\log q(x_T)$ 丢掉，是因为 $q(x_T)$ 是加噪后的图像，也没有可学习的参数。

对 $x_{i+1} \cdots x_T$ 而言，这部分积分：

$$
\int p(x_T | x_{T-1}) \cdots p(x_{i+1} | x_i)\mathrm dx_T \cdots \mathrm dx_0
$$

因为这一块和真正待学习的 $x_i, x_{i-1}$ 无关，可以先积出来一个常数，然后就可以丢掉了。

而对 $x_i \cdots x_0$ 而言，我们有

$$
p(x_i | x_{i-1}) \cdots p(x_1 | x_0) p(x_0) = p(x_i | x_{i-1}) p(x_{i-1} | x_0) p(x_0)
$$

也就是刚刚提到的多步并一步的加噪。现在改写得到的 ELBO 如下：

$$
ELBO= - \sum_{i=1}^T \int p(x_i | x_{i-1}) p(x_{i-1} | x_0) p(x_0) \log q(x_{i-1} | x_i) \mathrm dx_T \cdots \mathrm dx_0
$$

下面我们要对 $q$ 进行建模了，我们还是借鉴从 VAE 里面学到的观点，它虽然作为一个在 $x_i$ 上“去噪”的过程，但仍然可以将其建模成一个条件正态分布：

$$
q(x_{i-1} | x_i) = \mathcal{N}(x_{i-1}; x_i, \sigma_t^2)
$$

简单展开一下然后取个对数：

$$
-\log q(x_{i-1} | x_i) \propto \frac{1}{2\sigma_t^2} \| x_{i-1} - \mu(x_i) \|^2
$$

下面，我们对均值 $\mu(x_i)$ 进行讨论。

由于在生成时，$x_i = \alpha_i x_{i-1} + \beta_i \varepsilon_i$，也就是 $x_{i-1} = \frac{1}{\alpha_i}(x_i - \beta_i \varepsilon_i)$。我们希望去噪之后，尽量贴合原分布 $x_{i-1}$，也就是取

$$
\mu(x_i) = \frac{1}{\alpha_i} \left[ x_i - \beta_i \varepsilon_\theta(x_i, i) \right]
$$

这里的 $\varepsilon_\theta(x_i, i)$ 就是可学习的去噪网络。由此可得：

$$
\begin{align*}
    \| x_{i-1} - \mu(x_i) \|^2 &= \| x_{i-1} - \frac{1}{\alpha_i} \left[ \alpha_i x_{i-1} + \beta_i \varepsilon_i - \beta_i \varepsilon_\theta(x_i, i) \right] \|^2\\
    &= \frac{\beta_i^2}{\alpha_i^2} \| \varepsilon_\theta(x_i, i) - \varepsilon_i \|^2
\end{align*}
$$

这个损失函数的意思是，我们输入每一步的带噪图片 $x_i$ 以及时间参数 $i$，用来预测噪声。

当然，我们也可以让损失不依赖于 $x_i$ 而是像之前一样直接从 $x_0$ 获取，对其展开一下：

$$
\begin{align*}
    x_i &= \alpha_i x_{i-1} + \beta_i \varepsilon_i = \alpha_i \left( \hat{\alpha}_{i-1} x_0 + \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} \right) + \beta_i \varepsilon_i\\
    &= \hat{\alpha}_i x_0 + \alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i
\end{align*}
$$

这样，我们的损失就只依赖于固定的原分布 $p(x_0)$ 以及两个随机变量，代回来得到损失函数：

$$
\sum_{i=1}^T \frac{\beta_i^2}{\alpha_i^2 \sigma_i^2} \mathbb{E}_{x_0 \sim p(x_0), \hat{\varepsilon}_{i-1}, \varepsilon_i \sim \mathcal{N}(0, I)} \left[ \| \varepsilon_i - \varepsilon_\theta\left( \hat{\alpha}_i x_0 + \alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i, i \right) \|^2 \right]$$

对 $\alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i$ 而言,其为两个正态分布的叠加，就可写作一个正态分布 $\mathcal{N}\left( 0, \sqrt{\alpha_i^2 \hat{\beta}_{i-1}^2 + \beta_i^2} \right)$，其中 $\alpha_i^2 (1 - \hat{\alpha}_{i-1}^2) + \beta_i^2 = 1 - \hat{\alpha}_i^2 = \hat{\beta}_i^2$，因此，我们可以写成：

$$
\alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i=\hat{\beta}_i^2\varepsilon,\quad\varepsilon\sim\mathcal{N}(0,I)
$$

为了消掉 $\hat{\varepsilon}_{i-1}, \varepsilon_i$ 中的一个，这里需要配一个 $w$，主要用2条性质：
$\begin{cases} \hat{\beta}_i w, \ w \sim \mathcal{N}(0, I) \\ \mathbb{E}[\varepsilon w^\top] = 0 \end{cases}$

而 $w$ 也需要能用 $\hat{\varepsilon}_{i-1}, \varepsilon_i$ 表达。考虑到 $\hat{\varepsilon}_{i-1}和\varepsilon_i$ 的独立性，交换 $\varepsilon$ 中的系数，取 $\hat{\beta}_i w = \beta_i \hat{\varepsilon}_{i-1} - \alpha_i \hat{\beta}_{i-1} \varepsilon_i$ 即可满足以上要求。

再从 $\varepsilon, w$ 中解出 $\varepsilon_i = \frac{\beta_i \varepsilon - \alpha_i \hat{\beta}_{i-1} w}{\hat{\beta}_i}$（利用 $\beta^2_t+\alpha^2_t\hat\beta^2_{t−1} = \hat\beta_i^2$ ）

这样期望项变成了：

$$
\mathbb{E}_{w \sim \mathcal{N}(0, I), \varepsilon \sim \mathcal{N}(0, I)} \left[\| \frac{\beta_i \varepsilon - \alpha_i \hat{\beta}_{i-1} w}{\hat{\beta}_i} - \varepsilon_\theta\left( \hat{\alpha}_i x_0 + \beta_i \varepsilon, i \right)\|^2 \right]
$$

由于 $w$ 和 $\varepsilon$ 独立，先对 $w$ 求期望得一常数，去掉之后，就得到了原论文 DPPM 的损失：

$$
\mathcal{L}_{\mathrm{DPPM}} = \sum_{i=1}^T \frac{\beta_i^4}{\hat{\beta}_i^2 \alpha_i^2 \sigma_i^2} \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I), x_0 \sim p(x_0)} \left[ \| \varepsilon - \frac{\hat{\beta}_i}{\beta_i} \varepsilon_\theta\left( \hat{\alpha}_i x_0 + \beta_i \varepsilon, i \right) \|^2 \right]
$$

## From the perspective of SDE

Yang Song 的文章 arXiv: 2011.13456 将扩散模型和得分匹配相联系，并且引入了随机微分方程作为它们共同的理论基础。这就大大提升了 DDPM 的理论高度，使之不局限于“加噪——去噪”的原初思路。

引入 SDE 的意义不仅在于找到一个数学工具来研究扩散模型，更在于其可以直接转化为概率流 ODE 进行求解，这就可以将 ODE 的数值解法用来加速扩散模型的收敛。这就催生了诸如 Euler, DPM Solver 等一众采样器。

让我们开始介绍 Song 的论文第一部分的工作：联系 DDPM 和得分匹配。这一节的目的，是关联上得分匹配算法的损失函数

$$
\| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2
$$

其中 $\nabla \log p(x_i)$ 被称作得分函数。感性理解，我们是在拟合一个梯度场，让这个梯度场去指引我们的生成。

### 从 DDPM 到得分匹配

为了推出得分匹配形式的损失，我们先引入 Tweddie's Formula。

回顾前向过程 $p(x_i | x_{i-1}) = \mathcal{N}(x_i; \hat{\alpha}_i x_0, \hat{\beta}_i^2 I)$

我们需要往回估计反向过程。考虑正态分布 $p(x|\theta) = \mathcal{N}(\theta, \sigma^2 I)$

其边缘分布 $p(x) = \int p(x|\theta) p(\theta) d\theta$，现在已知 $x$，我们要求 $\theta$ 即：

$$
\mathbb{E}[\theta | x] = \int \theta p(\theta | x) \mathrm d\theta = \int \theta \frac{p(x|\theta) p(\theta)}{p(x)} \mathrm d\theta
$$

由于 $p(x)$ 已知，可以提到积分号外：

$$
\mathbb{E}[\theta | x]= \frac{1}{p(x)} \int \theta \cdot \frac{1}{\sigma \sqrt{2\pi}} \exp\left[ -\frac{\|x - \theta\|^2}{2\sigma^2} \right] p(\theta) \mathrm d\theta
$$

这里我们凑一个 $\dfrac{\mathrm d p(x|\theta)}{\mathrm d x} = \dfrac{\theta - x}{\sigma^2} \cdot p(x|\theta)$，然后接着往下推：

$$
\begin{align*}
    \mathbb{E}[\theta | x]&= \frac{\sigma^2}{p(x)} \int \frac{\theta - x}{\sigma^2} p(x|\theta) p(\theta) + \frac{x}{\sigma^2} p(x|\theta) p(\theta) \mathrm d\theta\\
    &= \frac{\sigma^2}{p(x)} \int \frac{\mathrm d p(x|\theta)}{\mathrm d x} p(\theta) \mathrm d\theta + \frac{\sigma^2}{p(x)} \int \frac{x}{\sigma^2} p(x|\theta) p(\theta) \mathrm d\theta
\end{align*}
$$

由于 $\dfrac{\mathrm d}{\mathrm d x}$ 和 $\theta$ 无关，则

$$
\int \frac{\mathrm d}{\mathrm d x} p(x|\theta) p(\theta) \mathrm d\theta = \frac{\mathrm d}{\mathrm d x} \int p(x|\theta) p(\theta) \mathrm d\theta = \frac{\mathrm d p(x)}{\mathrm d x}
$$

同理，后面一半可以提出 $x$，得到

$$
\frac{x}{p(x)} \int p(x|\theta) p(\theta) \mathrm d\theta = x
$$

因此：

$$
\mathbb{E}[\theta | x] = x + \frac{\sigma^2}{p(x)} \frac{\mathrm d}{\mathrm d x} p(x) = x + \sigma^2 \frac{\mathrm d}{\mathrm d x} \log p(x)
$$

若 $x$ 为向量，则写作 $x + \sigma^2 \nabla \log p(x)$，此即为 Tweedie's Formula．

把这个估计代回前向过程，即 ${\alpha}_i x_{i-1} = x_i + {\beta}_i^2 \nabla \log p(x_i)$

让我们回顾一下：$x_i = {\alpha}_i x_{i-1} + {\beta}_i \varepsilon_i$，代上去可得，$\nabla \log p(x_i) = -\dfrac{\varepsilon_i}{{\beta}_i}$。 这里已经有点味道了：之前我们已经讨论过 DDPM 的去噪过程是去学习每一步的噪声 $\varepsilon_i$，而这个得分函数恰巧也是这个形式，最多差一个系数。

回顾一下之前的推导，从 $\| x_{i-1} - \mu(x_i) \|^2$，我们有：

$$
\begin{cases} x_{i-1} = \frac{1}{{\alpha}_i} (x_i - {\beta}_i \varepsilon_i) \\ \mu(x_i) = \frac{1}{{\alpha}_i} (x_i - {\beta}_i \varepsilon_\theta(x_i, i)) \end{cases} \implies \| x_{i-1} - \mu(x_i) \|^2 = \frac{{\beta}_i^2}{{\alpha}_i^2} \| \varepsilon_\theta(x_i, i) - \varepsilon_i \|^2
$$

由 $-\varepsilon_i = {\beta}_i \nabla \log p(x_i)$，我们取 $s_\theta(x_i, i) = -\dfrac{1}{{\beta}_i} \varepsilon_\theta(x_i, i)$，可得

$$
\| x_{i-1} - \mu(x_i) \|^2=\dfrac{{\beta}_i^4}{{\alpha}_i^2 \sigma_i^2} \| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2
$$

注意，此时它只和 $x_i$ 有关了。

我们可以写出损失函数了：

$$
\mathcal{L}_{\text{DDPM}} = \sum_{i=1}^T \dfrac{\beta_i^4}{\alpha_i^2 \sigma_i^2} \mathbb{E}_{x_i \sim p(x_i)} \left[ \| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2 \right]
$$

这就是得分匹配形式的损失函数。我们需要训练一个网络 $s_\theta(x_i, i)$ 接受每一步的图像 $x_i$ 和时间 $i$ 去匹配这个得分函数 $\nabla \log p(x_i)$。

这里提一嘴，网上很多 DDPM 的得分匹配形式的推导，用的得分函数是 $\nabla_{x_i}\log p(x_i|x_0)=-\dfrac{\hat\varepsilon_i}{\hat\beta_i}$。不过这样推过来就稍显复杂。只要注意到

$$
p(x_i)=\int p(x_i|x_0)p(x_0)\mathrm dx_0=\mathbb{E}_{x_0\sim p(x_0)}[p(x_i|x_0)]
$$

再带入得分函数，就可以知道两者等价了。[此事在科学空间中已有记载](https://kexue.fm/archives/9509)。

### 关联上随机微分方程

#### 前向过程

下面我们开始介绍 Song 的论文第二部分：将加噪和去噪的过程关联上随机微分方程。

为此，我们考虑把一共 $T$ 步的离散过程，转化为对 $t\in[0,1]$ 的连续过程的微元近似，因此我们先做换元，引入连续量：

$$
x_i = x(t),\quad\alpha_i = \sqrt{1 - \frac{1}{T} \beta\left(t + \frac{1}{T}\right)} = \sqrt{1 - \Delta t \cdot \beta(t + \Delta t)},\\
x_{i+1} = x\left(t + \frac{1}{T}\right) = x(t + \Delta t),\quad\beta_i = \sqrt{\frac{1}{T}} \beta\left(t + \frac{1}{T}\right) = \sqrt{\Delta t \cdot \beta(t + \Delta t)}
$$

这里 $T$ 即总步数，$\dfrac{1}{T}$ 即我们要引入的时间微元 $\Delta t$。

我们对 $\alpha_i$ 作泰勒展开 $\alpha_i \sim 1 - \dfrac{\beta(t + \Delta t) \cdot \Delta t}{2}$，然后替换一下，得到：

$$
x(t + \Delta t) = \left[ 1 - \frac{\beta(t + \Delta t) \cdot \Delta t}{2} \right] x(t) + \sqrt{\beta(t + \Delta t)} \cdot \sqrt{\Delta t} \ \varepsilon(t)
$$

减去 $x(t)$ 得到：

$$
\mathrm dx = -\frac{\beta(t) \cdot x(t)}{2} \mathrm dt + \sqrt{\beta(t)} \cdot \sqrt{\mathrm dt} \cdot \varepsilon(t)
$$

取 $f[x(t), t] = -\dfrac{\beta(t) \cdot x(t)}{2}，g(t) = \sqrt{\beta(t)}，\mathrm dw = \varepsilon(t) \cdot \sqrt{\mathrm dt}$（其中 $\mathrm dw$ 为布朗运动噪声，即“扩散项”），则有：

$$
\mathrm dx = f[x(t), t] \mathrm dt + g(t) \mathrm dw
$$

（为什么布朗运动的噪声和 $\sqrt{\mathrm dt}$ 有关呢？请参阅本文[附录 I](https://dicaeopolis.github.io/DNN/model-expr/DDPM/#i)）

这就是加噪过程满足的 SDE。在原论文中对应 VP-SDE 那一节，也就是离散近似的 SDE。

#### 变量替换

我们对这个 SDE 做一些变换，导出其更有用的形式。

从 $\alpha_i \sim 1 - \dfrac{\beta(t + \Delta t) \cdot \Delta t}{2}$ 也就是 $\alpha_i = 1 - \dfrac{\beta(t_i)}{2}\mathrm dt$

让我们计算 $\hat \alpha_i$ 在连续意义上的对应 $\hat \alpha(t)$，由于涉及到连乘，我们两边取对数：

$$
\begin{align*}
    \log \hat \alpha(t)&=\sum_{k=1}^i\log \alpha_i\\
    &=\sum_{k=1}^i\log (1 - \dfrac{\beta(t_i)}{2}\mathrm dt)\\
    &\sim \sum_{k=1}^i - \dfrac{\beta(t_i)}{2}\mathrm dt\\
    &=-\frac 12\int_0^t\beta(t) \mathrm dt
\end{align*}
$$

这里利用了 $\log$ 的一阶泰勒展开。对应的，我们有

$$
\hat \beta^2(t)=1-\hat \alpha^2(t)=1-\exp(-\int_0^t\beta(t) \mathrm dt)
$$

以及

$$
f(t)=-\dfrac{\beta(t)}{2}=\dfrac{\mathrm d \log\hat\alpha(t)}{\mathrm dt}
$$

如果我们对 $\hat \beta^2(t)$ 求导：

$$
\dfrac{\mathrm d \hat \beta^2(t)}{\mathrm dt}=\beta(t)\left(\exp(-\int_0^t\beta(t) \mathrm dt)\right)=\beta(t)(1-\hat \beta^2(t))=\beta(t)-\beta(t)\hat \beta^2(t)
$$

也就是

$$
g^2(t)=\beta(t)=\dfrac{\mathrm d \hat \beta^2(t)}{\mathrm dt}-\dfrac{2\mathrm d \log\hat\alpha(t)}{\mathrm dt}\hat \beta^2(t)
$$

在此意义下我们的 SDE 写成：

$$
\mathrm dx=f(t)x(t)\mathrm dt+g(t)\mathrm dw
$$

引入这部分推导，主要是和之前 DDPM 多步并一步的目的是一样的，我们要消去比较麻烦的 $\beta(t)$，转化为可以一步得到的 $\hat\beta(t)$ 和 $\hat\alpha(t)$，同时，也是为后面 DPM Solver 的推导服务。

#### 反向过程

那么如何获得去噪过程的反向 SDE 呢？又如何与刚才得到的得分匹配形式相联系呢？当然是利用贝叶斯公式，为此我们先将上面的 SDE 写成条件分布：

$$
p(x_{t+\Delta t} | x_t) = \mathcal{N}\bigl(x_{t+\Delta t}; \ x_t + f_x(t) \mathrm dt, \ g^2(t) \mathrm dt \cdot I\bigr)
$$

现在，就可以利用贝叶斯公式了

$$
\begin{align*}
    p(x_t | x_{t+\Delta t}) &= \dfrac{p(x_{t+\Delta t} | x_t) \cdot p(x_t)}{p(x_{t+\Delta t})}\\
    &= \exp\left[ \log p(x_{t+\Delta t} | x_t) + \log p(x_t) - \log p(x_{t+\Delta t}) \right]\\
    &\propto \exp\left[ -\dfrac{1}{2 g^2(t) \Delta t} \| x_{t+\Delta t} - x_t - f_x(t) \Delta t \|^2 + \log p(x_t) - \log p(x_{t+\Delta t}) \right]
\end{align*}
$$

为了算下去，我们要对 $\log p(x_{t+\Delta t})$ 作展开：

$$
\log p(x_{t+\Delta t}) \approx \log p(x_t) + (x_{t+\Delta t} - x_t) \cdot \nabla_x \log p(x_t) + O(\Delta t)
$$

然后作差：

$$
\log p(x_t) - \log p(x_{t+\Delta t}) = -\dfrac{1}{2 g^2(t) \Delta t} \left[ (x_{t+\Delta t} - x_t) \cdot \nabla_x \log p(x_t) \cdot 2 g^2(t) \Delta t \right] + O(\Delta t)
$$

我们的目的其实是写成一个和正态分布类似的 exp 加模平方的形式。为此，这里我们可以配一个

$$
\left[ g^2(t) \Delta t \cdot \nabla_x \log p(x_t) \right]^2 + 2 \left[ g^2(t) \nabla_x \log p(x_t) \Delta t \right] \times f_x(t) \Delta t
$$

因为它们都是 $\Delta t$ 的二阶项，最后都能消失。不过，配上之后就变成了完全平方式：

$$
p(x_t | x_{t+\Delta t}) = \exp\left[ -\frac{1}{2 g^2(t) \Delta t} \left\| x_{t+\Delta t} - x_t - \left[ f_x(t) - g^2(t) \nabla_x \log p(x_t) \right] \Delta t \right\|^2 \right]
$$

写成正态分布形式：

$$
p(x_t | x_{t+\Delta t})\sim \mathcal{N}\bigl( x_t; \ x_{t+\Delta t} - \left[ f_x(t) - g^2(t) \nabla_x \log p(x_{t+\Delta t}) \right] \Delta t, \ g^2(t+\Delta t) \Delta t \cdot I \bigr)
$$

求极限 $\Delta t\rightarrow 0$，再由条件分布转化为 SDE，得到：

$$
\mathrm dx = \left[ f[x(t), t] - g^2(t) \nabla_x \log p(x_t) \right] \mathrm dt + g(t) \mathrm dw
$$

这就是反向过程的 SDE 了。

对于 $f, g$ 而言，它们完全确定，因此我们需要估计得分函数 $\nabla_x \log p(x_t)$；或者换成离散形式的记号：$\nabla \log p(x_i)$。

使用一个神经网络 $s_\theta(x_i, i)$ 来拟合得分函数，就得到目标函数：

$$
\sum_{i=1}^T \lambda_i \mathbb{E}_{x_i \sim p(x_i)} \left[ \| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2 \right] = \mathcal{L}_{\mathrm{DDPM}}
$$

这里 $\lambda_i$ 是基于 $p(x_i)$ 引入“噪声尺度不一”的归一化因子。

至此，DDPM、得分匹配和 SDE 的理论已然打通。我们就可以基于丰富发展的 SDE 理论，玩一些花活了。

### 将 SDE 变成 ODE

这一节介绍 Song 的论文的第三部分：概率流 ODE 的推导。

也就是，将

$$
\mathrm dx = \left[ f[x(t), t] - g^2(t) \nabla_x \log p(x_t) \right] \mathrm dt + g(t) \mathrm dw
$$

转化为概率流 ODE。

我们**肯定不能**直接把 $g(t) \mathrm dw$ 项给丢掉，因为方差影响了 SDE 诸多解的“弥散程度”。因此我们需要考虑这一项对总体趋势的影响。或者我们也可以这样看：能不能使用什么手段，手动引入一个可控的方差也就是 $\sigma(t) \mathrm dw$ 来代替原来扩散项，这样就能间接实现整合。

这个手段就是 Fokker-Planck 方程。

#### 前向过程的 Fokker-Planck 方程

考虑让 $y$ 和 $x$ 一一对应，那么

$$
p(x)=\int \delta(x-y)p(y)\mathrm dy=\mathbb E_{y}[\delta(x-y)]
$$

那么对于 $p(x_{t+\Delta t})$ 而言：

$$
\begin{align*}
    p(x_{t+\Delta t})&=\mathbb E_{x_{t+\Delta t}}[\delta(x-x_{t+\Delta t})]\\
    &=\mathbb E_{x_{t+\Delta t}}[\delta(x-x_{t}-\Delta x)]\\
    &\approx\mathbb E_{x_{t+\Delta t}}[\delta(x-x_{t})-\Delta x\cdot\nabla_x\delta(x-x_{t})+\dfrac 12 \Delta x^2\cdot\nabla_x^2\delta(x-x_{t})]\\
    &=\mathbb E_{x_{t+\Delta t}}[\delta(x-x_{t})-(f(t)x_t\mathrm dt)\cdot\nabla_x\delta(x-x_{t})+\dfrac 12 g^2(t)\mathrm dt\cdot\nabla_x^2\delta(x-x_{t})]\\
    &=p(x_t)-\nabla_x [f(t)x_t p(x_t)\mathrm dt]+\dfrac 12g^2(t)\mathrm dt\nabla_x^2p(x_t)
\end{align*}
$$

此即为 Fokker-Planck 方程：

$$
\dfrac{\partial p}{\partial t}=-\nabla_x [f(t)x_t p(x_t)]+\dfrac 12g^2(t)\nabla_x^2p(x_t)
$$

注意到整个推导是和前向过程的 $\mathrm dt$ 前面的系数无关的，因此对于反向过程我们也可以带进去得到：

$$
\dfrac{\partial p}{\partial t}=-\nabla_x [ \left[ f[x(t), t] - g^2(t) \nabla_x \log p(x_t) \right] p(x_t)]+\dfrac 12g^2(t)\nabla_x^2p(x_t)
$$

由于二阶项前面的系数直接对应 $\mathrm dw$ 前面的系数，这样就给了我们操作空间，也就是引入一个 $\dfrac 12\sigma^2(t)\nabla_x^2p(x_t)$

## Samplers

### Euler

本节介绍 Song 的论文第四部分的工作，使用 ODE 求解器来实现采样。

### DPM Solver

本节主要介绍 arXiv:2206.00927 的工作。

## Appendices

### I. 布朗运动的二次变分

我们要推导一个布朗运动 $B(t)$ 满足 $\mathrm dB=\sqrt{\mathrm dt}$，即 $(\mathrm{d}B)^2=\mathrm{d}t$。

我们换成积分式，也就是在 $[0,T]$ 内有

$$
\int_0^T (\mathrm{d}B)^2=\int_0^T\mathrm{d}t=T
$$

换成定义式，也就是对该区间的一个划分 $\Pi$，最大步长记作 $|\Pi|$，然后证明极限：

$$
\lim_{|\Pi|\rightarrow0}\sum_{i}[B(t_{i+1})-B(t_i)]^2=\lim_{|\Pi|\rightarrow0} S_n=T
$$

这就是布朗运动的**二次变分**。由于布朗运动的独立性，有

$$
B(t_{i+1})-B(t_i)=\Delta B_i\sim N(0,\Delta t_i)
$$

则根据正态分布二阶矩的性质， $\mathbb{E}[(\Delta B_i)^2]=\Delta t_i$，叠在一起就可以得到

$$
\mathbb{E}[S_n]=\sum\Delta t_i=\int_0^T\mathrm{d}t=T
$$

而 $\mathrm{Var}[(\Delta B_i)^2]=\mathbb{E}[(\Delta B_i)^4]-\mathbb{E}[(\Delta B_i)^2]^2=3(\Delta t_i)^2-(\Delta t_i)^2=2(\Delta t_i)^2$，即

$$
\mathrm{Var}[S_n]=\sum 2(\Delta t_i)^2\le 2|\Pi|T
$$

故 $|\Pi|\to 0$ 则 $\mathrm{Var}[S_n]\to 0$，根据大数定律，

$$
\lim_{|\Pi|\rightarrow0} S_n=\lim_{|\Pi|\rightarrow0} \mathbb{E}[S_n]=T
$$

这样就得到了 $(\mathrm{d}B)^2=\mathrm{d}t$。

题外话：由于布朗运动是处处连续处处不可导的，这才导致了其二次变分的值不为零。考虑一个连续函数 $f$，我们来考虑其二次变分，利用中值定理：

$$
\begin{align*}
    \lim_{|\Pi|\rightarrow0}\sum_{i}[f(t_{i+1})-f(t_i)]^2&=\lim_{|\Pi|\rightarrow0}\sum_{i}[\Delta t_i f'(s_i)]^2\\
    &\le \lim_{|\Pi|\rightarrow0}|\Pi|\sup_{x\in[0,T]} f(x) \sum_i\Delta t_i\\
    &=\lim_{|\Pi|\rightarrow0}|\Pi|\sup_{x\in[0,T]} f(x) T\\
    &=\lim_{|\Pi|\rightarrow0}O(|\Pi|)\\
    &=0
\end{align*}
$$
