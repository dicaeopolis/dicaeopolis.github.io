# DDPM 理论: 从多阶段变分自编码器到基于得分匹配的随机微分方程

（手稿转写，待整理）

联合分布：$p(x_0, \cdots, x_T) = p(x_T | x_{T-1}) p(x_{T-1} | x_{T-2}) \cdots p(x_1 | x_0) p(x_0)$
$q(x_0, \cdots, x_T) = q(x_0 | x_1) q(x_1 | x_2) \cdots q(x_{T-1} | x_T) q(x_T)$
变分推断即最小化 KL 散度 \quad Decoder：去噪
$KL(p \Vert q) = \int p \log \frac{p}{q} dx_T \cdots dx_0$
$= \int p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0) \log \frac{p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0)}{q(x_0 | x_1) \cdots q(x_{T-1} | x_T) q(x_T)} dx_T \cdots dx_0$
其中，$p(x_t | x_{t-1}) = \mathcal{N}(x_t; \alpha_t x_{t-1}, \beta_t^2 I)，\alpha_t^2 + \beta_t^2 = 1$
即：加噪过程仅采样，不含任何可学习的参数。
即：$\int p \log \frac{p}{q} dx_T \cdots dx_0 = \underbrace{\int p \log p dx_T \cdots dx_0}_{\text{常数}} - \int p \log q dx_T \cdots dx_0$
于是我们得到了近似的 ELBO：
$- \int \left[ p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0) \right] \left( \sum_{i=1}^T \log q(x_{i-1} | x_i) + \underbrace{\log q(x_T)}_{\text{常数（正态分布）}} \right) d\cdots$
$= - \sum_{i=1}^T \int p(x_T | x_{T-1}) \cdots p(x_1 | x_0) p(x_0) \log q(x_{i-1} | x_i) d\cdots$
对 $x_{i+1} \cdots x_T$ 而言，这部分积分中的 $\int p(x_T | x_{T-1}) \cdots p(x_{i+1} | x_i)$ 因为上面一项和 $x_i$ 无关，可以先积出来 $\times \text{Const.} \ d\cdots$
对 $x_i \cdots x_0$ 而言，$p(x_i | x_{i-1}) \cdots p(x_1 | x_0) p(x_0) = p(x_i | x_{i-1}) p(x_{i-1} | x_0) p(x_0)$
$= - \sum_{i=1}^T \int p(x_i | x_{i-1}) p(x_{i-1} | x_0) p(x_0) \log q(x_{i-1} | x_i) d\cdots$
对 $q$ 的建模：（在 $x_i$ 上“去噪”）
$q(x_{i-1} | x_i) = \mathcal{N}(x_{i-1}; x_i, \sigma_t^2)$
$-\log q(x_{i-1} | x_i) \propto \frac{1}{2\sigma_t^2} \| x_{i-1} - \mu(x_i) \|^2$
在生成时，$x_i = \alpha_i x_{i-1} + \beta_i \varepsilon_i \implies x_{i-1} = \frac{1}{\alpha_i}(x_i - \beta_i \varepsilon_i)$
由此可取 $\mu(x_i) = \frac{1}{\alpha_i} \left[ x_i - \beta_i \varepsilon_\theta(x_i, i) \right] \quad$ 可学习的去噪
$\implies \| x_{i-1} - \mu(x_i) \|^2 = \| x_{i-1} - \frac{1}{\alpha_i} \left[ \alpha_i x_{i-1} + \beta_i \varepsilon_i - \beta_i \varepsilon_\theta(x_i, i) \right] \|^2$
$= \frac{\beta_i^2}{\alpha_i^2} \| \varepsilon_\theta(x_i, i) - \varepsilon_i \|^2$
又 $x_i = \alpha_i x_{i-1} + \beta_i \varepsilon_i = \alpha_i \left( \hat{\alpha}_{i-1} x_0 + \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} \right) + \beta_i \varepsilon_i$
$= \hat{\alpha}_i x_0 + \alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i$
$-ELBO \implies \sum_{i=1}^T \frac{\beta_i^2}{\alpha_i^2 \sigma_i^2} \mathbb{E}_{\substack{x_0 \sim p(x_0), \\ \hat{\varepsilon}_{i-1}, \varepsilon_i \sim \mathcal{N}(0, I)}} \left[ \| \varepsilon_i - \varepsilon_\theta\left( \hat{\alpha}_i x_0 + \alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i, i \right) \|^2 \right]$
对 $\alpha_i \hat{\beta}_{i-1} \hat{\varepsilon}_{i-1} + \beta_i \varepsilon_i$ 而言，可写作一个正态分布 $\mathcal{N}\left( 0, \sqrt{\alpha_i^2 \hat{\beta}_{i-1}^2 + \beta_i^2} \right)，其中\alpha_i^2 (1 - \hat{\alpha}_{i-1}^2) + \beta_i^2 = 1 - \hat{\alpha}_i^2 = \hat{\beta}_i^2$
这里需要配一个 $w$，主要用2条性质：
$\begin{cases} \hat{\beta}_i w, \ w \sim \mathcal{N}(0, I) \\ \mathbb{E}[\varepsilon w^\top] = 0 \end{cases}$（也可用 $\hat{\varepsilon}_{i-1}, \varepsilon_i$ 表达）。考虑到 $\hat{\varepsilon}_{i-1}和\varepsilon_i$ 的独立性，交换 $\varepsilon$ 中的系数，取 $\hat{\beta}_i w = \beta_i \hat{\varepsilon}_{i-1} - \alpha_i \hat{\beta}_{i-1} \varepsilon_i$，再从 $\varepsilon, w$ 中解出 $\varepsilon_i = \frac{\beta_i \varepsilon - \alpha_i \hat{\beta}_{i-1} w}{\hat{\beta}_i}$（利用 $\sigma_i^2 \hat{\beta}_i^2 = \beta_i^2$ ），故：
$-ELBO = \mathbb{E}_{\substack{w \sim \mathcal{N}(0, I), \\ \varepsilon \sim \mathcal{N}(0, I)}} \left[ \frac{\beta_i \varepsilon - \alpha_i \hat{\beta}_{i-1} w}{\hat{\beta}_i} - \varepsilon_\theta\left( \hat{\alpha}_i x_0 + \beta_i \varepsilon, i \right) \right]^2$
由于 $w$ 和 $\varepsilon$ 独立，先对 $w$ 求期望（得一常数，可去掉），得到DPPM的损失：
$\mathcal{L}_{\text{DPPM}} = \sum_{i=1}^T \frac{\beta_i^4}{\hat{\beta}_i^2 \alpha_i^2 \sigma_i^2} \mathbb{E}_{\substack{\varepsilon \sim \mathcal{N}(0, I), \\ x_0 \sim p(x_0)}} \left[ \| \varepsilon - \frac{\hat{\beta}_i}{\beta_i} \varepsilon_\theta\left( \hat{\alpha}_i x_0 + \beta_i \varepsilon, i \right) \|^2 \right]$
现在引入 Score - Based SDE.
为此，回顾前向过程 $p(x_i | x_{i-1}) = \mathcal{N}(x_i; \hat{\alpha}_i x_0, \hat{\beta}_i^2 I)$
我们需要往回估计反向过程。考虑正态分布 $p(x|\theta) = \mathcal{N}(\theta, \sigma^2 I)$
其边缘分布 $p(x) = \int p(x|\theta) p(\theta) d\theta$，现在已知 $x$，求 $\theta$，
即 $\mathbb{E}[\theta | x] = \int \theta p(\theta | x) d\theta = \int \theta \frac{p(x|\theta) p(\theta)}{p(x)} d\theta -- p(x)$ 已知，则
$= \frac{1}{p(x)} \int \theta \cdot \frac{1}{\sigma \sqrt{2\pi}} \exp\left[ -\frac{\|x - \theta\|^2}{2\sigma^2} \right] p(\theta) d\theta$
这里我们凑一个 $\frac{d p(x|\theta)}{d x} = \frac{\theta - x}{\sigma^2} \cdot p(x|\theta)$
$= \frac{\sigma^2}{p(x)} \int \frac{\theta - x}{\sigma^2} p(x|\theta) p(\theta) + \frac{x}{\sigma^2} p(x|\theta) p(\theta) d\theta$
$= \frac{\sigma^2}{p(x)} \int \frac{d p(x|\theta)}{d x} p(\theta) d\theta + \frac{\sigma^2}{p(x)} \int \frac{x}{\sigma^2} p(x|\theta) p(\theta) d\theta$
由于 $\frac{d}{d x}$ 和 $\theta$ 无关，$\int \frac{d}{d x} p(x|\theta) p(\theta) d\theta = \frac{d}{d x} \int p(x|\theta) p(\theta) d\theta = \frac{d p(x)}{d x}$
同理，后面一半可以提出 $x$ 得到 $\frac{x}{p(x)} \int p(x|\theta) p(\theta) d\theta = x$
因此 $\mathbb{E}[\theta | x] = x + \frac{\sigma^2}{p(x)} \frac{d}{d x} p(x) = x + \sigma^2 \frac{d}{d x} \log p(x)．$
若 $x$ 为向量，则写作 $x + \sigma^2 \nabla \log p(x)$，此即为 Tweedie's Formula．
把这个估计去回前向过程，即 $\hat{\alpha}_i x_{i-1} = x_i + \hat{\beta}_i^2 \nabla \log p(x_i)$
让我们回顾一下；$x_i = \hat{\alpha}_i x_{i-1} + \hat{\beta}_i \varepsilon_i$，代上去可得，$\nabla \log p(x_i) = -\frac{\varepsilon_i}{\hat{\beta}_i}$
现在回到那 ELBO 的那个平方式：$\| x_{i-1} - \mu(x_i) \|^2$，我们有：
$\begin{cases} x_{i-1} = \frac{1}{\hat{\alpha}_i} (x_i - \hat{\beta}_i \varepsilon_i) \\ \mu(x_i) = \frac{1}{\hat{\alpha}_i} (x_i - \hat{\beta}_i \varepsilon_\theta(x_i, i)) \end{cases} \implies = \frac{\hat{\beta}_i^2}{\hat{\alpha}_i^2} \| \varepsilon_\theta(x_i, i) - \varepsilon_i \|^2$
由 $-\varepsilon_i = \hat{\beta}_i \nabla \log p(x_i)$，我们取 $s_\theta(x_i, i) = -\frac{1}{\hat{\beta}_i} \varepsilon_\theta(x_i, i)$，可得
$\frac{\hat{\beta}_i^4}{\hat{\alpha}_i^2 \sigma_i^2} \| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2$，注意此时它只和 $x_i$ 有关了．
我们可以写出损失函数了：
$\mathcal{L}_{\text{DDPM}} = \sum_{i=1}^T \frac{\beta_i^4}{\alpha_i^2 \sigma_i^2} \mathbb{E}_{x \sim p(x)} \left[ \| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2 \right]$.
这就是得分匹配形式的损失函数．
下面，将加噪过程视作一个 SDE：
令 $x_i = x(t)，\alpha_i = \sqrt{1 - \frac{1}{T} \beta\left(t + \frac{1}{T}\right)} = \sqrt{1 - \delta t \cdot \beta(t + \delta t)}$；
$x_{i+1} = x\left(t + \frac{1}{T}\right) = x(t + \delta t)，\beta_i = \sqrt{\frac{1}{T}} \beta\left(t + \frac{1}{T}\right) = \sqrt{\delta t \cdot \beta(t + \delta t)}$．
$T$ 即总步数，$\frac{1}{T}$ 即时间微元 $\delta t$；对 $\alpha_i$ 作泰勒展开：$\alpha_i \sim 1 - \frac{\beta(t + \delta t) \cdot \delta t}{2}$．
$\implies x(t + \delta t) = \left[ 1 - \frac{\beta(t + \delta t) \cdot \delta t}{2} \right] x(t) + \sqrt{\beta(t + \delta t)} \cdot \sqrt{\delta t} \ \varepsilon_t$
$\implies dx = -\frac{\beta(t) \cdot x(t)}{2} dt + \sqrt{\beta(t)} \cdot \sqrt{dt} \cdot \varepsilon(t)$．
取 $f[x(t), t] = -\frac{\beta(t) \cdot x(t)}{2}，g(t) = \sqrt{\beta(t)}，dw = \varepsilon(t) \cdot \sqrt{dt}$（其中 $dw$ 为布朗运动噪声，即“扩散项”），则有：
$dx = f[x(t), t] dt + g(t) dw$.
这就是加噪过程满足的 SDE．
那么如何获得去噪过程的反向 SDE 呢？又如何与刚才得到的得分匹配形式相联系呢？当然是利用贝叶斯．
将上面的 SDE 写成条件分布：$p(x_{t+\delta t} | x_t) = \mathcal{N}\bigl(x_{t+\delta t}; \ x_t + f_x(t) dt, \ g^2(t) dt \cdot I\bigr)$．
利用贝叶斯公式：
$p(x_t | x_{t+\delta t}) = \frac{p(x_{t+\delta t} | x_t) \cdot p(x_t)}{p(x_{t+\delta t})}$
$= \exp\left[ \log p(x_{t+\delta t} | x_t) + \log p(x_t) - \log p(x_{t+\delta t}) \right]$
$\propto \exp\left[ -\frac{1}{2 g^2(t) dt} \| x_{t+\delta t} - x_t - f_x(t) dt \|^2 + \log p(x_t) - \log p(x_{t+\delta t}) \right]$.
对 $\log p(x_{t+\delta t})$ 作展开：
$\log p(x_{t+\delta t}) \approx \log p(x_t) + (x_{t+\delta t} - x_t) \cdot \nabla_x \log p(x_t) + O(\delta t)$.
$\log p(x_t) - \log p(x_{t+\delta t}) = -\frac{1}{2 g^2(t) \delta t} \left[ (x_{t+\delta t} - x_t) \cdot \nabla_x \log p(x_t) \cdot 2 g^2(t) \delta t \right] + O(\delta t)$
这里我们可以配一个 $\left[ g^2(t) \delta t \cdot \nabla_x \log p(x_t) \right]^2 + 2 \left[ g^2(t) \nabla_x \log p(x_t) \delta t \right] \times f_x(t) \delta t$（因为都是 $\delta t$ 的二阶项，最后都能消失）；不过，配上之后就变成了完全平方式：
$p(x_t | x_{t+\delta t}) = \exp\left[ -\frac{1}{2 g^2(t) \delta t} \left\| x_{t+\delta t} - x_t - \left[ f_x(t) - g^2(t) \nabla_x \log p(x_t) \right] \delta t \right\|^2 \right]$
近似为正态分布：
$\sim \mathcal{N}\bigl( x_t; \ x_{t+\delta t} - \left[ f_x(t) - g^2(t) \nabla_x \log p(x_{t+\delta t}) \right] \delta t, \ g^2(t+\delta t) \delta t \cdot I \bigr)$
再由条件分布转化为 SDE：
$dx = \left[ f[x(t), t] - g^2(t) \nabla_x \log p(x_t) \right] dt + g^2(t) dw$
对于 f, g 而言，它们完全确定，因此我们需要估计得分函数 $\nabla_x \log p(x_t)$；或者换成离散形式的记号：$\nabla \log p(x_i)$。
使用一个神经网络 $s_\theta(x_i, i)$ 来拟合得分函数，就得到目标函数：
$\sum_{i=1}^T \lambda_i \mathbb{E}_{x_i \sim p(x_i)} \left[ \| s_\theta(x_i, i) - \nabla \log p(x_i) \|^2 \right] = \mathcal{L}_{\text{DDPM}}$
这里 $\lambda_i$ 是基于 $p(x_i)$ 引入“噪声尺度不一”的归一化因子。
至此，DDPM、得分匹配和 SDE 的理论已然打通。
