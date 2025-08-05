---
comments: true
---

# 神经网络优化器概述

## 何以优化

神经网络的目的是在训练数据集上实现**结构风险最小化**以获得良好的拟合和泛化能力。简单说，如果我们在训练集 $X$ 上有一个定义明确的损失函数 $\mathcal{L}(X;\theta)$（表示我们的结构风险），那么所有优化器的目的都是设计一个算法来寻找合适的 $\theta$ 以获得 $\text{argmin}_\theta\ \mathcal{L}(X;\theta)$。

## 寻找最小值

让我们回忆熟悉的极（小）值寻找方法。对于二次函数而言，我们可以很容易获得极小值的闭型表达式。且这一极小值是全局的。

对于更复杂的一元函数，我们很容易可以证明那些前 $2k-1(k = 1,2,\cdots)$ 阶导数为 $0$，且第 $2k$ 阶导数不为 $0$ 的点为一个极值点。（证明不难，反复进行泰勒展开即可）。

可是神经网络的损失函数是一个极为复杂的多元参数求极小值的过程。我们几乎无法找到一个表达式来表述这个极小值。

但是，我们仍有能力去研究这个极小值附近的数学性质。换言之，如果我们想要下山，即便我们不知道山谷的具体位置，但是向下走一定能走到真正的谷底。唯一的区别就是我们的用时、路径和结果位置不一样罢了。这样，就引入了我们评判优化器的几个核心指标。

## 什么样的优化器是好的优化器？

还是沿用下山的比喻。

- 我们希望尽可能快速地下山。也就是我们希望优化器的收敛速率尽可能快。
- 我们希望我们不会困在小山沟里面，而是深入真正的谷底。也就是我们希望优化器不会困于一些很坏的点，比如说对于做传统算法的同学而言比较熟悉的局部最优点（e.g. 例如一些性质不好的贪心）或者现在主流研究的鞍点（梯度平缓的非极值点），而是有走出去寻找全局最小值的能力。事实上，现在主流的研究方向并非关系**全局**最小，而是鞍点逃逸。毕竟前者只轻微影响训练效果，而后者在更大程度上影响训练速度。
- 我们希望不要指错路，不要乱指路。也就是我们希望在快速收敛的情况下训练稳定，不要出现损失尖峰或是大幅度震荡。

## SGD 系列算法

### 随机梯度下降 SGD

所有的SGD理论讲解都会使用下山的比喻。我们也接着沿用。考虑我们在半山腰，浓雾笼罩，只能看到脚底一片地。如果我们要尽可能快地下山，肯定是**沿着最陡的方向往下**走。

考虑泛函 $\mathcal{L(x; \theta)}$ 是当前的模型 $\theta$ 在某一训练数据 $x$ 下的损失地形。什么地方最陡呢？当然是 $\nabla\mathcal{L(x;\theta)} = \sum_{i=1}^{n} \dfrac{\partial\mathcal{L(x;\theta)}}{\partial \theta_i}\vec{e}_i$ 也就是梯度方向了。证明也很简单，由于各个分量 $\vec{e}_i$ 正交，让每个方向都朝自己的方向变化就可以叠加出最大的变化率。

但是我们不能光看一个数据，那样的话针对性过强了，容易过拟合。那怎么办呢？我们可以一次性取多个数据，具体而言，我们在训练集里面随机取 $|\mathcal{B}|$ 个数据，称作一个 Batch（批量），在这个 Batch 下面我们计算平均梯度 $\dfrac{1}{|\mathcal{B}|}\sum_{i=1}^{|\mathcal{B}|} \nabla\mathcal{L}(x_i;\theta)$，作为“最陡”方向的参考。

下面的问题就是往这个方向走多远的问题了。这是一个我们可以调整的超参数，大了，走得更快，但是容易震荡；小了，收敛又太慢。（此处埋一个伏笔，嘻嘻）这个超参数叫做学习率，用来表征当前的梯度对模型权重进行更新的参考价值。

现在就可以祭出我们的 mini-batch SGD 算法了！对于数据集 $X$ 而言，每次随机抽取 $|\mathcal{B}|$ 条数据 $x$。计算上一代模型 $\theta_{n-1}$ 的更新步长

$$
\begin{align*}
    \Delta \theta &= -\dfrac{\eta}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\nabla\mathcal{L}(x_i;\theta_{n-1})\\
    \theta_n &= \theta_{n-1}+\Delta \theta
\end{align*}
$$

这里加负号意思是“下降”。

这样不仅使用的平均梯度，参考价值较强（而且如果 batch size 更大，对原有数据集分布的估计就更好），更赞的是一个 Batch 里面所有梯度的计算都是独立的，所以天生适合利用 GPU 进行高度并行化的计算！

下面我们来评估一下这个 mini-batch SGD 算法。

- 收敛速率上，对于梯度大的方向收敛快，梯度小的方向收敛慢。如果说损失地形的最低点四周都是很陡的斜坡，那很好，但存在这种混合情况：考虑一个开口向上的椭圆抛物面，并假设我们初始点在椭圆长轴端点附近，这样梯度方向近似和短轴方向平行，模型就一直在前后横跳，真正向下移动（长轴分量）很少。事实上这种情况对应的是一个条件数很大的海森矩阵，相关分析参考后文。
- 寻找全局最小值的能力：在一个平缓的鞍点处算法表现得很“懒”，除非把学习率调大，但是过大的学习率会导致损失不收敛。
- 小学习率当然会使得训练稳定。但是还是那个老生常谈的问题……

### 动量法随机梯度下降 SGDM

#### 动量的引入

人往高处走，水往低处流。我们可以感性体会一下，相比其我们根据坡度（梯度）小心翼翼地下山，从山顶滚落的巨石似乎能比我们更快且更好地找到真正谷底的位置。

让我们对这块石头进行建模。考虑一个单位物体在势场 $U$ 中做带阻尼的自由运动。（带阻尼是为了让物体的动能耗散，以停止在最小值）那么它所受梯度力 $F=-\nabla U$，即 $ma = -\nabla U$。取时间微元 $\beta_3$，则速度更新为 $-v_{n+1} = -\beta_1v_n - \beta_3 a = -\beta_1v_n - \beta_3\dfrac{\nabla U}{m}$，其中 $\beta_1<1$ 表征阻尼损耗，位置更新为 $\theta_{n+1} = \theta_n - \beta_3v_n$。事实上这里 $m$ 表征惯性的大小，惯性大，不易受力移动；惯性小，容易往下移动。这也体现出学习率的一点性质。在更新权重的时候我们可以把时间步长和质量两个参量统一考虑成学习率 $\eta$。现在我们可以把质量乘上去，也就是考虑**动量**：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    M_{n}&=\beta_1M_{n-1}+\beta_3g_n\\
    \theta_n&=\theta_{n-1}-\eta M_n
\end{align*}
$$

（若不做特殊说明，$\nabla\mathcal{L({x};\theta_{n-1})}$ 一概指一个 mini-batch 的平均梯度即 $\dfrac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\nabla\mathcal{L}(x_i;\theta_{n-1})$）

这样就得到了**动量法随机梯度下降**即 SGD with Momentum 或 SGDM 算法了。式子里面的 $\beta_1$ 指的是动量衰减因子，可以理解成某种摩擦阻力，要不然就会一直在极小值周围做（近似）的椭圆天体运动不收敛， $\beta_3$ 是梯度的参考系数，而 $\eta$ 就是学习率了。

#### Nesterov 加速

如果把刚刚 SGDM 的式子展开：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    \theta_n&=\theta_{n-1}-\eta (\beta_1M_{n-1}+\beta_3g_n)\\
    &=\theta_{n-1}-\eta \beta_1M_{n-1}-\eta\beta_3g_n
\end{align*}
$$

可以看到其实我们对参数进行了**两步**更新，而第二步更新使用的梯度却是更新前参数的梯度。如果我们考虑让第二步更新使用的梯度是**第一步更新后参数的梯度**，也就是让 $g'_n=\nabla\mathcal{L}({x};\theta_{n-1}-\eta \beta_1M_{n-1})$，就得到了 Nesterov 加速优化后的 SGDM。

虽然网上99%对这个的讲解都是停留在这里就完了，但是我们很难对 $\theta_{n-1}-\eta \beta_1M_{n-1}$ 这个前瞻位置的参数直接求一次梯度。为什么？因为我们最后要在代码里面使用 `loss.backward()` 把梯度求出来，但是这个梯度不是前瞻位置的梯度，这不就废了吗，说好的加速，计算量反倒翻倍了……

为此，我们需要寻找无需进行前瞻位置梯度计算的等效形式。

##### 进阶推导

为简便起见，下面的推导统一设 $\beta_3=1$。

让我们从

$$
\theta_n=\theta_{n-1}-\eta \beta_1M_{n-1}-\eta g'_n
$$

入手（其中 $g'_n=\nabla\mathcal{L}({x};\theta_{n-1}-\eta \beta_1M_{n-1})$），为了让这个 $g'_n$ 能够较好地被分离出去，首先在左右两边配上一个 $-\eta \beta_1M_n$，有点类似于去找前瞻位置。然后：

$$
\begin{align*}
    \theta_n-\eta\beta_1M_n&=\theta_{n-1}-\eta(1+\beta_1)M_n\\
    &=\theta_{n-1}-\eta(1+\beta_1)(\beta_1M_{n-1}+g'_n)\\
    &=(\theta_{n-1}-\eta\beta_1M_{n-1})-\eta[(1+\beta_1)g'_n+\beta_1^2M_{n-1}]
\end{align*}
$$

我们可以做一个代换：

$$
\begin{align*}
    \hat\theta_n&=\theta_n-\eta\beta_1M_n\\
    \hat M_n &= (1+\beta_1)g'_n+\beta_1^2M_{n-1}
\end{align*}
$$

这样就有了

$$
\hat\theta_n=\hat\theta_{n-1}-\eta\hat M_n
$$

并且循环带入 $M_n$ 的定义式展开 $\hat M_n$：

$$
\begin{align*}
    \hat M_n &= (1+\beta_1)g'_n+\beta_1^2M_{n-1}\\
    &=(1+\beta_1)g'_n+\beta_1^2g'_{n-1}+\beta_1^3g'_{n-2}+\cdots
\end{align*}
$$

利用高中就学过的错位相减（让 $\hat M_n$ 和  $\beta\hat M_{n-1}$ 相减），我们可以得到 $\hat M_n$ 的递推式：

$$
\hat M_n = \beta_1\hat M_{n-1}+g'_n+\beta_1(g'_n-g'_{n-1})
$$

整理一下我们得到递推公式：

$$
\begin{align*}
    g'_n&=\nabla\mathcal{L({x};\hat\theta_{n-1})}\\
    \hat M_{n}&=\beta_1\hat M_{n-1}+g'_n+\beta_1(g'_n-g'_{n-1})\\
    \hat\theta_n&=\hat\theta_{n-1}-\eta\hat M_n
\end{align*}
$$

由于初始时的 Nesterov 修正项是 $0$，这个递推式可以保证与原来的形式完全等效。

但是我们可以发现，即使这个 $g_n$ 取到原来的梯度，也能通过这种方式（两次迭代的梯度之差）来得到 Nesterov 加速等效的结果。

但是这样做要我们保存两份梯度，有没有更省显存的做法呢？有的。

还是一个和高中数列题很像的思路，我们把在 $\hat M_n$ 的计算中长得比较像的拉到一边去：

$$
\begin{align*}
    \hat M_n - g'_n &= \beta_1 g'_n + \beta_1(\hat M_{n-1} - g'_{n-1})\\
    \dfrac{\hat M_n - g'_n}{\beta_1}&=g'_n + (\hat M_{n-1} - g'_{n-1})
\end{align*}
$$

这里有一个分母 $\beta_1$ 不好看，我们做代换 $\beta_1\tilde M_n = \hat M_n - g'_n$，就有：

$$
\tilde M_n=g'_n+\beta_1\tilde M_{n-1}
$$

整理一下我们得到递推公式：

$$
\begin{align*}
    g'_n&=\nabla\mathcal{L({x};\hat\theta_{n-1})}\\
    \tilde M_n&=g'_n+\beta_1\tilde M_{n-1}\\
    \hat\theta_n&=\hat\theta_{n-1}-\eta(\beta_1\tilde M_n+ g'_n)
\end{align*}
$$

由于我们在整个变换过程中只是使用了变量代换，和一开始的 Nesterov 加速法是等效的，所以对于这个式子而言，我们完全可以这样写：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    M_n&=g_n+\beta_1M_{n-1}\\
    \theta_n&=\theta_{n-1}-\eta(\beta_1 M_n+ g_n)
\end{align*}
$$

这样对 Nesterov 加速的实现就非常简单了！只需要把权重更新项从 $\eta M_n$ 换成 $\eta(\beta_1 M_n+ g_n)$ 即可。

当然有的实现会考虑 $\beta_3=(1-\beta_1)$，这个时候权重更新项就变成了 $\eta[\beta_1 M_n+ (1-\beta_1)g_n]$。

#### 评述

SGDM 能够具有更快的收敛速率，尤其对于梯度不对称场景下，能够实现均衡的梯度累积，即减缓前后横跳，加速向下滚动。动量居功至伟。

### 正则化优化

我们考虑一般的 $L_2$ 正则化用以对权重大小进行惩罚限制。在 SGD 场景下：

$$
\begin{align*}
    g_{n} &= -\eta\nabla\left(\mathcal{L}({x};\theta_{n-1})+\dfrac{\lambda}{2}|\theta_{n-1}|^2\right)\\
    &=-\eta\nabla\mathcal{L}({x};\theta_{n-1})-\eta\lambda\theta_{n-1}\\
    \theta_n&=\theta_{n-1}+g_n\\
    &=(1-\eta\lambda)\theta_{n-1}-\eta\nabla\mathcal{L}(x;\theta_{n-1})
\end{align*}
$$

这样我们就可以以数乘代替繁琐且耗时的梯度计算，这被叫做“解耦的权重衰减”（Decoupled Weight Decay）。在后面的优化器中（比如 AdamW），我们基本不会直接使用原教旨主义的 $L_2$ 正则化，而是采用这种权重衰减的方式，尽管在更复杂的优化器下，这两者数学上并不等效。

### SGDM 的代码实现

下面，让我们来赏析一下 `torch.optim.SGD` 对 SGDM 的实现。中文注释是我让 Gemini 帮我读代码给出的注解。

<details>

<summary>SGDM 的实现</summary>

```python
def _single_tensor_sgd(
    params: list[Tensor],
    grads: list[Tensor],
    momentum_buffer_list: list[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    # 这两个参数与自动混合精度（AMP）中的梯度缩放有关，用于跳过无效更新。
    # 此函数是基础实现，不处理这些情况，故断言它们为 None。
    assert grad_scale is None and found_inf is None

    # 循环遍历每一个参数及其对应的梯度和动量缓冲区
    for i, param in enumerate(params):
        # 获取当前参数的梯度。如果目标是最大化（maximize=True），则反转梯度方向。
        grad = grads[i] if not maximize else -grads[i]
        
        # --- 步骤 1: 应用权重衰减 (Weight Decay / L2 正则化) ---
        if weight_decay != 0:
            # 注意: 这里的实现是将权重衰减项加到梯度上，而不是从权重中直接减去（解耦权重衰减）。
            # 这等价于在损失函数中加入了 L2 正则化项 0.5 * weight_decay * param^2。
            # 更新前的梯度 g' = g + w * theta
            
            # 使用嵌套 if 是为了绕过 TorchScript JIT 编译器的规则，并处理可微的 weight_decay。
            if isinstance(weight_decay, Tensor):
                if weight_decay.requires_grad:
                    # 如果 weight_decay 本身是需要计算梯度的张量（例如在元学习中），
                    # 必须克隆 param 来进行乘法，以避免在反向传播中出现原地修改错误。
                    grad = grad.addcmul(param.clone(), weight_decay)
                else:
                    # 如果 weight_decay 是张量但无需梯度，则使用常规的 add 操作。
                    grad = grad.add(param, alpha=weight_decay)
            else:
                # 如果 weight_decay 是一个普通的浮点数，这是最常见的情况。
                grad = grad.add(param, alpha=weight_decay)

        # --- 步骤 2: 计算动量并更新梯度 ---
        if momentum != 0:
            # 获取当前参数的动量缓冲区（momentum buffer），我们称之为 M
            buf = momentum_buffer_list[i]

            if buf is None:
                # 如果是第一次更新该参数，动量缓冲区为空。
                # 初始化动量 M_0 = g' (当前梯度)
                # 使用 clone().detach() 来创建一个不带梯度历史的新张量作为初始动量。
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                # 如果已有动量，则进行更新。
                # 公式: M_t = momentum * M_{t-1} + (1 - dampening) * g'_t
                # momentum 是动量因子 (β)，dampening 抑制了新梯度对动量的影响。
                # 当 dampening=0 时，这就是标准动量更新公式 M_t = β * M_{t-1} + g'_t
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                # 如果使用 Nesterov 加速梯度 (NAG):
                # 更新所用的梯度变为: g''_t = g'_t + momentum * M_t
                grad = grad.add(buf, alpha=momentum)
            else:
                # 如果使用标准动量:
                # 更新所用的梯度就是动量本身: g''_t = M_t
                grad = buf
        
        # --- 步骤 3: 使用最终的梯度更新参数 ---
        # 最终更新公式: param_t = param_{t-1} - lr * g''_t
        
        # 同样，使用嵌套 if 来处理 lr 可能是一个需要梯度的张量的情况。
        if isinstance(lr, Tensor):
            if lr.requires_grad:
                # 如果 lr 需要梯度，必须使用 addcmul，它有为所有输入（包括lr）定义的导数。
                # value=-1 实现减法。
                param.addcmul_(grad, lr, value=-1)
            else:
                # 如果 lr 是张量但无需梯度，使用 add_ 和 alpha=-lr。
                param.add_(grad, alpha=-lr)
        else:
            # 如果 lr 是一个普通的浮点数（最常见情况），使用最高效的 add_ 操作。
            param.add_(grad, alpha=-lr)
```

</details>

### 其他讨论

#### 梯度与海森矩阵估计

我们回顾一下怎么给一个多元函数 $f({x})$ 做二阶泰勒展开。

在一阶的情况下，很明显有

$$
f({x})\approx f({x_0})+\nabla f({x_0})\cdot ({x-x_0})
$$

类似的，我们可以把求导算子应用到梯度上面，也就是得到梯度的雅可比矩阵，或者叫做**海森矩阵**：

$$
H=\left(\frac{\partial^2}{\partial x_i\partial x_j}\right)_{1\le i,j\le n}=\left(
\begin{matrix}
    \frac{\partial^2}{\partial x_1^2} & \frac{\partial^2}{\partial x_1\partial x_2} & \cdots & \frac{\partial^2}{\partial x_1\partial x_n}\\
    \frac{\partial^2}{\partial x_2\partial x_1} & \frac{\partial^2}{\partial x_2^2} & \cdots & \frac{\partial^2}{\partial x_2\partial x_n}\\
    \vdots & \vdots & \ddots & \vdots\\
    \frac{\partial^2}{\partial x_n\partial x_1} & \frac{\partial^2}{\partial x_n\partial x_2} & \cdots & \frac{\partial^2}{\partial x_n^2}\\
\end{matrix}
\right)
$$

那么二阶的泰勒展开就变成了

$$
f({x})\approx f({x_0})+\nabla f({x_0})({x-x_0})+\dfrac{1}{2}({x-x_0})^\top H f({x_0})({x-x_0})
$$

让我们用熟悉的记号重写一下：

$$
\mathcal{L}(\theta_n+\Delta\theta)\approx \mathcal{L}(\theta_n)+g_n^\top\Delta\theta+\dfrac{1}{2}\Delta\theta^\top H \mathcal{L}(\theta_n)\Delta\theta
$$

在 SGD 场景下，$\Delta\theta=-\eta g_n$，损失函数的改变量为：

$$
\mathcal{L}(\theta_n+\Delta\theta)- \mathcal{L}(\theta_n)\approx -\eta g_n^\top g_n+\dfrac{1}{2}\eta^2g_n^\top H \mathcal{L}(\theta_n)g_n
$$

可见，在 $H$ 较大（尤其是条件数较大，对应之前说的椭圆抛物面）的地方，SGD 的更新是相当低效的。

让我们以另外一个视角观察前面的式子。为了找到最小值，事实上我们是在寻找损失函数**梯度的零点**。这样我们可以挪用数值分析里面找零点相当高效的算法：牛顿迭代法。

回顾一下牛顿法的内容，对于函数 $f(x)$ 和初始估计 $x_{n-1}$，更新方式为：

$$
\Delta x = -\dfrac{f(x_{n-1})}{f'(x_{n-1})}\\
x_n=x_{n-1}+\Delta x
$$

牛顿法的特征就是收敛高效，且是**自适应**的，陡峭的地方下降快，平缓的地方精度高。

在优化器的语境里面，其实就是让参数更新量 $\Delta \theta=-[H\mathcal{L}]^{-1}\nabla \mathcal{L}$。而前面所述的“陡峭”和“平缓”恰巧对应的就是一阶导（梯度）的导数也就是**海森矩阵**！

由于海森矩阵（及其逆矩阵）的计算量非常之大，与其计算它不如用这个时间跑几轮算得快的近似算法。

而 SGD 其实就是取的 $[H\mathcal{L}]^{-1}=\eta I$ 这个估计，相当于抹平各个方向的差异做了统一的更新。

对于海森矩阵而言，有没有更好的估计方式呢？有的！要不然为什么我们会引入动量呢？

让我们考虑一个椭圆抛物面

$$
f({x})=\dfrac 12{x}^\top A{x}+{b}^\top {x}
$$

那么在这个面上的任一点有梯度 $\nabla f=A{x}-{b}$ 以及海森矩阵 $Hf=A$。

令梯度等于 $0$，实质就是求解线性方程组 $H{x}={b}$，为此我们定义残差 ${r}=H{x}-{b}$

在任意一本数值分析教材里面都会讲到求解线性方程组的一百万种方法，包括高斯消元法，雅可比迭代，预条件法等。在其中和先前我们提到的动量法最相关的是**共轭梯度法**。下面简要介绍一下这个方法。

**定义**：若 $A$ 为 $n\times n$ 的对称正定矩阵，$u,v$ 为两个 $n$ 维的向量，则两者的 **$A$-内积** 定义为：

$$
\langle u,v\rangle _A=u^\top Av
$$

特别的，如果 $\langle u, v\rangle _A=0$，则称两个向量关于 $A$ 共轭。

现在我们有一个参数的初始估计 $\theta_n$，得到了梯度 $g_n$ 也就是之前提到的残差。

下面我们需要寻找参数更新量 $M_n$，吸纳之前关于“共轭”的讨论，我们希望各个 $M$ 之间的优化是独立的，也就是 $\langle M_i,M_j\rangle _H=0\quad(i\ne j)$ 成立。

感觉是否很像 Gram-Schmidt 正交化？是的！当然共轭梯度法使用的是一个等效更简单的形式（具体怎么从正交化等式推到这个简单形式比较复杂而且也有点跑题了，我视情况在后面给一个附录进行证明），也就是

$$
\beta_n = \dfrac{g_n^\top g_n}{g_{n-1}^\top g_{n-1}}\\
M_n=g_n+\beta_nM_{n-1}
$$

其实直接用 Gram-Schmidt 正交化也无可厚非（虽然这就涉及到要把 $H$ 纳入计算），因为无论如何我们都要把 $\beta_n$ 估计为一个固定的超参数 $\beta$，我们要的是下面那个动量式子的结构而不是参数的表达式，毕竟参数可以估计。

当然共轭梯度还利用 $\alpha_i=\dfrac{g_n^\top g_n}{M_n^\top H M_n}$ 来计算更新步长，不过这个的计算意义不大，因为我们一是不知道 $H$，二是利用的固定学习率 $\eta$ 来近似替代。

现在我们把 $\beta_i$ 也估计为固定的超参数 $\beta$，那我们就可以下论断了：**动量法随机梯度下降是采用固定参数近似对 $H^{-1}$ 的共轭梯度法求解**。

对于海森矩阵而言，有没有更好的估计方式呢？有的！让我们不要把参数固定死，来一个自适应调节。

## 自适应学习率改进策略

我没要求你一定得用那种最新最好的 Optimizer，我不是恶魔。

可是，用 SGD 优化 LLM 是什么意思？你才 21 岁吧？再这样下去，你 21 岁用 SGD，42 岁用 SGD with Momentum，84 岁就该问 Who is Adam 了。

作为 $\theta$，我可能真该收敛到鞍点，真的。

### AdaGrad

#### 外积近似

AdaGrad 的精髓是拿梯度近似海森矩阵 $H$，以此实现自适应调整。但是这需要我们对损失地形有更多的探索。

这一部分，我主要是参考 arXiv:2304.09871 和[苏剑林的这篇博客](https://kexue.fm/archives/10588)的内容来推导。

在目标参数 $\theta$ 附近取近似解 $\theta_n$，我们可以把梯度做一个一阶近似：$g_n=H(\theta-\theta_n)$

近似解可以随机选取，因此考虑其服从 $N(\theta, \sigma^2 I)$，为了弄出平方我们把它乘上自己的转置：

$$
g_ng_n^\top=H(\theta-\theta_n)(\theta-\theta_n)^\top H^\top
$$

事实上一开始 AdaGrad 就是考虑的采用的这种外积方案，但是计算量过大，我们考虑只取 $H$ 的对角元（这个在 SGD 中已经有效地使用过一次了），并且在期望意义下 $(\theta-\theta_n)(\theta-\theta_n)^\top=E=\sigma^2I$，这样就可以写成

$$
H\approx\dfrac{1}{\sigma}\sqrt{g_n\odot g_n}
$$

由此，便可以祭出 AdaGrad 大法了：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    G_{n}&=G_{n-1}+g_n\odot g_n\\
    \theta_n&=\theta_{n-1}-\dfrac{\eta}{\sqrt{\epsilon+G_n}} g_n
\end{align*}
$$

为了防止除零错误，$\epsilon$ 是一个小正数。在实践上也会有把 $\epsilon$ 提到根号外的情况，都是等价的。

#### AdaGrad 的代码实现

同样让我们看看 `PyTorch` 对这个算法的实现。

<details>

<summary>AdaGrad 的实现</summary>

```python
# _get_value(t: Tensor) -> float: 从单元素张量中提取其浮点数值，类似于 t.item()
# _make_sparse(grad, grad_indices, grad_values): 使用给定的索引和值创建一个新的稀疏张量

def _single_tensor_adagrad(
    params: list[Tensor],
    grads: list[Tensor],
    state_sums: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    # 这两个参数与自动混合精度（AMP）的梯度缩放有关，此特定实现不支持，因此断言它们为 None
    assert grad_scale is None and found_inf is None

    # 使用 zip 同时遍历参数、梯度、状态累加和、步数这四个列表
    for param, grad, state_sum, step_t in zip(params, grads, state_sums, state_steps):
        # 更新步数计数器（原地操作）
        step_t += 1
        # 从 Tensor 中获取步数的标量值（例如通过 .item()）
        step = _get_value(step_t)
        # 如果是最大化问题（maximize=True），则反转梯度方向，执行梯度上升
        grad = grad if not maximize else -grad

        # 应用权重衰减（L2 正则化）
        if weight_decay != 0:
            # Adagrad 的权重衰减与稀疏梯度不兼容，因为 add 操作在稀疏张量上定义不同
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients" # 权重衰减选项与稀疏梯度不兼容
                )
            # 对于稠密梯度，将权重衰减项加到梯度上。公式: grad = grad + param * weight_decay
            grad = grad.add(param, alpha=weight_decay)

        # 根据学习率衰减公式，计算当前步骤的有效学习率 (clr)
        # 公式: clr = lr / (1 + (step - 1) * lr_decay)
        clr = lr / (1 + (step - 1) * lr_decay)

        # 根据梯度是稀疏还是稠密，选择不同的更新路径
        if grad.is_sparse:
            # --- 稀疏梯度更新路径 ---
            # 合并稀疏梯度中相同索引的值，确保索引唯一。这对于后续的非线性操作（如平方）是必需的。
            grad = grad.coalesce()
            grad_indices = grad._indices()  # 获取稀疏梯度的非零元素索引
            grad_values = grad._values()   # 获取稀疏梯度的非零元素值

            # 将当前梯度值的平方，以稀疏张量的形式，累加到历史状态 `state_sum` 中
            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            # 从 `state_sum` 中仅抽取出与当前梯度非零位置相对应的累积值
            std = state_sum.sparse_mask(grad)
            # 计算分母：对抽出的累积值开方，然后加上 eps 以保证数值稳定性
            std_values = std._values().sqrt_().add_(eps)
            # 更新参数：仅更新梯度中非零索引对应的参数元素
            # 更新公式：param[indices] -= clr * (grad_values / std_values)
            param.add_(
                _make_sparse(grad, grad_indices, grad_values / std_values), alpha=-clr
            )
        else:
            # --- 稠密梯度更新路径 ---
            # 检查参数是否为复数类型
            is_complex = torch.is_complex(param)
            if is_complex:
                # 如果是复数，则将其视为一个实数张量进行后续计算，其形状会增加一个维度2（实部和虚部）
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            
            # Adagrad 核心步骤：将梯度的平方累加到 state_sum 中（原地操作）
            # 公式: state_sum = state_sum + grad * grad
            state_sum.addcmul_(grad, grad, value=1)
            
            # 计算分母 std = sqrt(state_sum) + eps
            if differentiable:
                # 如果要求整个优化过程可微分，则使用返回新张量的 `+` 操作
                std = state_sum.sqrt() + eps
            else:
                # 否则，使用原地操作 `add_` 以节省内存并可能提高速度
                std = state_sum.sqrt().add_(eps)
            
            # 执行参数更新（原地操作）
            # 公式: param = param - clr * (grad / std)
            param.addcdiv_(grad, std, value=-clr)

            # 如果参数是复数，需要将作为实数视图的变量转换回其复数表示
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)
```

</details>

AdaGrad 通过累积的 $G$ 来实现对 Hessian 的近似，**按理说**应该具有更加优秀的学习率调度。毕竟，AdaGrad 就是 Adaptive Gradient 的省略嘛！

但是事实上我们可以发现，如果在一个并不好的，梯度很大的初始位置开始进行优化，那累积在 $G_n$ 里面的梯度将会是“一辈子都抹不去的东西”，$G_n$ 的值只会越来越大，即使走出这样的地方，仍然会因为这个“历史包袱”而寸步难行。尤其是刚刚的近似只是对靠近最优点能够很有效，有没有办法从梯度能够获得对 Hessian 矩阵的更好估计呢？这就要祭出 RMSprop 了。

### RMSprop

其实我们想要的是一种“窗口平均”，因为 $H\approx\dfrac{1}{\sigma}\sqrt{g_n\odot g_n}$ 是在接近最优点的统计意义下近似的，如果离最优点比较远，那参数更新量大一些也无妨，离最优点比较近，就不要让之前的结果影响到。

这种窗口平均肯定不能直接保存最近 $k$ 个梯度的列表再求平均，这显然太费显存：

$$
G_{n+1}=\frac 1k (kG_n+p_n-p_{n-k})
$$

其中 $p_n = g_n\odot g_n$。不过我们可以把 $p_{n-k}$ 近似成 $G_n$，也就是使用平均值来近似单一值，然后做一个变量替换 $\beta_2=\dfrac{k-1}{k}$ 来使式子好看，这样我们相比于 AdaGrad，就不用增加任何临时存储了！由此得到的是**滑动窗口平均**，即：

$$
G_{n+1} = \beta_2 G_n + (1-\beta_2)g_n\odot g_n
$$

这种平均是不是似曾相识？回想起之前关于动量法的讨论（取 $\beta_3=(1-\beta_1)$ ）：

$$
M_n=(1-\beta_1)g_n+\beta_1M_{n-1}
$$

看，这里的动量计算其实也是在取梯度的滑动窗口平均。

这就得到了 RMSprop 算法了：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    \theta_n&=\theta_{n-1}-\dfrac{\eta}{\sqrt{\epsilon+G_n}} g_n
\end{align*}
$$

RMS 指的就是 $\sqrt{\epsilon+G_n}$，既有滑动窗口的平方平均 (Mean Square)，又在最后开了根(Root)。

prop的意思就是反向传播了。毕竟我们是对神经网络做的优化。

#### RMSprop 的代码实现

<details>

<summary>RMSprop 的实现</summary>

```python
def _single_tensor_rmsprop(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    grad_avgs: list[Tensor],
    momentum_buffer_list: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
    # 循环遍历每一个参数及其对应的梯度和状态
    for i, param in enumerate(params):
        # 获取当前参数的更新步数
        step = state_steps[i]

        # --- CUDA Graph 捕获相关的检查 ---
        # 如果代码正在被 torch.compile 编译，编译器会处理图捕获的检查。
        # 见 note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            # 获取支持 CUDA Graph 捕获的设备列表（通常是 'cuda'）
            capturable_supported_devices = _get_capturable_supported_devices()
            # 断言：如果启用了 capturable，参数和其状态必须在支持的设备上
            assert (
                param.device.type == step.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        # 获取当前参数的梯度
        grad = grads[i]
        # 如果是最大化问题 (maximize=True)，则反转梯度方向（梯度上升）
        grad = grad if not maximize else -grad
        # 获取当前参数的梯度平方的移动平均值
        square_avg = square_avgs[i]

        # 步数加 1
        step += 1

        # --- 权重衰减 (Weight Decay) ---
        # 如果设置了权重衰减（L2 正则化）
        if weight_decay != 0:
            # 将权重衰减项加到梯度上。公式: grad = grad + param * weight_decay
            # 也就是解耦的权重衰减。
            grad = grad.add(param, alpha=weight_decay)

        # --- 处理复数张量 ---
        # 检查参数是否为复数类型
        is_complex_param = torch.is_complex(param)
        if is_complex_param:
            # 如果是复数，将其视为实数张量进行处理。
            # 例如，一个形状为 [N] 的复数张量会变成形状为 [N, 2] 的实数张量，
            # 最后一维分别代表实部和虚部。
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            square_avg = torch.view_as_real(square_avg)

        # --- 更新梯度平方的移动平均值 (RMS) ---
        # 公式: square_avg = alpha * square_avg + (1 - alpha) * grad^2
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        # --- 计算分母 `avg` ---
        if centered:
            # --- Centered RMSprop ---
            # 获取梯度的移动平均值
            grad_avg = grad_avgs[i]
            if is_complex_param:
                # 同样处理复数情况
                grad_avg = torch.view_as_real(grad_avg)
            
            # 更新梯度的移动平均值。公式: grad_avg = alpha * grad_avg + (1 - alpha) * grad
            grad_avg.lerp_(grad, 1 - alpha)
            
            # 计算分母。公式: avg = sqrt(square_avg - grad_avg^2)
            # 这实际上是梯度的（移动）方差的平方根
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_()
        else:
            # --- 标准 RMSprop ---
            # 计算分母。公式: avg = sqrt(square_avg)
            avg = square_avg.sqrt()

        # --- 添加 epsilon 以保证数值稳定性 ---
        if differentiable:
            # 如果要求操作可微分，使用 `add` (返回新张量) 而不是 `add_` (原地修改)
            avg = avg.add(eps)
        else:
            # 否则，使用原地操作 `add_` 以提高效率，防止分母为零
            avg = avg.add_(eps)

        # --- 参数更新步骤 ---
        if momentum > 0:
            # --- 带动量的更新 ---
            # 获取动量缓冲
            buf = momentum_buffer_list[i]
            if is_complex_param:
                 # 同样处理复数情况
                buf = torch.view_as_real(buf)
            
            # 更新动量缓冲。公式: buf = momentum * buf + grad / avg
            buf.mul_(momentum).addcdiv_(grad, avg)
            # 使用动量缓冲更新参数。公式: param = param - lr * buf
            param.add_(buf, alpha=-lr)
        else:
            # --- 不带动量的标准更新 ---
            # 直接更新参数。公式: param = param - lr * (grad / avg)
            param.addcdiv_(grad, avg, value=-lr)
```

</details>

代码里面提到了 Centered RMSprop，其实还是为了解决“不在最小值周围”的问题。因为我们在最小值周围选点，梯度的期望是 $0$，但是如果不在周围，梯度的期望就要另行计算，怎么计算呢？和之前的思路一样，同步对梯度做滑动窗口平均即可，然后计算 $G_n$ 的适合，减去这个期望平方值，就相当于做了一次中心化了。

代码里面还提到了“动量缓冲”，可以这样理解：RMSprop 是自适应学习率的 SGD，那么我们用相同的方式给 SGDM 添加自适应学习率，就得到了 RMSprop with Momentum 了，具体实现参考刚刚的代码，其实就是使用动量项 $M_n = \beta_3 M_{n-1} + \dfrac{\eta}{\sqrt{\epsilon+G_n}} g_n$ 再乘以学习率作为参数更新量。

#### AdaDelta

让我们回到在 AdaGrad 里面讨论的海森矩阵近似：

$$
H\approx\dfrac{1}{\sigma}\sqrt{g_n\odot g_n}
$$

在 RMSprop 中，我们能够高效计算 $\sqrt{g_n\odot g_n}$，而对于 $\sigma$，我们直接用学习率估计的，但考虑到 $\sigma$ 自身的意义（也就是 $\mathbb{E}[(\theta_n-\theta)(\theta_n-\theta)^\top]$ 即参数离最优解的期望欧几里得距离），如果当前预期参数比较远，$\sigma$ 就该比较大，反之则较小。怎么估计这个距离呢？AdaDelta 提出的方案是使用**参数更新量的滑动窗口平均**。也就是：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    X_n&=\beta_4X_{n-1}+(1-\beta_4)\Delta\theta_{n-1}\odot\Delta\theta_{n-1}\\
    \theta_n&=\theta_{n-1}-\dfrac{\sqrt{\epsilon+X_n}}{\sqrt{\epsilon+G_n}} g_n
\end{align*}
$$

可以看到 AdaDelta 已经完全实现了自适应调节，连学习率的估计都实现了自动化调整。

下面是 AdaDelta 的代码实现：

<details>

<summary>AdaDelta 的实现</summary>

```python
def _single_tensor_adadelta(
    params: list[Tensor],
    grads: list[Tensor],
    square_avgs: list[Tensor],
    acc_deltas: list[Tensor],
    state_steps: list[Tensor], # 注意：此函数中 state_steps 仅被递增，但未在核心算法中使用
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
    # --- CUDA Graph 捕获相关的检查 ---
    # 如果代码正在被 torch.compile 编译，编译器会处理图捕获的检查。
    if not torch.compiler.is_compiling() and capturable:
        # 获取支持 CUDA Graph 捕获的设备列表
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        # 断言：如果启用 capturable，所有参数和状态都必须在支持的设备上
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"如果 capturable=True, params 和 state_steps 必须在支持的设备上: {capturable_supported_devices}."

    # 循环遍历每一个参数及其对应的梯度和状态
    for param, grad, square_avg, acc_delta, step in zip(
        params, grads, square_avgs, acc_deltas, state_steps
    ):
        # 步数加 1 (在 Adadelta 核心算法中未使用，但为保持优化器接口一致性而保留)
        step += 1
        # 如果是最大化问题，则反转梯度
        grad = grad if not maximize else -grad

        # --- 应用权重衰减 ---
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # --- 处理复数张量 ---
        if torch.is_complex(param):
            # 将所有状态和梯度都视为实数张量进行计算
            square_avg = torch.view_as_real(square_avg)
            acc_delta = torch.view_as_real(acc_delta)
            grad = torch.view_as_real(grad)

        # --- Adadelta 算法核心步骤 ---
        
        # 1. 更新梯度平方的移动平均值 E[g^2]_t
        # 公式: E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        
        # 2. 计算梯度的均方根 RMS[g]_t
        # 公式: RMS[g]_t = sqrt(E[g^2]_t + eps)
        std = square_avg.add(eps).sqrt_()
        
        # 3. 计算上一步参数更新量的均方根 RMS[Δx]_{t-1}
        # 公式: RMS[Δx]_{t-1} = sqrt(E[Δx^2]_{t-1} + eps)
        # 这里的 acc_delta 存储的是 E[Δx^2]_{t-1}
        delta = acc_delta.add(eps).sqrt_()

        # 为了可微性，如果需要，克隆 delta，以防后续的原地操作破坏计算图
        if differentiable:
            delta = delta.clone()
            
        # 4. 计算当前的参数更新量 Δx_t
        # 公式: Δx_t = (RMS[Δx]_{t-1} / RMS[g]_t) * g_t
        # delta.div_(std) 对应 -> / RMS[g]_t
        # .mul_(grad)    对应 -> * g_t
        # 此时，`delta` 变量存储的是计算出的更新量 Δx_t
        delta.div_(std).mul_(grad)
        
        # 5. 更新参数更新量平方的移动平均值 E[Δx^2]_t，为下一步做准备
        # 公式: E[Δx^2]_t = rho * E[Δx^2]_{t-1} + (1 - rho) * (Δx_t)^2
        # acc_delta 此时仍是 E[Δx^2]_{t-1}
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        # --- 应用最终更新 ---
        
        # 如果是复数，将计算出的实数更新量转换回复杂的视图
        if torch.is_complex(param):
            delta = torch.view_as_complex(delta)
        
        # 6. 更新参数
        # 公式: x_{t+1} = x_t - lr * Δx_t
        # PyTorch 的实现中保留了 lr 作为缩放系数，默认为 1
        param.add_(delta, alpha=-lr)
```

</details>

我们已经在动量加速和自适应学习率两条道路上走了很远了，那么，有没有一种方法，能够无缝融合，真正集这两家武功之大成呢？有的，这就是接下来要讨论的 Adam 优化器，也就是目前最广泛使用的一个优化器。

### Adam

我们已经知道，通过滑动窗口平均梯度的平方，可以得到学习率的一个自适应调整；通过引入动量，可以让我们有更快的收敛速率。如果我们将自适应学习率调整融入动量法之中，Adam 优化器就自然而然地诞生了。

具体来说，Adam 优化器是这样计算的：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    M_n&=(1-\beta_1)g_n+\beta_1M_{n-1}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    \hat M_n&=\dfrac{M_n}{1-\beta_1^{n-1}}\\
    \hat G_n&=\dfrac{G_n}{1-\beta_2^{n-1}}\\
    \theta_n&=\theta_{n-1}-\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}} \hat M_n
\end{align*}
$$

可以看到，$M_n$ 和 $G_n$ 的计算与先前的优化器并无二致，自适应学习率调整也和 RMSprop 一样。但是 Adam 还额外做了一个**随步数衰减**的缩放，这是因为迭代初期时没有填满滑动窗口导致 $M_n$ 和 $G_n$ 事实上偏小，所以需要这个 $\dfrac{1}{1-\beta^n}$ 来补偿。

等着看代码吗？别急，Adam 优化器在提出之后，也是经历了如过山车一般起伏的波折，现在的 Adam 实现早就不是原来那个 Adam 了。

何以见得？且听下回分解。

### Adam 的变体们

#### AMSGrad

人怕出名猪怕壮， Adam 自宣布自己拥有 SOTA 级别的收敛效果后，便遭到了许多批评，其中许多不无道理。第一个扔过来的炸弹是收敛性问题，在 [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ) 这篇文章里，作者认为学习率倒数的差分即

$$
\Gamma_n = \dfrac{\sqrt{G_n}}{\eta}-\dfrac{\sqrt{G_{n+1}}}{\eta}
$$

由于滑动平均的缘故，没法做到像 SGD 和 AdaGrad 一样，让它恒为正。这意味着学习率虽然自适应调整了，但是一会调大一会调小，在这反复横跳，哪来的收敛？？？

不过存在一个简单粗暴的 clip 方案来解决这个问题。既然你嫌弃学习率一会大，一会小，而造成这个出现变动的核心原因就是 $G_n$ 不单调递增，那我直接让 $G_n$ 取目前所有 $G$ 的最大值，也就是只有出现新的最大值才更新 $G_n$，不就完美解决了嘛！

也就是说相对 Adam，AMSGrad 只做了一点小修改：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    M_n&=(1-\beta_1)g_n+\beta_1M_{n-1}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    \hat M_n&=\dfrac{M_n}{1-\beta_1^n}\\
    \hat G_n&=\max\{\hat G_{n-1},G_n\}\\
    \theta_n&=\theta_{n-1}-\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}} M_n
\end{align*}
$$

也就相当于把 Adam 对 $G_n$ 的修偏估计换成了取极大值，这样不仅解决了嫌 $G_n$ 偏小的问题，还解决了学习率反复横跳的问题，可谓一石二鸟。

#### AdamW

不过一波未平一波又起，在 arXiv:1711.05101v1 这篇文章里面，作者揭露了 Adam 优化器和 $L_2$ 正则化一同使用时出现的问题。

让我们回顾一下怎么在 SGD 上面做权重衰减：

$$
\begin{align*}
    g_{n} &= -\eta\nabla\left(\mathcal{L}({x};\theta_{n-1})+\dfrac{\lambda}{2}|\theta_{n-1}|^2\right)\\
    &=-\eta\nabla\mathcal{L}({x};\theta_{n-1})-\eta\lambda\theta_{n-1}\\
    \theta_n&=\theta_{n-1}+g_n\\
    &=(1-\eta\lambda)\theta_{n-1}-\eta\nabla\mathcal{L}(x;\theta_{n-1})
\end{align*}
$$

在 SGD 中，将 $L_2$ 正则化项的梯度（即 $\lambda\theta_{n-1}$）加到损失梯度上，与最后对权重进行乘性衰减（即乘以 $(1-\eta\lambda)$）是等效的。然而，在 Adam 这样的自适应学习率优化器中，这种等效性被打破了。

当时几乎所有的深度学习框架，在实现 Adam 的权重衰减时，都采用了将 $L_2$ 正则项的梯度加到 $\nabla\mathcal{L}$ 上的方式。这意味着，权重衰减项 $\lambda\theta_{n-1}$ 也会被 Adam 的自适应学习率 $\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}}$ 所缩放。这会产生一个意想不到的后果：对于那些历史梯度很大（即 $G_n$ 很大）的权重，它们获得的权重衰减效果会变小；而对于那些不经常更新、历史梯度很小（即 $G_n$ 很小）的权重，它们的权重衰减效果反而更强。这与我们使用权重衰减的初衷——对所有的大权重进行同等惩罚——是相悖的。

AdamW 的提出就是为了解决这个问题。它的核心思想是解耦权重衰减（Decoupled Weight Decay）。它不再将权重衰减伪装成 $L_2$ 正则化并加入梯度计算，而是将其从梯度更新中分离出来，直接在参数更新的最后一步实现，就像在 SGD 中那样。

这样我们就得到了 AdamW 即带有**解耦权重衰减**的 Adam 优化器：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    M_n&=(1-\beta_1)g_n+\beta_1M_{n-1}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    \hat M_n&=\dfrac{M_n}{1-\beta_1^n}\\
    \hat G_n&=\dfrac{G_n}{1-\beta_2^n}\\
    \theta_n&=\theta_{n-1}-\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}} M_n-\eta\lambda\theta_{n-1}
\end{align*}
$$

至此，AdamW 大杀四方，现在已经成为transformer训练中的默认优化器了。

讲了这么多，让我们一窥代码真容：

<details>

<summary> Adam, AMSGrad, AdamW 的实现</summary>

```python
def _single_tensor_adam(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],             # 一阶矩估计（动量） m_t
    exp_avg_sqs: list[Tensor],          # 二阶矩估计（自适应学习率项） v_t
    max_exp_avg_sqs: list[Tensor],      # AMSGrad 用的历史最大二阶矩
    state_steps: list[Tensor],          # 步数 t
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,                      # 是否启用 AMSGrad
    has_complex: bool,
    beta1: Union[float, Tensor],        # 一阶矩的指数衰减率
    beta2: Union[float, Tensor],        # 二阶矩的指数衰减率
    lr: Union[float, Tensor],           # 学习率
    weight_decay: float,                # 权重衰减系数
    eps: float,                         # 防止除以零的极小值
    maximize: bool,
    capturable: bool,                   # 是否支持 CUDA Graph 捕获
    differentiable: bool,               # 是否要求操作可微分
    decoupled_weight_decay: bool,       # 是否使用 AdamW 的解耦权重衰减
):
    assert grad_scale is None and found_inf is None

    # 如果在 TorchScript (JIT) 环境下，由于 JIT 对类型推断的限制，直接断言超参数为 float
    if torch.jit.is_scripting():
        assert isinstance(lr, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)

    # 为了优化，如果 beta1 是 Tensor，预先将其按设备和类型存入字典，避免循环内重复转换
    if isinstance(beta1, Tensor):
        beta1_dict: Optional[DeviceDtypeDict] = {(beta1.device, beta1.dtype): beta1}
    else:
        beta1_dict = None

    # 循环处理每个参数
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # --- CUDA Graph 捕获检查 ---
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        # 步数加 1
        step_t += 1

        # --- 步骤 1: 应用权重衰减 ---
        if weight_decay != 0:
            if decoupled_weight_decay:
                # AdamW: 解耦权重衰减。直接在参数上乘以一个衰减因子。
                # 公式: param_t = param_t * (1 - lr * weight_decay)
                param.mul_(1 - lr * weight_decay)
            else:
                # 标准 Adam: 权重衰减作为 L2 正则化项加入梯度。
                # 公式: grad_t = grad_t + weight_decay * param_{t-1}
                # 嵌套 if 是为了处理可微分和 JIT 的情况
                if differentiable and isinstance(weight_decay, Tensor):
                    if weight_decay.requires_grad:
                        grad = grad.addcmul(param.clone(), weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

        # --- 处理复数 ---
        if torch.is_complex(param):
            # 将所有相关张量都视为实数进行计算
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        device = param.device

        # 如果 beta1 是 Tensor，从字典中获取对应设备和类型的版本
        if beta1_dict is not None:
            dtype = param.dtype
            key = (device, dtype)
            if key not in beta1_dict:
                beta1_dict[key] = beta1.to(device=device, dtype=dtype, non_blocking=True)
            device_beta1: Union[float, Tensor] = beta1_dict[key]
        else:
            device_beta1 = beta1

        # --- 步骤 2: 更新一阶和二阶矩估计 ---
        # 更新一阶矩估计 m_t (exp_avg)
        # 公式: m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t
        exp_avg.lerp_(grad, 1 - device_beta1)

        # 更新二阶矩估计 v_t (exp_avg_sq)
        # 公式: v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t^2
        # 同样，嵌套 if 是为了处理可微分情况
        if differentiable and isinstance(beta2, Tensor):
            if beta2.requires_grad:
                # 使用 lerp 实现可微分的更新，数学上等价于下面的 addcmul
                exp_avg_sq.lerp_(torch.square(grad), weight=1 - beta2)
            else:
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # --- 步骤 3: 参数更新 ---
        # capturable 或 differentiable 模式下，所有计算都使用张量操作以保留计算图
        if capturable or differentiable:
            step = step_t

            # --- 计算偏差修正项 ---
            # 嵌套 if 用于处理 beta 是可微张量的情况
            if differentiable and isinstance(beta1, Tensor):
                if beta1.requires_grad:
                    bias_correction1 = 1 - beta1 ** step.clone()
                else:
                    bias_correction1 = 1 - beta1**step
            else:
                bias_correction1 = 1 - beta1**step

            if differentiable and isinstance(beta2, Tensor):
                if beta2.requires_grad:
                    bias_correction2 = 1 - beta2 ** step.clone()
                else:
                    bias_correction2 = 1 - beta2**step
            else:
                bias_correction2 = 1 - beta2**step

            # --- 计算更新步长和分母 ---
            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()
            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # AMSGrad: 维护历史最大二阶矩
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]
                
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sqs[i])
                
                # 使用最大二阶矩计算分母
                # 这里做了一些数学变换，将 step_size 合并计算，以减少张量读写
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                # 标准 Adam: 使用当前二阶矩计算分母
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            # 执行最终更新
            if differentiable:
                param.addcdiv_(exp_avg.clone(), denom)
            else:
                param.addcdiv_(exp_avg, denom)
        
        # 非 capturable/differentiable 的常规路径（效率更高）
        else:
            step = _get_value(step_t)

            # --- 计算偏差修正项 ---
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # --- 计算步长和分母 ---
            step_size = lr / bias_correction1
            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # AMSGrad: 更新并使用历史最大二阶矩
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                # 标准 Adam: 使用当前二阶矩
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            # --- 执行最终更新 ---
            # 公式: param_t = param_{t-1} - step_size * (m_hat / (sqrt(v_hat) + eps))
            param.addcdiv_(exp_avg, denom, value=-step_size)

        # --- 复数转换回来 ---
        # 如果启用了 AMSGrad 并且参数是复数，将 max_exp_avg_sqs 视图转换回来
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])
```

</details>

</details>

#### Adamax

先前提到 Adam 由于无法控制 $G_n$ 的单调性而可能陷入无法收敛的状况，并且也介绍了 AMSGrad 提出的 clip 方案。而 Adamax 却提出了一个有所不同的 clip 方案。

Adamax 的思想，最初是想把 $G_n$ 对平均梯度的 $L_2$ 估计（也就是 $g_n\odot g_n$ 项）扩展到 $L_p$ 估计：

$$
G_{n}=\beta_2 G_n + (1-\beta_2)g_n^p\\
\theta_n=\theta_{n-1}-\dfrac{\eta}{G_n^{\frac 1p}} M_n
$$

我们单独把学习率自适应权重 $G_n^{\frac 1p}$ 提取出来展开算：

$$
\begin{align*}
    G_n^{\frac 1p} &= \beta_2 G_n + (1-\beta_2)g_n^p\\
    &=(1-\beta_2)^{\frac 1p}\left(\sum_{i=1}^{n}\beta_2^i g^p_{n-i}\right)^{\frac 1p}
\end{align*}
$$

显然这种推广在任意的 $p$ 下是无法解决任何问题的，但是如果我们让 $p\rightarrow\infty$ 也就是取 $L_\infty$ 范数，就会得到：

$$
\begin{align*}
    \lim_{p\rightarrow\infty}(1-\beta_2)^{\frac 1p}\left(\sum_{i=1}^{n}\beta_2^i g^p_{n-i}\right)^{\frac 1p}&=\lim_{p\rightarrow\infty}\left(\sum_{i=1}^{n}\beta_2^i g^p_{n-i}\right)^{\frac 1p}\\
    &=\max\left\{\beta_2^i |g_{n-i}|\right\}_{i=1\dots n}
\end{align*}
$$

写成递推式子就是

$$
G_n = \max\{\beta_2G_{n-1}, |g_n|\}
$$

因此 Adamax 宣称自己相对 Adam，能够解决不收敛问题，还可以简省计算量。不过这样魔改，真的能对 Hessian 做更好的估计吗……

#### Nadam

读到这里，我相信任何一位读者都可以独立发明出 Nadam，毕竟我们在讲 SGDM 的时候花了大力气推导了 Nesterov 加速的式子，总不可能到了 Adam 这一块就完全不管了吧。是的，Nadam 的 novelty 就在于把 Nesterov 加速梯度引入到了 Adam 的计算之中。
我们考虑直接引入 Nesterov 加速项，也就是权重更新项从 $\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}}\hat M_n$ 换成 $\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}}[\dfrac{\beta_1 M_n}{1-\beta_1^{n-1}}+\dfrac{(1-\beta_1) g_n}{1-\beta_1^{n-1}}]$。这个也是 arXiv/1609.04747 推导出来的的。

但是考虑到 Adam 增加了对梯度二阶矩的估计，因此如果一直使用固定的 $\beta_1$ 的话，其实是偏大的。如果我们看提出 Nadam 的原论文 [Incorporating Nesterov Momentum into Adam](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)，就可以发现它的思路有一定的差异。

在后面一篇论文中，作者并没有固定规定一个 $\beta_1$，而是使用 $\mu_{n}$ 来调整更新比例，也就是它认为动量和权重的更新应该是如下的：

$$
\begin{align*}
    M_n&=\mu_{n}M_{n-1}+\beta_3 g_n\\
    \theta_n&=\theta_{n-1}-(\mu_{n+1}M_n+\beta_3 g_n)
\end{align*}
$$

这里 $\mu_{n+1}$ 代表的就是 Nesterov 加速的前瞻性。下面我们取 $\beta_3=(1-\mu_n)$，由于是对 $\mu_n$ 进行连乘，权重更新项也就变成了 $\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}}[\dfrac{\mu_{n+1} M_n}{1-\prod_{i=1}^{n+1}\mu_i}+\dfrac{(1-\mu_{n}) g_n}{1-\prod_{i=1}^{n}\mu_i}]$

为了解决之前偏大的问题，PyTorch 在 Nadam 的实现里对 $\beta_1$ 采用了衰减的策略。

具体而言，它引入了 $\mu_n=\beta_1 \left(1 - 0.5 \cdot 0.96^{n \cdot d}\right)$ 的估计，那么最后权重的更新方式变成：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    M_n&=(1-\beta_1)g_n+\beta_1M_{n-1}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    \mu_n&=\beta_1 \left(1 - 0.5 \cdot 0.96^{n \cdot d}\right)\\
    \mu_{n+1}&=\beta_1 \left(1 - 0.5 \cdot 0.96^{(n+1) \cdot d}\right)\\
    \hat \mu_{n+1} &=\hat\mu_n \mu_{n+1}\\
    \hat M_n&=\dfrac{\mu_{n+1}M_n}{1-\hat \mu_{n+1}}\\
    \hat G_n&=\dfrac{G_n}{1-\beta_2^{n-1}}\\
    \theta_n&=\theta_{n-1}-\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}} (\hat M_n+\dfrac{(1-\mu_n)g_n}{1-\hat\mu_n})
\end{align*}
$$

### Shampoo

让我们回顾那个在最优点附近的 Hessian 近似： $H\approx\dfrac{1}{\sigma^2} \sqrt{GG^\top}$，Shampoo 的思想是选取更精确的近似以逼进 $GG^\top$。

提一嘴，这里的 $GG^\top$ 指的是将 $g$ 展平之后的外积，也就是 $\text{vec}(g)\text{vec}(g)^\top$，鉴于之前我们一直研究的都是 $g$ 的对角线乘积近似，由于简化很多所以没有特别明确这个维度问题，因此在这里明确一下。

Shampoo 优化器的第一步，是考虑现在的多层神经网络内，层之间是相互独立的。因此可以把大型的 $GG^\top$ 给分块对角化，每一个对角块对应某个层的梯度外积。

但是即使这样，单层的参数量也很大，考虑一个 $n\times m$ 的 fc layer，$GG^\top$ 的参数量就来到了 $(mn)\times(nm)$，直接平方。而这就带领我们进入 Shampoo 优化器推导的真正精妙之处。

作者认为，海森矩阵可以由两个小矩阵的 Kronecker 积近似，也就是

$$
H=(GG^\top)^{-1}=L\otimes R
$$

这样一拆开参数量暴降到 $L$ 的 $n^2$ 加上 $R$ 的 $m^2$。可以理解成 $L$ 捕获输入维的信息，$R$ 捕获输出为的信息（不过我觉得有点强行解释了哈哈，因为关键是节省计算量，看一路过来我们都是在寻求尽可能**高效**而不是最有道理的优化器）。

下面推导 $L$ 和 $R$ 的更新式。先介绍 Kronecker 积的几个小性质：$\text{vec}(BXA^\top)=(A\otimes B)\text{vec}(X)$ 和 $(A\otimes B)^{-1}=(A^{-1}\otimes B^{-1})$ （转置亦然）。

取 $B=A=G$， $X = I$ 那么 $\text{vec}(GG^\top)=(G\otimes G)\text{vec}(I)$ 但这和我们期待的结构仍有距离，不过我们可以换成未展平的原矩阵也就是利用：

$$
(g\otimes g)(g\otimes g)^\top=(g\otimes g)(g^\top\otimes g^\top)=gg^\top \otimes g^\top g
$$

来近似 $H$。

上述推导非常类似 K-FAC 算法，具体请参考[这个博客](https://blog.csdn.net/xbinworld/article/details/105184601)。

无论如何根据我们之前的经验，这里的 $L$ 和 $R$ 也必然是要取滑动平均的，也就是

$$
L_n = \beta L_{n-1} + g_n g_n^\top\\
R_n = \beta R_{n-1} + g_n^\top g_n
$$

那么我们对参数进行更新，就是计算 $H^{-1}g$，展开，并利用 Kronecker 积的性质，得到

$$
(L\otimes R)^{-1}g=L^{-1}gR^{-1}
$$

事实上 Shampoo 优化器并未采用求逆的方案，因为时间开销较大。作者采用的方案是引入一个新参数 $p$ 来计算逆 $p$ 次方根。默认 $p=4$，也就是遵循下面的更新策略：


$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x};\theta_{n-1})}\\
    L_n &= \beta L_{n-1} + g_n g_n^\top\\
    R_n &= \beta R_{n-1} + g_n^\top g_n\\
    P_{L,t}&=L_n^{-\frac 14}\\
    P_{R,t}&=R_n^{-\frac 14}\\
    \tilde{G}_n &= P_{L,t} g_n P_{R,t}\\
    \theta_{n} &= \theta_{n-1} - \eta \tilde{G}_n
\end{align*}
$$

这里下标出现了 $t$ 是因为考虑到取逆 $p$ 次根的复杂性，我们不必每一轮迭代都去计算这两个预条件子 $P_{L,t}$ 和 $P_{R,t}$ 而是可以选择在多轮周期之后再更新。

## 符号梯度下降

如果是近似 Hessian 矩阵是优化器理论发展的一条“明线”，那么对梯度取“符号”来计算，则是对应的一条“暗线”。

在接下来的讨论中，我们将看到刚刚讨论的那些优化器是如何在这条“暗线”下走向统一的。同时，这条“暗线”也渐渐越挑越明，逐渐成为大规模神经网络训练优化的新的理论指导。

### Rprop

Rprop 的出现早于 RMSprop，从命名风格就可以看出它们的一脉相承。

回忆一下 RMSprop 的计算，它提供了一个梯度缩放系数 $\sqrt{G_n}$，其中 $G$ 是对 $g^2$ 的平均。

那么最后的参数更新就变成了 $-\eta\dfrac{g}{\sqrt{\bar g^2}}$，如果我们考虑全量（Full batch）更新，也就是让 $\mathcal{|B|}=n$ 即 Batch size 等于样本数量，那么我们甚至可以把这个“平均梯度”的平均去掉。这样实际的更新量就是梯度的**符号函数** $\text{sign}(g)$ 了！

这就是 Rprop 的更新原理。也就是所有符号梯度下降优化器的理论核心：梯度的**方向**相比其在不同方向的**大小**更重要！

回到我们之前讨论的那个椭圆抛物面，如果我们过于依赖梯度大小，就会造成“反复横跳”的问题，因为梯度在我们预期的优化方向的**垂直**方向上，有相当大的大小，这就影响了优化器在 Hessian 矩阵条件数相当大的时候，逃离鞍点的能力。

让我们来看看 Rprop 的更新公式：

$$
\begin{align*}
    g_n&=\nabla\mathcal{L({x_{\text{full}}};\theta_{n-1})}\\
    \hat g_n&=\text{sign}(g_n)\\
    \theta_n&=\theta_{n-1}-\eta\hat g_n
\end{align*}
$$

### Lion

### Muon

## 非梯度参数优化

### L-BFGS

### 模拟退火

### 遗传算法

## 超参数优化

### 网格搜索

### 贝叶斯搜索

### Optuna 框架

## 优化器的逆向思考（对抗攻击）
