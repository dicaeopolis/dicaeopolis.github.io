# 前馈神经网络的反向传播

本文为《医学神经网络与机器学习》的第二次作业。主要是推导 FFN 的 BP 算法，或者说类似于手动构建计算图，来感受梯度流动的过程。

## 目标

从总体上说，对于一个神经网络的参数 $\theta$，我们使用优化器进行参数更新：

$$
\theta_{\mathrm{new}}=\mathrm{Optimizer}[\theta_{\mathrm{old}},\nabla_{\theta}\mathcal{L}(x,y,\theta)]
$$

不管我们使用小批量随机梯度下降：

$$
\theta_{\mathrm{new}} = \theta_{\mathrm{old}}-\dfrac{\eta}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\nabla_{\theta}\mathcal{L}(x,y,\theta)
$$

还是 Adam 优化器：

$$
\begin{align*}
    g_n&=\dfrac{1}{|\mathcal{B}|}\sum^{|\mathcal{B}|}_{i=1}\nabla_{\theta}\mathcal{L}(x,y,\theta)\\
    M_n&=(1-\beta_1)g_n+\beta_1M_{n-1}\\
    G_{n}&=\beta_2 G_n + (1-\beta_2)g_n\odot g_n\\
    \hat M_n&=\dfrac{M_n}{1-\beta_1^{n}}\\
    \hat G_n&=\dfrac{G_n}{1-\beta_2^{n}}\\
    \theta_{\mathrm{new}} &= \theta_{\mathrm{old}}-\dfrac{\eta}{\sqrt{\epsilon+\hat G_n}} \hat M_n
\end{align*}
$$

我们都要具体地对每一层的权重 $W^{(l)}$ 和偏置 $b^{(l)}$ 计算梯度（或者说微分）来进行更新：

$$
\begin{align*}
    W^{(l)} &\leftarrow \mathrm{Optimizer}[W^{(l)},\dfrac{\partial \mathcal{L}}{\partial W^{(l)}}]\\
    b^{(l)} &\leftarrow \mathrm{Optimizer}[b^{(l)},\dfrac{\partial \mathcal{L}}{\partial b^{(l)}}]
\end{align*}
$$

这样就自然引出了我们的计算目标：

$$
\begin{cases}
    \mathrm{损失对每一层权重的偏导：}\dfrac{\partial \mathcal{L}}{\partial W^{(l)}}\\
    &\quad \\
    \mathrm{损失对每一层偏置的偏导：}\dfrac{\partial \mathcal{L}}{\partial b^{(l)}}
\end{cases}
$$

## 前向过程

对于一个前馈神经网络而言，每一层的计算方式如下：首先拿到输入 $a^{(l)}$ 也就是上一层的输出 $o^{(l-1)}$ 或者原始输入 $x$，然后计算线性变换：

$$
z^{(l)}=W^{(l)}a^{(l)}+b^{(l)}
$$

最后通过该层的激活函数 $\phi_l$ 得到输出，具体而言现在所有激活函数都是标量激活函数，利用广播机制得到的输出：

$$
o^{(l)}=\phi_l(z^{(l)})
$$

一个 $L$ 层的神经网络的最后一层输出 $o^{(L)}$ 被送入损失函数 $\mathcal{L}$ 中，用来计算输入经过这个前馈神经网络的损失值。

## 反向传播

由于最后一层输出直接连接损失值，我们从这一层开始计算，首先是解决激活函数前后的部分：

$$
\delta_L =\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}=\dfrac{\partial\mathcal{L}}{\partial o^{(L)}}\dfrac{\partial o^{(L)}}{\partial z^{(L)}}
$$

由于 $o^{(L)}$ 直接作为损失的输入，因此第一项就是损失对这个向量的梯度：

$$
\dfrac{\partial\mathcal{L}}{\partial o^{(L)}}=\nabla_{o^{(l)}}\mathcal{L}
$$

第二项是一个雅可比矩阵，但是注意到这个矩阵只有主对角线有值，因为激活函数并不会将 $z$ 的分量混合，也就是说我们是通过广播机制计算的激活 $o^{(l)}_i=\phi_L(z^{(l)}_i)$，自然 $\dfrac{\partial o^{(L)}_i}{\partial z^{(L)}_j}$ 对任何 $i\neq j$ 都是 $0$，而对于 $i=j$ 而言，其实就是对标量函数 $\phi_L$ 求导数。也就是说：

$$
\delta_L =\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}=\dfrac{\partial\mathcal{L}}{\partial o^{(L)}}\dfrac{\partial o^{(L)}}{\partial z^{(L)}}=\nabla_{o^{(L)}}\mathcal{L}\odot \phi_L'(z^{(L)})
$$

然后基于中间结果 $z$ 处理权重和偏置：

$$
\begin{align*}
    \dfrac{\partial\mathcal{L}}{\partial W^{(L)}}&=\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}\dfrac{\partial z^{(L)}}{\partial W^{(L)}}=\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}a^{(L)}=\delta_L a^{(L)}\\
    \dfrac{\partial\mathcal{L}}{\partial b^{(L)}}&=\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}\dfrac{\partial z^{(L)}}{\partial b^{(L)}}=\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}=\delta_L
\end{align*}
$$

最后，我们可以求出对 $a^{(L)}$ 的偏导，也就是对上一层输出 $o^{(L-1)}$ 的偏导：

$$
\dfrac{\partial\mathcal{L}}{\partial a^{(L)}}=\dfrac{\partial\mathcal{L}}{\partial z^{(L)}}\dfrac{\partial z^{(L)}}{\partial a^{(L)}}={W^{(L)}}^\top\delta_L$$

这里的 ${W^{(L)}}^\top$ 可以从雅可比矩阵直接得出，或者考虑类似标量微分式，但是注意乘法的形状对应，就可以导出这个式子。

注意到整个流程我们可以反向继续进行，也就是说我们求得了对 $a^{(l+1)}$ 的偏导，或者说对 $o^{(l)}$ 的偏导之后，我们就可以算出中间结果的偏导：

$$
\delta_l =\dfrac{\partial\mathcal{L}}{\partial z^{(l)}}=\dfrac{\partial\mathcal{L}}{\partial o^{(l)}}\dfrac{\partial o^{(l)}}{\partial z^{(l)}}=({W^{(l+1)}}^\top\delta_{l+1})\odot \phi_l'(z^{(l)})
$$

类似地，基于中间结果 $z$ 处理权重和偏置：

$$
\begin{align*}
    \dfrac{\partial\mathcal{l}}{\partial W^{(l)}}&=\dfrac{\partial\mathcal{l}}{\partial z^{(l)}}\dfrac{\partial z^{(l)}}{\partial W^{(l)}}=\dfrac{\partial\mathcal{l}}{\partial z^{(l)}}a^{(l)}=\delta_l a^{(l)}\\
    \dfrac{\partial\mathcal{l}}{\partial b^{(l)}}&=\dfrac{\partial\mathcal{l}}{\partial z^{(l)}}\dfrac{\partial z^{(l)}}{\partial b^{(l)}}=\dfrac{\partial\mathcal{l}}{\partial z^{(l)}}=\delta_l
\end{align*}
$$

这样就得到了整个反向传播的流程。