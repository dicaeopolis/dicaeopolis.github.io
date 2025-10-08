# 《概率论与数理统计》课程作业与笔记归档

供存档和复习用。

## 第一次作业

![alt text](image-11.png)

i. A, B 互不相容，即 AB 无交集，也就是 A 一定在 B 的补集里面。$P(A\bar B)=P(A)=1/2$

ii. 由 $P(A)=P(AB)+P(A\bar B)$ 可得 $P(A\bar B)=3/8$

![alt text](image-12.png)

11 取 7 排列，这个 7 字母词有两个 i 以及两个 b 可以互换都满足条件，所以

$$
P=\dfrac{2\times2}{A^7_{11}}=\dfrac{2\times2}{5\times6\times7\times8\times9\times10\times11}=\dfrac{1}{415800}
$$

![alt text](image-13.png)

我们分成四个区域

![alt text](image-14.png)

$$
P=\dfrac{B\cap (A\cup \bar B)}{A\cup \bar B}=\dfrac{\mathrm{II}}{\mathrm{I+II+IV}}=\dfrac{0.2}{0.5+0.2+0.1}=\dfrac{1}{4}
$$

## 第二次作业

![alt text](image-19.png)

(1)

第一次取分为两个步骤：选箱子+选零件。

$$
P(A)=\dfrac{1}{2}\times\dfrac{10}{50}+\dfrac{1}{2}\times\dfrac{18}{30}=\dfrac{2}{5}
$$

(2)

根据条件概率公式：

$$
P(B|A)=\dfrac{P(AB)}{P(A)}
$$

其中

$$
P(AB)=\dfrac{1}{2}\times\dfrac{10}{50}\times\dfrac{9}{49}+\dfrac{1}{2}\times\dfrac{18}{30}\times\dfrac{17}{29}=\dfrac{276}{1421}
$$

则

$$
P(B|A)=\dfrac{276}{1421}\times\dfrac{5}{2}=\dfrac{690}{1421}
$$

![alt text](image-20.png)

(1) 和 (2) 必然假。 $P(AB)=P(A)P(B)=0$ 与 $P(A)>0, P(B)>0$ 矛盾

(3) 必然假。 若 $B$ 与 $A$ 不相容则必然 $P(B)<1-P(A)$，这就矛盾。

(4) 可能对。考虑随机变量 $x$ 服从一个 $[0,1]$ 上的均匀分布。

成立的情形：

$$
P(A)=P(0<x<0.6)\\
P(B)=P(0.24<x<0.84)
$$

则 $P(AB)=P(A)P(B)=0.36$

不成立的情形：

$$
P(A)=P(0<x<0.6)\\
P(B)=P(0.4<x<1)
$$

则 $P(AB)=0.2\neq P(A)P(B)=0.36$

![alt text](image-21.png)

$$
P(B)=\underbrace{0.8\times(1-2\%)^3}_{P(BA_1)}+\underbrace{0.15\times(1-10\%)^3}_{P(BA_2)}+\underbrace{0.05\times(1-90\%)^3}_{P(BA_3)}=0.8623536\\
$$

拿 Python 算了一下数值解：

$$
\begin{align*}
    P(A_1|B)=\dfrac{P(BA_1)}{P(B)}&=0.8731378868250795\\
P(A_2|B)=\dfrac{P(BA_2)}{P(B)}&=0.12680413231880752\\
P(A_3|B)=\dfrac{P(BA_3)}{P(B)}&=0.00005798085611285204\\
\end{align*}
$$

什么是随机变量：如果一个变量 $x$ 在每一次观测时的值不能被完全先验地确定则称其为一个随机变量。

## 第三次作业

![alt text](image-42.png)

(1)

$$
X\sim Ge(p)\Leftrightarrow P(X=n)=p(1-p)^{n-1}
$$

也就是在前 $n-1$ 次尝试都失败了，最后一次成功就收手。

(2)

$$
P(X=n)=C_{n-1}^{r-1} p^r(1-p)^{n-r}
$$

最后一次尝试必须成功然后收手，前面的 $n-1$ 次尝试里面可以任意安排 $r-1$ 次成功尝试的位置。

(3)

和 (1) 一样平移一下的几何分布。

$$
P(X=n) = p(1-p)^n
$$

由此

$$
\begin{align*}
    P(x\in\mathrm{Even.})&=\sum_{i=1}^\infty P(X=2i)\\
    &=p[(1-p)^0+(1-p)^2+\cdots]\\
    &=p\times\dfrac{1}{1-(1-p)^2}\\
    &=\dfrac{1}{2-p}
\end{align*}
$$

带入即可得到值为 $0.6451612903225806$

![alt text](image-43.png)

题目是要让我们用泊松分布近似二项分布。出事故的车辆数 $X$ 满足：

$$
X\sim B(1000,0.0001)
$$

近似即

$$
X\sim Po(0.1)
$$

由泊松分布公式：$P(x=k) =\dfrac{\lambda^k\mathrm{e}^{-\lambda}}{k!}$ 可得：

$$
\begin{align*}
    P(X\ge 2)&=1-P(x=1)-P(x=0)\\
    &=1-\dfrac{0.1^1\exp(-0.1)}{1!}-\dfrac{0.1^0\exp(-0.1)}{0!}\\
    &=1-1.1\exp(-0.1)\\
    &=0.00467884016044440
\end{align*}
$$

![alt text](image-44.png)

(1)

根据定义：

$$
\begin{align*}
    P(x<2)&=F(2)=\ln 2\\
    P(0<x\le 3)&=F(3)-F(0)=1\\
    P(2<x<5/2)&=F(5/2)-F(2)=\ln 5-2\ln 2
\end{align*}
$$

(2)

直接求导即可：

$$
f(x)=
\begin{cases}
    0,&x<1,x\ge \mathrm{e}\\
    \dfrac{1}{x},&1\le x<\mathrm{e}\\
\end{cases}
$$

![alt text](image-45.png)

几何概型。成立的事件域为：

$$
\begin{align*}
    (4K)^2-4\times4\times(K+2)\ge 0&\iff K^2-K-2\ge 0\\
    &\iff (K-2)(K+1)\ge 0\\
    &\iff K\in (0,2)
\end{align*}
$$

因此 $P=2/5$
