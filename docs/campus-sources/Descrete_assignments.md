# 《离散数学》课程作业与笔记归档

供存档和复习用。

## 第一周作业

![alt text](image-15.png)

记 2 + 2 = 4 为 p，3 + 3 = 6 为 q。

- (1) $p\rightarrow q$, 真
- (2) $p\rightarrow \neg q$，假
- (3) $\neg p\rightarrow q$，真
- (4) $\neg p\rightarrow \neg q$，真
- (5) $p\leftrightarrow q$，真
- (6) $p\leftrightarrow \neg q$，假
- (7) $\neg p\leftrightarrow q$，假
- (8) $\neg p\leftrightarrow \neg q$，真

其实就是考察两个联结词的真值表。

![alt text](image-16.png)

- (1) 记 “2 是偶数” 为 p，记 “2 是素数” 为 q，则命题符号化为 $p\land q$
- (2) 记 “小王聪明” 为 p，记 “小王用功” 为 q，则命题符号化为 $p\land q$
- (3) 记 “天气很冷” 为 p，记 “老王来了” 为 q，则命题符号化为 $p\land q$
- (4) 记 “他吃饭” 为 p，记 “他看电视” 为 q，则命题符号化为 $p\land q$
- (5) 记 “下大雨” 为 p，记 “他乘公共汽车上班” 为 q，则命题符号化为 $p\rightarrow q$
- (6) 记 “下大雨” 为 p，记 “他乘公共汽车上班” 为 q，则命题符号化为 $q\rightarrow p$
- (7) 记 “下大雨” 为 p，记 “他乘公共汽车上班” 为 q，则命题符号化为 $q\rightarrow p$
- (8) 记 “经一事” 为 p，记 “长一智” 为 q，则命题符号化为 $q\rightarrow p$

不……不 = 出发……否则不 = 只有……才

![alt text](image-17.png)

- (1) $p\lor (q\land r)=0\lor (0\land 1)=0$
- (2) $(p\leftrightarrow r)\land (\neg q\lor s)=(0\leftrightarrow 1)\land (\neg q\lor s)=0$ 这里可以直接利用 $\land$ 的短路特性
- (3) $(p\land (q\lor r))\rightarrow ((p\lor q)\land(r\land s))=(0\land (0\lor 1))\rightarrow ((0\lor 0)\land(1\land 1))=1$ 同样是利用蕴含符号左边 $\land$ 的短路特性得到那一坨是 $0$ 然后再利用蕴含符号本身的短路特性得到结果
- (4) $\neg (p\lor (q\rightarrow (r\land \neg p)))\rightarrow (r\lor \neg s)=\neg (0\lor (0\rightarrow (r\land \neg p)))\rightarrow (1\lor \neg 1)=1\rightarrow 1=1$

![alt text](image-18.png)

- (1) $\neg((p \land q) \to p)=\neg (\neg (p \land q)\lor p)=(p \land q)\land \neg p=q\land( p\land\neg p)=0$ 为矛盾式。
- (2) $((p \to q) \land (q \to p)) \leftrightarrow (p \leftrightarrow q)=(p \leftrightarrow q)\leftrightarrow(p \leftrightarrow q)=1$ 为重言式。
- (3) $(\neg p \to q) \to (q \to \neg p)=(\neg(\neg p) \lor q)\to (\neg q \lor \neg p)=(p \lor q)\to \neg(q \land p)=\neg(p \lor q)\lor \neg(q \land p)=\neg(p \lor q\land q \land p)=\neg(p\land q)$ 为可满足式。