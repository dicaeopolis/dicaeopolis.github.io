# 《数据结构》期末复习题题解

## 第五版第一套题目
### 单项选择题

1.**数据结构是指**______。  
   A. 一种数据类型  
   B. 数据的存储结构  
   C. 一组性质相同的数据元素的集合  
   D. 相互之间存在一种或多种特定关系的数据元素的集合  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：D。</b> 三短一长选最长好吧。其实数据结构的精妙之处就是利用好数据之间的相互关系来组织数据，以获得更佳的性能。ABC选项描述都没有抓住“数据间相互关系”这个关键词。

<p></p>

</details>

2.**以下算法的时间复杂度为**______。  
   ```c
   void fun(int n)
   {  int i = 1, s = 0;
      while (i <= n)
      {  s += i + 100; i++;  }
   }
   ```  
   A. $O(n)$  
   B. $O(\sqrt{n})$  
   C. $O(n\log_2 n)$  
   D. $O(\log_2 n)$  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：A。</b> 本题 s 是一个障眼法，决定循环次数的只有 i 和 n。根据代码，i 从 1 开始，每一次循环 i 都增加 1，直到 i 的值增加到 n 为止。因此一共执行了 n 次循环，选A。

<p></p>

<b>变体：如果把循环条件改成 `s <= n`，则应该选哪个选项呢？</b>

<details>
<summary>点击查看解答</summary>
<p></p>

考虑循环执行了 x 次，则
$$
s=\sum_{i=1}^{x}100+i = 100x+\dfrac{x(x-1)}{2}
$$
也就是 s 增长率在 x^2 量级，由于循环条件为 s<=n，就可以得出 x=O(√n)。
<p></p>
当然也可以利用上式把 x 关于 n 的精确表达式解出来再做分析，原理是一样的。
<p></p>
原书给的答案是B，其实想考的是这个，但是出题人有点草台班子，弄巧成拙了。

<p></p>
</details>

<p></p>

</details>

3.**在一个长度为$n$的有序顺序表中删除第一个元素值为$x$的元素时，在查找元素$x$时采用二分查找方法，此时删除算法的时间复杂度为**______。  
   A. $O(n)$  
   B. $O(n\log_2 n)$  
   C. $O(n^2)$  
   D. $O(\sqrt{n})$  


<details>

<summary> 答案 </summary>

<p></p>

<b>答案：A。</b> 这里删除算法分两步：第一步找到待删除的第一个元素，这一步由于原来的表是有序的，所以可以使用二分法，每次排除一半的区间，时间复杂度为 O(log_2 n)，第二步是把这个元素删除，由于采用线性表存储所以时间复杂度是 O(n)。两步加起来，由于第一步相比第二步用时太短，几乎可以忽略不计，所以总的时间复杂度为 O(n)。

<details>
<summary>为什么线性表随机删除单个元素的时间复杂度是 O(n) ?</summary>
<p></p>

随机删除元素，则任一下标 i (为方便书写，从 1 开始)的元素被选中的概率都是 1/n，删除操作需要把后面的 (n-i) 个元素都往前移动一位，因此移动次数的期望值为：

$$
E=\sum_1^{n}\dfrac {n-i}{n}= \dfrac{n-1}2
$$

是 O(n) 级别的操作。
<p></p>
延伸讨论：链表呢？有没有其他数据结构能够做到查找-删除操作比 O(n) 更快？
<p></p>
1. 对链表而言，虽然删除操作可以简单通过改变前后指针指向实现 O(1) 的复杂度，但是查找对应元素却需要 O(n) 的时间。
<p></p>
2. 当然有。比如各种平衡树。参考：https://oi-wiki.org/ds/bst/本质上平衡树就是为了减少这一操作的复杂度而利用了树形结构并且引入各种结构调整操作来限制树的深度，以防止出现最坏效率。当然，除了树形结构可以利用分治优化，链式结构也可以，比如跳表:https://oi-wiki.org/ds/skiplist/。

<p></p>
</details>

<p></p>
</details>

4.**若一个栈采用数组$s[0..n-1]$存放其元素，初始时栈顶指针为$n$，则以下元素$x$进栈的操作正确的是**______。  
   A. $\text{top}++;\ s[\text{top}] = x;$  
   B. $s[\text{top}] = x;\ \text{top}++;$  
   C. $\text{top}--;\ s[\text{top}] = x;$  
   D. $s[\text{top}] = x;\ \text{top}--;$  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：C。</b> 其实本题考虑到ABD选项都会造成数组下标越界访问就可以解出。但这个题本意是说从这个数组的尾端向前装数据，也能构成一个栈。这么做可以和一个从前往后装数据的栈组合成一个对顶栈。

<p></p>
</details>


5.**设环形队列中数组的下标为$0\sim N-1$，其队头、队尾指针分别为$\text{front}$和$\text{rear}$（$\text{front}$指向队列中队头元素的前一个位置，$\text{rear}$指向队尾元素的位置），则其元素个数为**______。  
   A. $\text{rear} - \text{front}$  
   B. $\text{rear} - \text{front} - 1$  
   C. $(\text{rear} - \text{front})\%N + 1$  
   D. $(\text{rear} - \text{front} + N)\%N$  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：D。</b> 因为这是一个循环队列，如果 rear 的下标大于 front，那么可以直接相减，但是也可能出现小于的情况，所以需要加上 N 来保证结果是正数，由于可能大于 N，所以要取模。

<p></p>
</details>

6.**若用一个大小为6的数组来实现环形队列，队头指针$\text{front}$指向队列中队头元素的前一个位置，队尾指针$\text{rear}$指向队尾元素的位置。若当前$\text{rear}$和$\text{front}$的值分别为0和3，当从队列中删除一个元素，再加入两个元素后，$\text{rear}$和$\text{front}$的值分别为**______。  
   A. 1和5  
   B. 2和4  
   C. 4和2  
   D. 5和1

<details>

<summary> 答案 </summary>

<p></p>


<b>答案：B。</b> 看图：
<p>
<a href="https://imgse.com/i/pEvSHCd"><img src="https://s21.ax1x.com/2025/05/16/pEvSHCd.png" alt="pEvSHCd.png" border="0" /></a></p>

<p></p>
</details>

7.**一棵高度为$h$（$h\geq1$）的完全二叉树至少有**______个结点。  
   A. $2^{h-1}$  
   B. $2^h$  
   C. $2^h + 1$  
   D. $2^{h-1} + 1$  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：A。</b>首先复习完全二叉树的定义：完全二叉树是除了最后一层以外其他层都满节点的二叉树，且最后一层的节点连续集中在最左边。题目问至少，所以我们让最后一层只有一个节点，那么总的节点数为：

$$
N=1+\sum_{i=0}^{h-2}2^i=2^{h-1}
$$

<p></p>
</details>



8.**设一棵哈夫曼树中有999个结点，该哈夫曼树用于对**______个字符进行编码。  
   A. 999  
   B. 499  
   C. 500  
   D. 501  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：C。</b>首先，哈夫曼树只有叶子节点可以表示数据的编码，因此我们其实就是要求出这个哈夫曼树叶子节点的个数。回忆构建哈夫曼树的过程，一开始有单独的一个叶子节点，然后每添加一个数据就新增一个叶子节点用来表示这个数据，并新增一个非叶子节点。因此叶子节点个数始终比非叶子节点数多 1 。利用这个性质就可算出答案是 500。

<p></p>
</details>


9.**一个含有$n$个顶点的无向连通图采用邻接矩阵存储，则该矩阵一定是**______。  
   A. 对称矩阵  
   B. 非对称矩阵  
   C. 稀疏矩阵  
   D. 稠密矩阵  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：A。</b>对于节点 i 和 j ，邻接矩阵里面的元素 G[i][j] 表示 i 到 j 的路径长度，同理元素 G[j][i] 表示 j 到 i 的路径长度。由于是一个无向图，所以两者应该相等。所以是对称矩阵。至于稀疏还是稠密，就取决于图本身了。

<p></p>
</details>


10.**设无向连通图有$n$个顶点、$e$条边，若满足**______，则图中一定有回路。  
    A. $e\geq n$  
    B. $e < n-1$  
    C. $e = n-1$  
    D. $2e\geq n$  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：A。</b>考虑一开始有 n 个独立的节点，从任意一个节点开始加边，我们尽量不要形成环，那么每一次加边都选择度为 0 的节点，加了 n - 1 次边之后我们发现没有度为 0 的节点了，也就是当 e = n - 1 时，连通图退化成树。此时从节点 i 到节点 j 再加一次边 e_n ，那么 i 到 j 原先的联通路径加上这个新加入的边刚刚好就构成一个环。

<p></p>
</details>


11.**如果从无向图的任一顶点出发进行一次广度优先遍历即可访问所有顶点，则该图一定是**______。  
    A. 完全图  
    B. 连通图  
    C. 有回路  
    D. 一棵树  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：B。</b>连通图是指图中任意两个顶点之间都存在路径相连的无向图。既然能够被搜索到，就意味着肯定有路径相连。完全图、有回路和树的定义更严格。题目中没有更多条件了，只能选到B。

<p></p>
</details>


12.**设有100个元素的有序表，在用折半查找时，不成功查找时最大的比较次数是**______。  
    A. 25  
    B. 50  
    C. 10  
    D. 7  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：D。</b>
折半搜索每次丢弃一半的区间，也就是：
$$
2^6=64\le 100\le 2^7=128
$$
找 6 次可能找不到，但是找 7 次一定找得到。
<p></p>
也可以这样理解：折半查找的时间复杂度是 O(log_2 n) 量级，只有D选项满足这个数量级。
<p></p>
</details>


13.**从100个元素确定的顺序表中查找某个元素（关键字为正整数），如果最多只进行5次元素之间的比较，则采用的查找方法只可能是**______。  
    A. 折半查找  
    B. 顺序查找  
    C. 哈希查找  
    D. 二叉排序树查找  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案 C。</b>

根据上面那个题，O(log_2 n)的算法的比较次数上界是7次，而这个算法比 O(log_2 n) 的比较次数还要少，也就意味着时间复杂度更优，而 AD 的时间复杂度都是 O(log_2 n)，B 是 O(n)，只有 C 的时间复杂度是常数级 O(1)。

<p></p>
</details>


14.**有一个含有$n$（$n>1000$）个元素的数据序列，某人采用了一种排序方法对其按关键字递增排序，该排序方法需要关键字比较，其平均时间复杂度接近最好的情况，空间复杂度为$O(1)$，该排序方法可能是**______。  
    A. 快速排序  
    B. 堆排序  
    C. 二路归并排序  
    D. 基数排序  

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：B。</b>本题其实给了三个限制条件来排除选项：
<p></p>
1. 需要关键字比较：排除基数排序，这个不需要关键字比较。
<p></p>
2. 平均时间复杂度接近最好的情况：排除快速排序，因为它的最坏时间复杂度是 O(n^2)。
<p></p>
3. 空间复杂度 O(1): 排除二路归并排序，因为它的空间复杂度是 O(n log_2 n)。
<p></p>
因此使用的是堆排序，堆排序是基于比较的原地算法，且最好、最坏和平均时间复杂度都是O(n log_2 n)。
<p></p>
</details>


15.**对一个线性序列进行排序，该序列采用单链表存储，最好采用**______方法。  
    A. 直接插入排序  
    B. 希尔排序  
    C. 快速排序  
    D. 都不适合

<details>

<summary> 答案 </summary>

<p></p>

<b>答案：A。</b>D选项拿来凑数的，别选。希尔排序需要进行数据分组，涉及到随机访问，不适合链表结构；快排需要进行多次随机交换，也不适合链表结构。只有插入操作对单链表的复杂度是 O(1) 的。

<p></p>
</details>

### 问答题
1.如果对含有 \( n(n>1) \) 个元素的线性表的运算只有4种：删除第一个元素；删除最后一个元素；在第一个元素前面插入新元素；在最后一个元素的后面插入新元素，则最好采用以下哪种存储结构，并简要说明理由。  
（1）只有尾结点指针没有头结点指针的循环单链表。  
（2）只有尾结点指针没有头结点指针的非循环双链表。  
（3）只有头结点指针没有尾结点指针的循环双链表。  
（4）既有头结点指针也有尾结点指针的循环单链表。  
<details>

<summary> 答案 </summary>
<p>(3)</p>
<p></p>
其实题目想让我们实现的数据结构叫做<b>双端队列</b>。因为插入和删除操作都集中在头尾，我们来分析一下这四个数据结构的时间复杂度：
<p></p>
(1) : 如果这个循环单链表只有尾节点，那么在进行插入和删除最后一个元素的时候，都要去寻找尾节点的前驱节点，但这又是一个单链表，没有前向信息，所以会浪费循环一次即 O(n) 的时间去找前驱。
<p></p>
(2) : 由于这个非循环的双链表没有头节点，在插入和删除第一个元素的时候都要横跨整个链表，也要花 O(n) 的时间。
<p></p>
(3) : 由于这是一个循环双链表，这时候相比于 (1) 和 (2) 而言，插入和删除都能够很方便（也就是 O(1) 时间）获取到头尾附近节点的地址，以进行修改。
<p></p>
(4) : 问题其实和 (1) 一样，即使有头节点也很难去找到尾节点的前驱节点。
<p></p>

我自己用带头尾指针的循环双链表实现了一个简单的双端队列，参考：https://dicaeopolis.github.io/stl-wheels/#deque

<p></p>
</details>

2.对于一个带权连通无向图 \( G \)，可以采用 Prim 算法构造出从某个顶点 \( v \) 出发的最小生成树，问该最小生成树是否一定包含从顶点 \( v \) 到其他所有顶点的最短路径。如果回答是，请予以证明；如果回答不是，请给出反例。 

<details>

<summary> 答案 </summary>

<p></p>
本题考察的是<b>最小（代价）生成树</b>和<b>最短路径树</b>的区别。
<p></p>
我们注意 Prim 算法在加边的时候优先选择<b>长度最短</b>的相邻边，但是 Dijkstra 算法的松弛操作选取的是<b>到源点距离最短</b>的相邻边。这很不一样。
<p></p>
构造反例时可以考虑构造一个树，然后将一个叶子节点和根节点相连，这条新边的权值小于原来根到这个叶子节点的路径长度即可，比如下面这个简单的反例：
<p></p>
<a href="https://imgse.com/i/pEvEBbn"><img src="https://s21.ax1x.com/2025/05/16/pEvEBbn.png" alt="pEvEBbn.png" border="0" /></a>
</details>

3.有一棵二叉排序树按先序遍历得到的序列为 \( (12, 5, 2, 8, 6, 10, 16, 15, 18, 20) \)。回答以下问题：
（1）画出该二叉排序树。  
（2）给出该二叉排序树的中序遍历序列。  
（3）求在等概率下的查找成功和不成功情况下的平均查找长度。  

<details>

<summary> 答案 </summary>


<p></p>
注意先序遍历是 ULR 的顺序，所以第一个节点一定是根节点，又因为 BST 的左子树都是小于根节点，右子树大于根节点，因此找到第一个大于根节点的数据就可以区分开左右子树了，如图：
<p></p>
<a href="https://imgse.com/i/pEvVDQe"><img src="https://s21.ax1x.com/2025/05/16/pEvVDQe.png" alt="pEvVDQe.png" border="0" /></a>
<p></p>
对于查找成功而言，找到每个节点的查找长度就是到根节点的路径长度加一（因为要算上根节点本身的查找）：

$$
L = \dfrac{1}{10}(1+2\times 2+3\times 4+4\times 3)=1.9
$$

对于查找不成功，考虑上图加上外部节点表示查找失败访问到的节点，如图：
<a href="https://imgse.com/i/pEvVWJf"><img src="https://s21.ax1x.com/2025/05/16/pEvVWJf.png" alt="pEvVWJf.png" border="0" /></a>

则长度等于：

$$
L = \dfrac{1}{11}(4\times 5+5\times 6)=\dfrac{30}{11}
$$

<p></p>

</details>


### 算法设计题  
1.（15分）假设二叉树 \( b \) 采用二叉链存储结构，设计一个算法 `void findparent(BTNode *b, ElemType x, BTNode *&p)` 求指定值为 \( x \) 的结点的双亲结点 \( p \)。提示：根结点的双亲为 `NULL`，若在二叉树 \( b \) 中未找到值为 \( x \) 的结点，\( p \) 也为 `NULL`。 

<details>

<summary> 答案 </summary>

<p></p>
思路是从根节点开始深搜，如果当前节点的子节点值为 x，那么该节点就是所求的节点 p。而且如果已经求得 p，那么剩下的搜索都可以剪枝跳过了。
<p></p>

```cpp
void findparent(BTNode *b, ElemType x, BTNode *&p)
{
   if(b == NULL || b->data == x)// 利用了或运算的短路性质，只有 b != NULL 才会访问 data，防止对空地址的解引用。
   {
      p = NULL;
      return ;
   }
   if( (b->lchild != NULL && b->lchild->data == x) || (b->rchild != NULL && b->rchild->data == x))
      p = b;
   else
   {
      findparent(b->lchild, x, p);
      if(p == NULL)
         findparent(b->rchild, x, p);
   }
}
```

<p></p>
</details>


2.（10分）假设一个有向图 \( G \) 采用邻接表存储，设计一个算法判断顶点 \( i \) 和顶点 \( j \)（\( i \neq j \)）之间是否相互连通，假设这两个顶点均存在。  

<details>

<summary> 答案 </summary>


<p></p>
任意使用一种搜索算法，如果从 i 出发能够搜索到 j，并且从 j 出发能够搜索到 i，那么就说明两者连通。为方便实现我这里使用深度优先搜索。
<p></p>

```cpp
bool vis[N];
std::array<std::vector<int>, N> G;
void dfs(int curr, const int& j, bool& tag)
{
   if(tag) return ;
   if(curr == j)
   {
      tag = true;
      return ;
   }
   vis[curr] = true;
   for(auto adj : G[curr])
      if(!vis[adj])
         dfs(adj, j, tag);
}
bool is_connected(const int& i, const int& j)
{
   std::fill(vis.begin(), vis.end(), 0);
   bool tag_i2j = false, tag_j2i = false;
   dfs(i, j, tag_i2j);
   if(tag_i2j)
   {
      std::fill(vis.begin(), vis.end(), 0);
      dfs(j, i, tag_j2i);
      if(tag_j2i) return true;
   }
   return false;
}
```
<p></p>
</details>


3.（15分）有一个含有 \( n \) 个整数的无序数据序列，所有的数据元素均不相同，采用整数数组 \( R[0..n-1] \) 存储，请完成以下任务：  
（1）设计一个尽可能高效的算法，输出该序列中第 \( k \)（\( 1 \leq k \leq n \)）小的元素，算法中给出适当的注释信息。提示：利用快速排序的思路。  
（2）分析你所设计的求解算法的平均时间复杂度，并给出求解过程。

**本题可以在洛谷上面做：https://www.luogu.com.cn/problem/P1923**

<details>

<summary> 答案 </summary>

<p></p>
我们考虑快速排序里面的划分操作，即：选取一个元素 e，然后把所有小于它的元素放在它的前边，大于它的元素放在它的后边，这样一个操作。那么此时，e 前面的数都小于 e，也就是说假设现在 e 的下标为 idx，则 e 就是第 (idx + 1) 小的数。如果 (idx + 1) < k，那么意味着第 k 小的数在 e 的后面，如果大于就是在前面，如果等于，直接返回 e 的值即可。这样每一次我们都排除一部分区间，就能递归求出 e 来。
<p></p>
代码:
<p></p>

```cpp
int partition(int pivot_index, int *data, int left, int right)
{
   int pivot = data[pivot_index], i = left, j = right;
   while(i < j)
   {
      while(i < j && data[j] >= pivot) --j;
      data[i] = data[j];
      while(i < j && data[i] <= pivot) ++i;
      data[j] = data[i];
   }
   data[i] = pivot;
   return i;
}
int kth(int *data, int left, int right, int k)
{
   if(left < right)
   {
      int pivot_index = partition(left, data, left, right);
      int n = pivot_index;
      if(n > k) return kth(data, left, pivot_index, k);
      if(n < k) return kth(data, pivot_index + 1, right, k);
      if(n == k) return data[pivot_index];
   }
   return data[left];
}
int solve_kth(int *data, int length, int k)
{
   return kth(data, 0, length - 1, k - 1);
}
```

<p></p>
对于长度为 m 的序列，进行一次划分需要的时间复杂度为 O(m)，同时问题规模变成 m / 2。

$$
\begin{equation*}
  \begin{aligned}
    T(n) &= O(n) + T(n / 2) \\
         &= O(n) + O(n / 2) + T(n / 4) \\
         &= O(n) + O(n / 2) + O(n / 4) + ... \\
         &= 2*O(n)\\
         &= O(n)
  \end{aligned}
\end{equation*}
$$
<p></p>

</details>

## 第五版第二套题目
### 单项选择题

1.以下数据结构中______属非线性结构。  
   A. 栈  
   B. 串  
   C. 队列  
   D. 平衡二叉树

<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D。</b>树不是线性结构。
<p></p>

</details>


2.以下算法的时间复杂度为______。  
```c
void func(int n)
{ 
    int i = 0, s = 0;
    while (s <= n)
    { 
        i++;
        s = s + i;
    }
}
```  
A. \( O(n) \)  
B. \( O(\sqrt{n}) \)  
C. \( O(n\log_2 n) \)  
D. \( O(\log_2 n) \)  

<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B。</b>参考第一套题的第二道单选题。
<p></p>

</details>

3.在一个双链表中，删除 \( p \) 所指结点（非首、尾结点）的操作是______。  
A. 
```
p->prior->next = p->next; 
p->next->prior = p->prior
```
B.
```
p->prior = p->prior->prior;
p->prior->prior = p
``` 
C.
```
p->next->prior = p;
p->next = p->next->next
``` 
D.
```
p->next = p->prior->prior;
p->prior = p->prior->prior
```


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A。</b>我们要做的就是让前后节点“绕过” p，也就是让 p 的前驱的后继设置成 p 的后继，同时 p 的后继的前驱设置成 p 的前驱。把这个操作翻译成代码就得到了 A 选项。
<p></p>

</details>

4.设 \( n \) 个元素的进栈序列是 \( 1、2、3、…、n \)，其输出序列是 \( p_1、p_2、…、p_n \)，若 \( p_1=3 \)，则 \( p_2 \) 的值为______。  
A. 一定是2  
B. 一定是1  
C. 不可能是1  
D. 以上都不对  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。p_2 要么是接着 p_1 出栈，此时值为2，要么就是接受更多进栈的数，而进栈序列是单调递增的，怎么都不可能下降到 1 。
<p></p>

</details>

5.在数据处理过程中经常需要保存一些中间数据，如果要实现先保存的数据先处理，则应采用______来保存这些数据。  
A. 线性表  
B. 栈  
C. 队列  
D. 单链表  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。题目明示了先进先出的数据结构，这就是队列。
<p></p>

</details>

6.中缀表达式 \( a*(b+c)-d \) 对应的后缀表达式是______。  
A. \( a\ b\ c\ d\ *\ +\ - \)  
B. \( a\ b\ c\ +\ *\ d\ - \)  
C. \( a\ b\ c\ *\ +\ d\ - \)  
D. \( -\ +\ *\ a\ b\ c\ d \)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B</b>。我们把答案的后缀表达式转换成中缀，技巧就是把参数压入一个栈里面，然后读到运算符就弹出最顶上两个数，直到得到结果。D 选项不是后缀表达式，下面分析 ABC:
<p></p>
A: a - (c * d + b)
<p></p>
B: (b + c) * a - d
<p></p>
C: b * c + a - d
<p></p>
</details>

7.设栈 \( s \) 和队列 \( q \) 的初始状态都为空，元素 \( a、b、c、d、e \) 和 \( f \) 依次通过栈 \( s \)，一个元素出栈后即进入队列 \( q \)，若6个元素出队的序列是 \( b、d、c、f、e、a \)，则栈 \( s \) 的容量至少能存______个元素。  
A. 2  
B. 3  
C. 4  
D. 5  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B</b>。首先队列是先进先出的，那么就有：出栈序列=入队序列=出队序列。
<p></p>
然后如图模拟一下栈的运行即可：
<a href="https://imgse.com/i/pEvZa0s"><img src="https://s21.ax1x.com/2025/05/16/pEvZa0s.png" alt="pEvZa0s.png" border="0" /></a>
<p></p>

</details>

8.执行以下______操作时，需要使用队列作为辅助存储空间。  
A. 图的深度优先遍历  
B. 二叉树的先序遍历  
C. 平衡二叉树查找  
D. 图的广度优先遍历  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。ABC 都是递归算法，只有 D 需要利用队列先进先出的特点保存广度信息。
<p></p>

</details>

9.若将 \( n \) 阶上三角矩阵 \( A \) 按列优先顺序压缩存放在一维数组 \( B[1..n(n+1)/2] \) 中，\( A \) 中第一个非零元素 \( a_{1,1} \) 存于 \( B \) 数组的 \( b_1 \) 中，则应存放到 \( b_k \) 中的元素 \( a_{i,j} \)（\( 1≤i≤j \)）的下标 \( i、j \) 与 \( k \) 的对应关系是______。  
A. \( i(i+1)/2 + j \)  
B. \( i(i−1)/2 + j \)  
C. \( j(j+1)/2 + i \)  
D. \( j(j−1)/2 + i \)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。按列存储，也就是每一列都摊平到一维上面，是对列坐标求和之后再加上行坐标，然后再把题目给的(1,1)特殊值带进去就得到 D 选项了。
<p></p>

</details>

10.一棵结点个数为 \( n \)、高度为 \( h \) 的 \( m \)（\( m≥3 \)）次树中，其总分支数是______。  
A. \( nh \)  
B. \( n+m \)  
C. \( n−1 \)  
D. \( h−1 \)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。除了根节点以外，所有节点都能被记作一个分支，因此是总结点数减去 1。
<p></p>

</details>

11.设森林 \( F \) 对应的二叉树为 \( B \)，\( B \) 中有 \( m \) 个结点，其根结点的右子树的结点个数为 \( n \)，森林 \( F \) 中第一棵树的结点个数是______。
A. \(m - n\)  
B. \(m - n - 1\)  
C. \(n + 1\)  
D. 条件不足，无法确定  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。本题考察左儿子右兄弟法。也就是右边的子树都和它的父母节点是同一层级的。右子树的节点数为 n，说明除了第一棵树以外所有树的节点个数为 n。
<p></p>

</details>

12.一棵二叉树的先序遍历序列为 ABCDEF、中序遍历序列为 CBAEDF，则后序遍历序列为______。  
   A. CBEFDA  
   B. FEDCBA  
   C. CBEDFA  
   D. 不确定  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。
<p></p>
首先先序遍历是 ULR 的顺序，中序遍历是 LUR 的顺序，后序遍历是 LRU 的顺序。从中序序列里面找到先序系列的第一个节点，就自然地划分开了左右子树，由此我们可以构建这样一个二叉树：

```
             A
            / \
           B   D
          /   / \
         C   E   F
```

得到后序序列为：CBEFDA
<p></p>

</details>

13.在一个具有 \(n\) 个顶点的有向图中，构成强连通图时至少有______条边。  
   A. \(n\)  
   B. \(n + 1\)  
   C. \(n - 1\)  
   D. \(n / 2\)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。要求强连通，则每个节点至少要有一个入度和一个出度，这样才能保证其他结点能访问到它，它也能访问到其它结点。不难想到把这 n 个点串成一个循环单链表，这样每一个点的出度就是下一个节点的入度，一共 n 条边。若比 n 更小，必然存在有结点没有入度或者出度。
<p></p>

</details>

14.对于有 \(n\) 个顶点的带权连通图，它的最小生成树是指图中任意一个______。  
   A. 由 \(n - 1\) 条权值最小的边构成的子图  
   B. 由 \(n - 1\) 条权值之和最小的边构成的子图  
   C. 由 \(n - 1\) 条权值之和最小的边构成的连通子图  
   D. 由 \(n\) 个顶点构成的极小连通子图，且边的权值之和最小  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。ABC 的说法都有一个潜在问题，即 n - 1 条权值之和最小的边不一定能构成一个连通子图。最小生成树的前提条件应当是一棵树，也就是要连通。
<p></p>

</details>

15.对于有 \(n\) 个顶点、\(e\) 条边的有向图，采用邻接矩阵表示，求单源最短路径的 Dijkstra 算法的时间复杂度为______。  
   A. \(O(n)\)  
   B. \(O(n + e)\)  
   C. \(O(n^2)\)  
   D. \(O(ne)\)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。Dijkstra 算法分两步，对每个节点而言，第一步求取最短路径边，第二步松弛，我们分开来看。
<p></p>
邻接矩阵是一个很糟糕的数据结构。在松弛找邻接边的时候就要遍历所有节点，时间复杂度 O(n)，所以第二步时间复杂度就来到了 O(n^2)。第一步我们可以使用链表做到总 O(n^2) 的时间复杂度或者利用堆实现更优的时间复杂度，但都无力回天，总的时间复杂度为 O(n^2)。
<p></p>

</details>

16.一棵高度为 \(h\) 的平衡二叉树，其中每个非叶子结点的平衡因子均为 0，则该树的结点个数是______。  
   A. \(2^{h - 1} - 1\)  
   B. \(2^{h - 1}\)  
   C. \(2^{h - 1} + 1\)  
   D. \(2^h - 1\)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。平衡因子是左子树高度减去右子树高度，由于平衡因子是0，意味着这是一颗满二叉树。因此节点个数：

$$
N=\sum_{i = 0}^{h - 1}2^i = 2^{h}-1
$$

<p></p>

</details>

17.在对线性表进行折半查找时，要求线性表必须______。  
   A. 以顺序方式存储  
   B. 以链接方式存储  
   C. 以顺序方式存储，且结点按关键字有序排序  
   D. 以链表方式存储，且结点按关键字有序排序  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。要折半，就得支持随机访问。因此必须按顺序存储，同时支持丢弃一半的区间。因此要有序。
<p></p>

</details>

18.假设有 \(k\) 个关键字互为同义词，若用线性探测法把这 \(k\) 个关键字存入哈希表中，至少要进行______次探测。  
   A. \(k - 1\)  
   B. \(k\)  
   C. \(k + 1\)  
   D. \(k(k + 1)/2\)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。这个就是 Codeforces 里面卡 std::unorded_map 的方法，由于它们具有相同的哈希，因此都会装进一个桶里面，拉链法哈希表退化成尾插的链表了，每新装一个数据就得从头探测到尾。
<p></p>

</details>

19.在以下排序算法中，某一趟排序结束后未必能选出一个元素放在其最终位置上的是______。  
   A. 堆排序  
   B. 冒泡排序  
   C. 直接插入排序  
   D. 快速排序  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。插入排序的有序区域并不是全局的有序区域，比如某一趟选出来恰好是最小值，则有序区都要往后挪一个位置。堆排序的有序区是全局的，冒泡排序冒泡一趟，冒上来的元素也不会变了，快排选取的 pivot 在一次划分之后也不会变。
<p></p>

</details>

20.在以下排序方法中，______不需要进行关键字的比较。  
   A. 快速排序  
   B. 归并排序  
   C. 基数排序  
   D. 堆排序  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。基数排序的特点就是不依赖关键字比较。已经证明依赖比较的排序算法时间复杂度不可能优于 O(n log n)。
<p></p>

</details>

### 问答题

1.**已知一棵度为 \(m\) 的树中有 \(n_1\) 个度为 1 的结点、\(n_2\) 个度为 2 的结点、…、\(n_m\) 个度为 \(m\) 的结点，问该树中有多少个叶子结点？（需要给出推导过程）**  


<details>

<summary> 答案 </summary>

<p></p>

$$
\sum_{i = 1}^m (i - 1)n_i
$$

<p></p>
我们考虑最朴素的构造：所有结点从根节点开始串成一条链，然后对于度为 k 的结点，补充 k - 1 个叶子结点，就得到了上式。接下来调整树的结构，可以发现无论怎么调整，只要满足题目条件，都不会改变叶子节点个数。
<p></p>

</details>

2.**设关键字序列 \(D = (1, 12, 5, 8, 3, 10, 7, 13, 9)\)，试完成下列各题：**  
   （1）依次取 \(D\) 中的各关键字，构造一棵二叉排序树 \(bt\)。  
   （2）如何依据此二叉树 \(bt\) 得到 \(D\) 的一个关键字递增序列。  
   （3）画出在二叉树 \(bt\) 中删除 12 后的树结构。


<details>

<summary> 答案 </summary>

<p></p>
<a href="https://imgse.com/i/pEvUH0I"><img src="https://s21.ax1x.com/2025/05/17/pEvUH0I.png" alt="pEvUH0I.png" border="0" /></a>
<p></p>

</details>


### 算法设计题

1.**设 \( A=(a_1, a_2, \cdots, a_n) \)，\( B=(b_1, b_2, \cdots, b_m) \) 是两个递增有序的线性表（其中 \( n、m \) 均大于 1），且所有数据元素均不相同。假设 \( A、B \) 均采用带头结点的单链表存放，设计一个尽可能高效的算法判断 \( B \) 是否为 \( A \) 的一个连续子序列，并分析你设计的算法的时间复杂度和空间复杂度。（15 分）**


<details>

<summary> 答案 </summary>

<p></p>
思路很简单，由于 A，B 都是有序的，因此失配一次，整个匹配都会失败。据此可以写出下面的代码：

```cpp
bool match(ListNode* a, ListNode* b)
{
   while(a != NULL)
   {
      if(a->data == b->data)
      {
         while(b != NULL)
         {
            if(a == NULL || a->data != b->data)
               return false;
            a = a->next;
            b = b->next;
         }
         return true;
      }
      a = a->next;
   }
   return false;
}
```

<p></p>

<details>

<summary> 变式：如果 A 和 B 并不有序，存在线性时间的算法吗？ </summary>

<p></p>
存在。使用 KMP 算法即可。
<p></p>

</details>

<p></p>

</details>

2.**假设二叉树 \( b \) 采用二叉链存储结构存储，试设计一个算法，求该二叉树中从根结点出发的一条最长的路径长度，并输出此路径上各结点的值。（15 分）**


<details>

<summary> 答案 </summary>

<p></p>
本题是树形 dp 入门题，同时代码稍作修改，就可以做树的长链剖分。

```cpp
enum class path { left, right };
struct BTNode {
   BTNode* lchild;
   BTNode* rchild;
   path p;
}

int dfs(BTNode* root, int depth)
{
   if(root == NULL) return depth;
   int ldepth = dfs(root->lchild, depth + 1);
   int rdepth = dfs(root->rchild, depth + 1);
   if(ldepth >= rdepth)
   {
      root->p = path::left;
      return ldepth;
   }
   root->p = path::right;
   return rdepth;
}

void output(BTNode* root)
{
   int dep = dfs(root, 1);
   std::cout << dep << std::endl;
   while(root != NULL)
   {
      std::cout << root->data << std::endl;
      if(root->p == path::right)
         root = root->rchild;
      else
         root = root->lchild;
   }
}
```

<p></p>

</details>

## 23年真题
### 单项选择题

1.设有序表中有 1000 个元素，则用二分查找查找元素 X 最多需要比较 ____ 次。  
   A. 25
   B. 10
   C. 7
   D. 1  

<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B</b>。参考第一套题的单选第12题。
<p></p>

</details>

2.设散列表中有 m 个存储单元，散列函数 H(key)=key % p，则 p 最好选择 ____ 。  
   A. 小于等于 m 的最大奇数
   B. 小于等于 m 的最大素数  
   C. 小于等于 m 的最大偶数
   D. 小于等于 m 的最大合数  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B</b>。为了让数据减少冲突，我们应该尽量避免出现整除的情况。
<p></p>

</details>


3.哈夫曼树有 m 个叶子结点，若用二叉链表作为存储结构，则该哈夫曼树共有 ____ 个空指针域。  
   A. 2m - 1
   B. 2m
   C. 2m + 1
   D. 4m  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B</b>。首先根节点有两个空指针域，然后每次合并时都增加一个新的根节点和新的叶子节点，叶子节点又有两个空指针域，因此是 2m。
<p></p>

</details>


4.字符 A、B、C 依次进入一个栈，按出栈的先后顺序组成不同的字符串，至多可以组成 ____ 个不同的字符串。  
   A. 14
   B. 5
   C. 6
   D. 8  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：B</b>。
<p></p>
字符串如下：

```
ABC
ACB
BAC
BCA
CBA
```

<p></p>
或者直接利用公式：

$$
N = \dfrac{C^n_{2n}}{n+1} = \dfrac{6!}{3!\times 3!\times 4} = 5
$$

<p></p>

</details>


5.栈和队列共同的特点是 ____ 。  
   A. 只允许在端点处插入和删除元素
   B. 都是先进后出  
   C. 都是先进先出
   D. 没有共同点  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。栈是先进后出，队列是先进先出。D选项凑数的，别选。
<p></p>

</details>


6.在二叉排序树中插入一个新的结点时，若树中不存在与待插入关键字相同的结点，且新结点的关键字小于根结点的关键字，则新结点将成为 ____ 。  
   A. 左子树的新叶子结点
   B. 左子树的分支结点  
   C. 右子树的新叶子结点
   D. 右子树的分支结点  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。首先新节点的关键字更小，因此插在左子树。因为一开始没有找到，所以是连在某个叶子节点的空指针域上面，形成新的叶子节点。


<details>

<summary> 如果不是 BST 而是使用平衡树呢？ </summary>

<p></p>
由于有旋转操作，因此无法保证插入的节点最后一定是叶子节点，但是一定是在左子树。
<p></p>

</details>

</details>


7.设有 n 个待排序的记录关键字，则在堆排序中需要 ____ 个辅助记录单元。  
   A. 1
   B. n
   C. log₂n
   D. n²  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。原地建堆。
<p></p>

</details>


8.设顺序线性表的长度为 30，分成 5 块，每块 6 个元素，如果采用分块查找，则其平均查找长度为 ____ 。  
   A. 6
   B. 11
   C. 5
   D. 6.5  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。首先确定在哪一个块里面，然后确定在块内的哪一个元素。 5 / 2 + 6 / 2 = 6.5。
<p></p>

</details>


9.下列关于二叉树遍历的叙述中，正确的是 ____ 。  
   A. 若一个叶子是某二叉树的中序遍历的最后一个结点，则它必是该二叉树的前序遍历的最后一个结点  
   B. 若一个结点是某二叉树的前序遍历的最后一个结点，则它必是该二叉树的中序遍历的最后一个结点  
   C. 若一个结点是某二叉树的中序遍历的最后一个结点，则它必是该二叉树的前序遍历的最后一个结点  
   D. 若一个树叶是某二叉树的前序遍历的最后一个结点，则它必是该二叉树的中序遍历的最后一个结点


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。前序：ULR，中序LUR，后序LRU，可见前序和中序都把右子树放在最后遍历。为了要让最后一个节点一致，必须保证右子树存在，因此这个结点一定是叶子。
<p></p>

</details>


10.邻接多重表与十字链表的共同特点是 ____ 。  
A. 都是针对有向图  
B. 边结点都需要一个 mark 域标明对应的边是否已被访问  
C. 都是针对无向图  
D. 所有的边只通过边结点表达一次  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：D</b>。十字链表针对有向图，邻接多重表针对无向图。其他内容可以自行复习教材内容。
<p></p>

</details>


11.以下四个选项中，有一项与其余三项不同，这一项是 ____ 。  
A. 二叉搜索树  
B. AVL 树  
C. B+树  
D. B-树  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b> BST 不是平衡的。
<p></p>

</details>


12. 最大容量为 n 的循环队列，队满时仍然保留一个数组元素为空。若 front 指向队列第一个元素，rear 指向队列最后一个元素，则队列长度的计算公式为 ____ 。  
A. rear-front-1  
B. rear-front+1  
C. (rear-front+n-1)%n  
D. (rear-front+n+1)%n  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。参考第一套的第五题。由于这里留了一个空元素，所以应该减去1。
<p></p>

</details>


13. 从二叉搜索树中查找一个元素时，其时间复杂度大致为 ____ 。  
A. O(log₂n)  
B. O(1)  
C. O(n²)  
D. O(n)  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：A</b>。考虑随机建立 BST 的平均深度为 O(log₂n) 即可。
<p></p>

</details>


14. 二叉树在线索化之后，仍不能有效求解的问题是 ____ ？  
A. 先序线索二叉树中求先序后继  
B. 中序线索二叉树中求中序后继  
C. 中序线索二叉树中求中序前驱  
D. 后序线索二叉树中求后序后继  


<details>

<summary> 答案 </summary>

<p></p>
<b>答案：C</b>。线索二叉树的线索是单向的，因此求取后继是 O(1) 的，而求前驱仍然需要 O(n)。
<p></p>

</details>



### 填空题

1. 将一棵拥有 m 个叶子结点和 n 个非叶子结点的树转换成对应的二叉树，则在此二叉树中有 ____ 个结点有（非空的）右孩子。  


<details>

<summary> 答案 </summary>

<p></p>

$$
m-1
$$

将一棵树转换为二叉树时，采用“左孩子右兄弟”规则。原树中每个非叶子节点的子节点转换为二叉树后，第一个子节点作为左孩子，其余子节点依次作为前一个兄弟的右孩子。因此，每个非叶子节点若有 \( k \) 个子节点，则转换后会产生 \( k-1 \) 个右孩子节点。

原树共有 \( m \) 个叶子节点和 \( n \) 个非叶子节点，总边数为 \( m + n - 1 \)。所有非叶子节点的子节点数目之和等于总边数，即 \( \sum k_i = m + n - 1 \)。每个非叶子节点贡献的右孩子数目为 \( k_i - 1 \)，因此总共有：
\[
\sum (k_i - 1) = \sum k_i - \sum 1 = (m + n - 1) - n = m - 1
\]
个。
<p></p>

</details>


2. 在对 m 阶 B-树插入元素的过程中，每向一个结点插入一个关键码后，若该结点的关键码个数等于 ____ 时，则必须把它分裂为 2 个结点。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


3. 假设关键字序列为(K₁, K₂, …, Kₙ)，则用筛选法建初始堆必须从第 ____ 个元素开始进行筛选。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


4. 在拥有 n 个结点的二叉排序树中，如果存在度为 2 的结点，那么这棵二叉排序树的最大高度为 ____ 。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


5. 索引存储结构是由 ____ 和 ____ 两部分组成，其中 ____ 部分要求按关键字排序。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


6. 已知一棵完全二叉树中共有 768 个结点，则该树中共有 ____ 个叶子结点。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


7. 在双向链表中指定的结点之前插入一个新结点需修改的指针数是 ____ 个。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


8. 一棵二叉树高度为 h，所有结点的度或为 0，或为 2，则这棵二叉树最少有 ____ 个结点。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


9. 在含 n 个顶点和 e 条边的无向图的邻接矩阵中，零元素的个数为 ____ 。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


10. 在单链表上难以实现的排序方法有 ____ 、 ____ 和 ____ 等。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


11. 不论是顺序存储结构的栈还是链式存储结构的栈，其入栈和出栈操作的时间复杂度均为 ____ 。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


12. 设某二叉树的前序遍历序列为 ABC，后序遍历序列为 BCA，则该二叉树的中序遍历序列为 ____ 。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


### 判断题

1. 在相同的规模 n 下，复杂度 O(n) 的算法在时间上总是优于复杂度 O(2ⁿ) 的算法。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


2. 消除递归不一定需要使用栈。


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


3. 若一棵二叉树的任意结点的左子树和右子树都是二叉搜索树，则这棵二叉树必是二叉搜索树。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


4. 循环队列就是用循环链表表示的队列。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


5. 在多关键字排序中，从第二低位关键字开始，所采用的单趟排序算法必须是稳定的排序算法。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


6. 哈夫曼编码一定是前缀编码；一套前缀编码，按照越长的编码分配给越低频率字符的原则进行分配，就一定是哈夫曼编码。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


7. 从连通无向图的同一个顶点出发，BFS 生成树的高度不会大于 DFS 生成树的高度。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


8. 利用筛选法对 n 个关键字进行建堆操作，算法复杂度为 O(n)。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


9. 至少有一个顶点的度为 1 的无向连通图不可能包含回路。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


10. 一个无向图的连通分量是其极大的连通子图。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


11. 排序所需要的所有操作都在外存中完成，则称之为外排序。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


12. 在对半查找的二叉判定树中，外部结点只可能出现在最下的两层结点中。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


13. 存在一棵总共有 2016 个结点的二叉树，其中有且仅有 16 个结点只有一个孩子。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


14. Floyd 算法求两个顶点的最短路径时，path_{k-1} 一定是 path_k 的子集。  


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


15. 不能向静态查找表中插入新的元素。


<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


### 综合题

1. 设有如下无向图 G，给出该图的最小生成树上边的集合并计算最小生成树各边上的权值之和。（8 分）



<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


2. 设有如下图所示的 AOE 网（其中 vi（i=1，2，…，6）表示事件，边上表示活动的天数）。（5 分）

（1）找出所有的关键路径。

（2）v3 事件的最早开始时间是多少。



<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


3. 试列出如下图中全部可能的拓扑排序序列。（8分）



<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


4. 已知某有向图有5个顶点，各顶点的度之和为14。且满足如下条件：  
(1)从顶点v0出发的深度优先遍历次序与广度优先遍历次序相同且唯一，是v0,v3,v1,v4,v2。  
(2)存在入度为0的顶点，也存在出度为0的顶点。  
(3)图中有且仅有一条回路，其路径长度大于2。  
请画出该图并画出该图的邻接表。（5分）  



<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>


### 算法设计题

1. 给定一棵二叉树，判断该二叉树在结构上是否属于左右对称的结构。要求用递归函数实现，且有最低的时间复杂度和空间复杂度。  



<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>



2. 给定正整数数组A，设计算法求从下标0出发走出数组范围的最小步数及每一步下标。输入示例：A = [2,5,1,1,1]，输出示例：3，0->2->1->6。要求非递归实现，且时间和空间复杂度最小。



<details>

<summary> 答案 </summary>

<p></p>

<p></p>

</details>

