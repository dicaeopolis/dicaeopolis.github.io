# STL的一些性能测试

众所周知 `regex` 库像马车一样慢，而 `unrodered_map` 也常常因为常数过大而被诸多算法竞赛选手所摒弃。但 `STL` 并非铁板一块，总会有一些好用且效率高的容器值得一用。本文试图对 `STL` 中的一些经典容器及算法进行性能测试与对比，看看哪些轮子是好用的。

## 测试平台和流程

本次测试使用 Intel® Pentium® Gold 8505 @ 2.50GHz 芯片，机带内存 8GB，操作系统为 Windows 24H2 26100.3775 ，编译环境为 MSYS2 ，编译器使用 clang 20.1.3 和 gcc 13.3.0。

对每一次测试取不同数据量，每个数据量针对不同编译器测量多次后取平均值。

测试数据由随机算法生成并保存。例如：


<details>
<summary>点击查看代码</summary>

```cpp
#include<iostream>
#include<random>
#include<chrono>

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    int n;
    std::cin >> n;
    std::cout << n;
    while(n--)
        std::cout<<engine()<<' ';
    return 0;
}
```

</details>

测试计时使用下面的脚本：


<details>
<summary>点击查看代码</summary>

```python
import os
import time
import random

data_size = int(1e5)
test_round = 50
sorces_filename = ['a', 'b', 'c']
datagen_path = 'datagen.exe'
testdata_filename = 'testdata.in'
output_filename = 'output.out'

print('[+] Cleaning directory.')
os.system('find . -type f -name "*.exe" ! -name "datagen.exe" -exec rm -f {} +')
os.system('rm *.in *.out')

#compile
gcc_instructions = [f'g++ -O2 -lm -o {fn}_gcc {fn}.cc' for fn in sorces_filename]
clang_instructions = [f'clang++ -O2 -lm -o {fn}_clang {fn}.cc' for fn in sorces_filename]
instructions = gcc_instructions + clang_instructions

print('[+] Compiling files.')
for cmd in instructions:
    print(f'  [+] Using command: {cmd}')
    os.system(cmd)

gcc_run_cmd = [(f'.\\{fn}_gcc < {testdata_filename} > {output_filename}', f'{fn}_gcc')\
                for fn in sorces_filename]
clang_run_cmd = [(f'.\\{fn}_clang < {testdata_filename} > {output_filename}', f'{fn}_clang')\
                  for fn in sorces_filename]
run_cmds = gcc_run_cmd + clang_run_cmd

time_data = {}
for cmd in run_cmds:
    time_data[cmd[1]] = 0

#test
for round in range(test_round):
    random.shuffle(run_cmds)
    print(f'[+] Test round {round + 1}:')
    print('[+] Cleaning directory.')
    os.system('rm *.in *.out')
    test_gen = f'({datagen_path} {data_size}) > {testdata_filename}'
    print('[+] Generating test data.')
    os.system(test_gen)
    for cmd in run_cmds:
        run_cmd = cmd[0]
        fn = cmd[1]
        print(f'  [+] Start testing file {fn}...')
        start_time = time.time()
        os.system(run_cmd)
        end_time = time.time()
        elapsed_time_ms = int(1000 * (end_time - start_time))
        print(f'  [-] Over. time usage: {elapsed_time_ms} ms')
        time_data[cmd[1]] += elapsed_time_ms
        #time.sleep(random.uniform(0,3))

print('[-] Time benchmark over.')
print()
print('-*- Results -*-')
print(f'Ran {test_round} rounds for {data_size} items.')
for item in time_data.items():
    print(f'file {item[0]} average run time: {int(item[1] / test_round)} ms.')
```

</details>

所有输入输出都使用 `std::cin` 和 `std::cout` 进行，流同步已经关闭。

## `std::vector`, `std::array` 和原生数组

本轮测试以下项目：

- 存入数据并输出
- 随机访问下标并求和
- 使用 `std::sort` 排序

### 顺序存储测试

<details>
<summary>点击查看代码</summary>

使用 std::vector：

<details>
<summary>点击查看代码</summary>

```cpp
#include<iostream>
#include<vector>

int main()
{
    std::vector<int> vec;
    int n;
    std::cin >> n;
    while (n--) {
        int x;
        std::cin >> x;
        vec.push_back(x);
    }
    long long sum = 0;
    for(auto i : vec) {
        std::cout << i << ' ';
        sum += i;
        sum %= 998244353;
    }
    std::cout << sum << '\n';
    for(auto it = vec.rbegin(); it != vec.rend(); ++it)
        std::cout << *it << ' ';
    return 0;
}
```

</details>

使用 std::array：


<details>
<summary>点击查看代码</summary>

```cpp
#include<iostream>
#include<array>
constexpr int SIZE = int(1e8+5);
std::array<int, SIZE> arr;

int main()
{
    int n;
    std::cin >> n;
    for(int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        arr[i] = x;
    }
    long long sum = 0;
    for(int i = 0; i < n; ++i) {
        std::cout << arr[i] << ' ';
        sum += arr[i];
        sum %= 998244353;
    }
    std::cout << sum << '\n';
    for(int i = n - 1; i >= 0; --i)
        std::cout << arr[i] << ' ';
    return 0;
}
```
</details>

使用原生数组：

<details>
<summary>点击查看代码</summary>

```cpp
#include<iostream>
constexpr int SIZE = int(1e8+5);
int arr[SIZE];

int main()
{
    int n;
    std::cin >> n;
    for(int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        arr[i] = x;
    }
    long long sum = 0;
    for(int i = 0; i < n; ++i) {
        std::cout << arr[i] << ' ';
        sum += arr[i];
        sum %= 998244353;
    }
    std::cout << sum << '\n';
    for(int i = n - 1; i >= 0; --i)
        std::cout << arr[i] << ' ';
    return 0;
}
```
</details>

</details>


### 随机访问测试

<details>
<summary>点击查看代码</summary>


使用 std::vector：

<details>
<summary>点击查看代码</summary>


```cpp
#include<iostream>
#include<vector>
#include<random>

int main()
{
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    std::vector<int> vec;
    int n, m;
    std::cin >> n;
    m = n;
    while (n--) {
        int x;
        std::cin >> x;
        vec.push_back(x);
    }
    long long sum = 0;
    for(int _ = 0; _ < m ; ++_) {
        auto i = vec[engine() % m];
        sum += i;
        sum %= 998244353;
    }
    std::cout << sum;
    return 0;
}
```

</details>

使用 std::array：


<details>
<summary>点击查看代码</summary>


```cpp
#include<iostream>
#include<array>
#include<random>

constexpr size_t SIZE = 1e6+5;
std::array<int, SIZE> vec;
int main()
{
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    int n, m;
    std::cin >> n;
    m = n;
    for(int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        vec[i] = x;
    }
    long long sum = 0;
    for(int _ = 0; _ < m ; ++_) {
        auto i = vec[engine() % m];
        sum += i;
        sum %= 998244353;
    }
    std::cout << sum;
    return 0;
}
```

</details>

使用原生数组：


<details>
<summary>点击查看代码</summary>


```cpp
#include<iostream>
#include<array>
#include<random>

constexpr size_t SIZE = 1e6+5;
int vec[SIZE];
int main()
{
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    int n, m;
    std::cin >> n;
    m = n;
    for(int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        vec[i] = x;
    }
    long long sum = 0;
    for(int _ = 0; _ < m ; ++_) {
        auto i = vec[engine() % m];
        sum += i;
        sum %= 998244353;
    }
    std::cout << sum;
    return 0;
}
```

</details>

</details>


### 排序测试

<details>
<summary>点击查看代码</summary>

使用 std::vector ：


<details>
<summary>点击查看代码</summary>


```cpp
#include<iostream>
#include<vector>
#include<random>
#include<algorithm>
int main()
{
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    std::vector<int> vec;
    int n, m;
    std::cin >> n;
    m = n;
    while (n--) {
        int x;
        std::cin >> x;
        vec.push_back(x);
    }
    std::sort(vec.begin(), vec.end());
    return 0;
}
```

</details>


使用 std::array：


<details>
<summary>点击查看代码</summary>


```cpp
#include<iostream>
#include<array>
#include<random>
#include<algorithm>
constexpr size_t SIZE = 1e6+5;
std::array<int, SIZE> vec;
int main()
{
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    int n, m;
    std::cin >> n;
    m = n;
    for(int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        vec[i] = x;
    }
    std::sort(vec.begin(), vec.begin() + m + 1);
    return 0;
}
```

</details>


使用原生数组：


<details>
<summary>点击查看代码</summary>


```cpp
#include<iostream>
#include<array>
#include<random>
#include<algorithm>

constexpr size_t SIZE = 1e6+5;
int vec[SIZE];
int main()
{
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    int n, m;
    std::cin >> n;
    m = n;
    for(int i = 0; i < n; ++i) {
        int x;
        std::cin >> x;
        vec[i] = x;
    }
    std::sort(vec, vec + m + 1);
    return 0;
}
```

</details>

</details>

### 结果和分析


<details>
<summary>点击查看测试结果</summary>
测试1：
```
-*- Results -*-
Ran 50 rounds for 1000 items.
file a_gcc average run time: 114.56 ± 44.77 ms (39.083%).
file b_gcc average run time: 114.82 ± 47.26 ms (41.157%).
file c_gcc average run time: 109.58 ± 32.59 ms (29.745%).
file a_clang average run time: 63.96 ± 71.77 ms (112.212%).
file b_clang average run time: 56.38 ± 44.58 ms (79.077%).
file c_clang average run time: 61.92 ± 49.93 ms (80.635%).

-*- Results -*-
Ran 50 rounds for 100000 items.
file a_gcc average run time: 415.8 ± 51.44 ms (12.37%).
file b_gcc average run time: 420.12 ± 51.33 ms (12.218%).
file c_gcc average run time: 425.22 ± 59.65 ms (14.028%).
file a_clang average run time: 351.1 ± 52.09 ms (14.835%).
file b_clang average run time: 353.44 ± 51.66 ms (14.615%).
file c_clang average run time: 352.08 ± 49.55 ms (14.074%).

-*- Results -*-
Ran 20 rounds for 1000000 items.
file a_gcc average run time: 3383.9 ± 306.06 ms (9.045%).
file b_gcc average run time: 3349.85 ± 397.41 ms (11.864%).
file c_gcc average run time: 3346.3 ± 306.4 ms (9.156%).
file a_clang average run time: 3186.65 ± 296.0 ms (9.289%).
file b_clang average run time: 3225.5 ± 329.33 ms (10.21%).
file c_clang average run time: 3218.15 ± 303.94 ms (9.445%).
```
测试2：
```
-*- Results -*-
Ran 50 rounds for 1000 items.
file a_gcc average run time: 90.12 ± 26.23 ms (29.107%).
file b_gcc average run time: 88.26 ± 13.46 ms (15.254%).
file c_gcc average run time: 86.8 ± 15.66 ms (18.044%).
file a_clang average run time: 32.16 ± 22.44 ms (69.763%).
file b_clang average run time: 28.6 ± 12.94 ms (45.254%).
file c_clang average run time: 30.24 ± 14.96 ms (49.455%).

-*- Results -*-
Ran 50 rounds for 100000 items.
file a_gcc average run time: 336.48 ± 21.05 ms (6.257%).
file b_gcc average run time: 334.84 ± 14.22 ms (4.246%).
file c_gcc average run time: 340.24 ± 18.11 ms (5.323%).
file a_clang average run time: 171.8 ± 16.36 ms (9.524%).
file b_clang average run time: 171.64 ± 14.08 ms (8.202%).
file c_clang average run time: 171.5 ± 13.84 ms (8.071%).

-*- Results -*-
Ran 20 rounds for 1000000 items.
file a_gcc average run time: 3029.5 ± 397.96 ms (13.136%).
file b_gcc average run time: 2998.55 ± 339.45 ms (11.321%).
file c_gcc average run time: 2968.05 ± 257.55 ms (8.677%).
file a_clang average run time: 1614.05 ± 114.01 ms (7.064%).
file b_clang average run time: 1641.65 ± 185.24 ms (11.284%).
file c_clang average run time: 1641.5 ± 146.92 ms (8.95%).
```
测试3：
```
-*- Results -*-
Ran 50 rounds for 1000 items.
file a_gcc average run time: 82.18 ± 15.84 ms (19.279%).
file b_gcc average run time: 87.38 ± 20.09 ms (22.997%).
file c_gcc average run time: 85.56 ± 20.65 ms (24.14%).
file a_clang average run time: 22.98 ± 13.12 ms (57.088%).
file b_clang average run time: 25.1 ± 13.76 ms (54.824%).
file c_clang average run time: 29.16 ± 22.7 ms (77.836%).

-*- Results -*-
Ran 50 rounds for 100000 items.
file a_gcc average run time: 348.94 ± 40.78 ms (11.686%).
file b_gcc average run time: 348.36 ± 35.0 ms (10.048%).
file c_gcc average run time: 345.16 ± 38.33 ms (11.105%).
file a_clang average run time: 170.74 ± 19.72 ms (11.552%).
file b_clang average run time: 176.56 ± 25.44 ms (14.411%).
file c_clang average run time: 174.12 ± 21.54 ms (12.369%).

-*- Results -*-
Ran 20 rounds for 1000000 items.
file a_gcc average run time: 2923.05 ± 241.33 ms (8.256%).
file b_gcc average run time: 2978.0 ± 278.09 ms (9.338%).
file c_gcc average run time: 2990.55 ± 287.74 ms (9.622%).
file a_clang average run time: 1619.85 ± 148.65 ms (9.177%).
file b_clang average run time: 1602.1 ± 137.03 ms (8.553%).
file c_clang average run time: 1648.2 ± 238.51 ms (14.471%).
```

</details>

注意这里`gcc`和`clang`有一定的I/O性能差距，但是容器本身的用时差距不大，甚至没有因为性能波动导致的时间差大。

结论：对于所有情形，各个容器的性能基本没有差别，因为这三个容器底层都是连续的内存块，抽象的时间成本非常低。但是考虑到数组和裸指针纠缠不清的关系，还是更推荐使用 `std::array` 和 `std::vector` 。

由于 `std::vector` 是指数扩容，均摊的时间复杂度为 $O(1)$ 。一般 `std::vector` 扩容的场合都是在读入阶段，所以性能开销也不大。而且即使 `std::vector` 的数据是申请在堆上面，对性能的影响也不大。

当然 `std::array` 就是原生数组很经典的零成本抽象了。


```
-*- Results -*-
Ran 20 rounds for 100000 items.
File a_gcc average run time: 2186.8 ± 306.2 ms (14.002%).
File b_gcc average run time: 2183.45 ± 462.09 ms (21.163%).
File c_gcc average run time: 2193.9 ± 496.88 ms (22.648%).
File unordered_map_gcc average run time: 3621.0 ± 416.51 ms (11.503%).
File a_clang average run time: 1509.0 ± 278.87 ms (18.48%).
File b_clang average run time: 1611.0 ± 418.77 ms (25.995%).
File c_clang average run time: 1510.6 ± 292.32 ms (19.351%).
File unordered_map_clang average run time: 3298.75 ± 1400.99 ms (42.47%).
```

## `std::unordered_map` 和手写哈希

本轮测试使用以下几份代码：

<details class = "warning">
<summary>Warning</summary>
由于 CRC64 的实现利用了编译期生成 CRC 表，以及手写 `unordered_map` 的实现里面用到了一些比较新的语言特性，请确保你的编译器支持 `c++17`。如果遇到如下错误：

```
./unordered_map.cc:13:35: warning: variable declaration in a constexpr function is a C++14 extension [-Wc++14-extensions]
   13 |         std::array<uint64_t, 256> table = {};
      |                                   ^
./unordered_map.cc:14:9: error: statement not allowed in constexpr function
   14 |         for (int i = 0; i < 256; ++i) {
      |         ^
./unordered_map.cc:55:9: error: 'auto' return without trailing return type; deduced return types are a C++14 extension
   55 |         auto& get_item(const key_type& key)
      |         ^
./unordered_map.cc:60:15: error: 'auto' return without trailing return type; deduced return types are a C++14 extension
   60 |         const auto& get_item(const key_type& key) const
      |               ^
1 warning and 3 errors generated.
```

或者如下错误：

```
./unordered_map.cc:11:41: error: constexpr function never produces a constant expression [-Winvalid-constexpr]
   11 |     constexpr std::array<uint64_t, 256> generate_crc64_table()
      |                                         ^~~~~~~~~~~~~~~~~~~~
./unordered_map.cc:18:13: note: non-constexpr function 'operator[]' cannot be used in a constant expression
   18 |             table[i] = crc;
      |             ^
D:/msys64/clang64/include/c++/v1/array:268:65: note: declared here
  268 |   _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 reference operator[](size_type __n) _NOEXCEPT {
      |                                                                 ^
./unordered_map.cc:23:41: error: constexpr variable 'crc64_table' must be initialized by a constant expression
   23 |     constexpr std::array<uint64_t, 256> crc64_table = generate_crc64_table();
      |                                         ^             ~~~~~~~~~~~~~~~~~~~~~~
./unordered_map.cc:18:13: note: non-constexpr function 'operator[]' cannot be used in a constant expression
   18 |             table[i] = crc;
      |             ^
./unordered_map.cc:23:55: note: in call to 'generate_crc64_table()'
   23 |     constexpr std::array<uint64_t, 256> crc64_table = generate_crc64_table();
      |                                                       ^~~~~~~~~~~~~~~~~~~~~~
D:/msys64/clang64/include/c++/v1/array:268:65: note: declared here
  268 |   _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 reference operator[](size_type __n) _NOEXCEPT {
      |                                                                 ^
2 errors generated.
```

请确保加上 `-std=c++17` 选项。
</details>

- `a.cc` : 原生 `std::unordered_map` 加 CRC64 哈希

<details>
<summary>点击查看代码</summary>

```
#include <array>
#include <string>
#include <iostream>
#include <unordered_map>

constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;

constexpr std::array<uint64_t, 256> generate_crc64_table()
{
    std::array<uint64_t, 256> table = {};
    for (int i = 0; i < 256; ++i) {
        uint64_t crc = i;
        for (int j = 0; j < 8; ++j)
            crc = (crc & 1) ? (crc >> 1) ^ CRC64_POLY : crc >> 1;
        table[i] = crc;
    }
    return table;
}

constexpr std::array<uint64_t, 256> crc64_table = generate_crc64_table();

uint64_t crc64(const uint8_t *data, size_t length)
{
    uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
    for (size_t i = 0; i < length; i++) {
        uint8_t index = (uint8_t)(crc ^ data[i]);
        crc = (crc >> 8) ^ crc64_table[index];
    }
    return crc ^ 0xFFFFFFFFFFFFFFFFULL;
}

struct my_hash {
    uint64_t operator()(const uint64_t& q) const
    {
        uint64_t data = q;
        uint8_t *d = (uint8_t*)&data;
        return crc64(d, sizeof(data));
    }
};

int main()
{
    int n, m;
    std::cin>>n>>m;
    std::unordered_map<uint64_t, bool, my_hash> map;
    while(n--)
    {
        uint64_t s;
        std::cin >> s;
        map[s] = true;
    }
    while(m--)
    {
        uint64_t q;
        std::cin >> q;
        std::cout << (map[q] ? "hit\n" : "miss\n");
    }
    return 0;
}    
```
</details>

- `b.cc` : 原生 `std::unordered_map` 加原生哈希

<details>
<summary>点击查看代码</summary>

```
#include <array>
#include <string>
#include <iostream>
#include <unordered_map>

int main()
{
    int n, m;
    std::cin>>n>>m;
    std::unordered_map<uint64_t, bool> map;
    while(n--)
    {
        uint64_t s;
        std::cin >> s;
        map[s] = true;
    }
    while(m--)
    {
        uint64_t q;
        std::cin >> q;
        std::cout << (map[q] ? "hit\n" : "miss\n");
    }
    return 0;
}    
```

</details>

- `c.cc` : 原生 `std::unordered_map` 加 [OI-Wiki 这个条目里面介绍的](https://oi-wiki.org/lang/csl/unordered-container/)哈希

<details>
<summary>点击查看代码</summary>

```
#include <array>
#include <chrono>
#include <string>
#include <iostream>
#include <unordered_map>

struct my_hash {
    static uint64_t splitmix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
  
    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM =
            std::chrono::steady_clock::now().time_since_epoch().count();
      return splitmix64(x + FIXED_RANDOM);
    }
  };

int main()
{
    int n, m;
    std::cin>>n>>m;
    std::unordered_map<uint64_t, bool, my_hash> map;
    while(n--)
    {
        uint64_t s;
        std::cin >> s;
        map[s] = true;
    }
    while(m--)
    {
        uint64_t q;
        std::cin >> q;
        std::cout << (map[q] ? "hit\n" : "miss\n");
    }
    return 0;
}
```

</details>

- `unordered_map.cc` : 手写实现哈希表加 CRC64 哈希

<details>
<summary>点击查看代码</summary>

```
#include <list>
#include <array>
#include <vector>
#include <chrono>
#include <utility>
#include <iostream>

namespace crc64 {    
    constexpr uint64_t CRC64_POLY = 0x42F0E1EBA9EA3693ULL;

    constexpr std::array<uint64_t, 256> generate_crc64_table()
    {
        std::array<uint64_t, 256> table = {};
        for (int i = 0; i < 256; ++i) {
            uint64_t crc = i;
            for (int j = 0; j < 8; ++j)
                crc = (crc & 1) ? (crc >> 1) ^ CRC64_POLY : crc >> 1;
            table[i] = crc;
        }
        return table;
    }

    constexpr std::array<uint64_t, 256> crc64_table = generate_crc64_table();

    uint64_t crc64(const uint8_t *data, size_t length)
    {
        uint64_t crc = 0xFFFFFFFFFFFFFFFFULL;
        for (size_t i = 0; i < length; i++) {
            uint8_t index = (uint8_t)(crc ^ data[i]);
            crc = (crc >> 8) ^ crc64_table[index];
        }
        return crc ^ 0xFFFFFFFFFFFFFFFFULL;
    }
}


constexpr size_t hash_mod = 126271;

template<typename key_type, typename value_type>
class unordered_map {
    typedef std::pair<key_type, value_type> mapped_type;
    private:
        std::vector<std::list<mapped_type>> table;
        size_t pair_cnt = 0;
        size_t get_idx(const key_type& key)
        {
            auto hash = crc64::crc64((uint8_t*) &key, sizeof(key));
            return static_cast<size_t>(hash % hash_mod);
        }
        const size_t get_idx(const key_type& key) const
        {
            auto hash = crc64::crc64((uint8_t*) &key, sizeof(key));
            return static_cast<size_t>(hash % hash_mod);
        }
        auto& get_item(const key_type& key)
        {
            auto idx = get_idx(key);
            return table[idx];
        }
        const auto& get_item(const key_type& key) const
        {
            auto idx = get_idx(key);
            return table[idx];
        }
    public:
        unordered_map() : table(hash_mod) {}
        value_type& operator[](const key_type& key)
        {
            auto& item = get_item(key);
            for(auto& val : item)
                if(val.first == key)
                    return val.second;
            value_type val;
            item.push_back(std::make_pair(key, val));
            ++pair_cnt;
            return item.back().second;
        }
        bool empty() const noexcept
        {
            return !(pair_cnt);
        }
        bool find(const key_type &key) const noexcept
        {
            const auto& item = get_item(key);
            for(const auto& val : item)
                if(val.first == key)
                    return true;
            return false;
        }
        void erase(const key_type& key)
        {
            auto& item = get_item(key);
            for(auto it = item.begin(); it != item.end(); ++it)
                if(it->first == key) {
                    item.erase(it);
                    --pair_cnt;
                    break;
                }
        }      
};

int main()
{
    int n, m;
    std::cin>>n>>m;
    unordered_map<uint64_t, bool> map;
    while(n--)
    {
        uint64_t s;
        std::cin >> s;
        map[s] = true;
    }
    while(m--)
    {
        uint64_t q;
        std::cin >> q;
        std::cout << (map.find(q) ? "hit\n" : "miss\n");
    }
    return 0;
}
```

</details>

测试数据使用下面的代码生成：

<details>
<summary>点击查看代码</summary>

```
#include<iostream>
#include<random>
#include<chrono>

int main(int argc, char* argv[])
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.tie(nullptr);
    std::random_device device;
    unsigned int seed = device();
    std::mt19937 engine(seed);
    int n = atoi(argv[1]), m = 3 * n;
    std::cout << n << ' ' << m << ' ';
    while(n--)
        std::cout<< (engine() % 100000) * 126271 + (n % 2) <<' ';
    while(m--)
        std::cout<< (engine() % 100000) * 126271 + (engine()) % 100 <<' ';
    return 0;
}
```

</details>

本来这个测试数据是准备卡原生哈希 `126271` 的模数的，

