---
class: 优化器
comments: true
---

# 神经网络优化器概述

## 写在前面

本文试图对 1951(SGD) 到 2024 (Muon) 的大部分主流优化器发展史做一个简单的概述。尽管“简单”，但也已经达到了上万字的规模。这是因为我并不满足于市面上大部分博客对优化器的介绍仅限于简单的罗列公式，相反，我更希望找出优化器进化的一两条贯穿整个历史长河的伏笔与线索（~~比如优化器里面有一堆 αβδη 什么的说明作者们都喜欢玩osu!mania还是段位吃~~），从而希望能够给列位看官一点启发。行笔仓促，错误在所难免，恳请大家批评指正。

本文的可视化借助了 `pytorch-optimizer` 库的 `viz_optimizers.py`，然后魔改了一下使其能适应较新的库版本，并支持生成动图。代码如下：

<details>

<summary> viz_optimizers_animated.py</summary>

```python
import glob
import math
import os
import shutil
import subprocess
import time

import matplotlib
matplotlib.use('Agg')

import imageio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from hyperopt import fmin, hp, tpe

import torch_optimizer as optim

sns.set_theme(style="whitegrid")

def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2

def rastrigin(tensor, lib=torch):
    x, y = tensor
    A = 10
    f = (A * 2 + (x**2 - A * lib.cos(x * math.pi * 2)) + (y**2 - A * lib.cos(y * math.pi * 2)))
    return f

def execute_steps(func, initial_state, optimizer_class, optimizer_config, num_iter=500):
    x = torch.Tensor(initial_state).requires_grad_(True)
    optimizer = optimizer_class([x], **optimizer_config)
    steps = np.zeros((2, num_iter + 1))
    steps[:, 0] = np.array(initial_state)
    for i in range(1, num_iter + 1):
        optimizer.zero_grad()
        f = func(x)
        f.backward(create_graph=True, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        optimizer.step()
        steps[:, i] = x.detach().numpy()
    return steps

def objective_rastrigin(params):
    optimizer_config = dict(lr=params["lr"])
    steps = execute_steps(rastrigin, (-2.0, 3.5), params["optimizer_class"], optimizer_config, 100)
    minimum = (0, 0)
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2

def objective_rosenbrock(params):
    optimizer_config = dict(lr=params["lr"])
    steps = execute_steps(rosenbrock, (-2.0, 2.0), params["optimizer_class"], optimizer_config, 100)
    minimum = (1.0, 1.0)
    return (steps[0][-1] - minimum[0]) ** 2 + (steps[1][-1] - minimum[1]) ** 2


def plot_static_image(steps, func_name, optimizer_name, lr, X, Y, Z, minimum):
    fig, ax = plt.subplots(figsize=(8, 8))
    if func_name == "rosenbrock":
        ax.contour(X, Y, Z, np.logspace(-0.5, 3.5, 20, base=10), cmap="jet")
    else:
        ax.contour(X, Y, Z, 20, cmap="jet")
    iter_x, iter_y = steps[0, :], steps[1, :]
    ax.plot(iter_x, iter_y, "r-x", label="Optimizer Path")
    ax.plot(iter_x[0], iter_y[0], 'go', markersize=10, label='Start')
    ax.plot(iter_x[-1], iter_y[-1], "rD", markersize=10, label="End")
    ax.plot(*minimum, "gD", markersize=10, label="Global Minimum")
    ax.legend()
    ax.set_title(f"{func_name.capitalize()} Function: {optimizer_name}\n{len(iter_x)-1} iterations, lr={lr:.6f}")
    output_path = f"docs/{func_name}_{optimizer_name}.png"
    plt.savefig(output_path)
    plt.close(fig)


def create_animation_with_fading_tail(
    steps, func_name, optimizer_name, lr, X, Y, Z, minimum,
    gif_resolution=256, tail_length=20, fade_length=30
):
    fig_size_inches = 8
    dpi = gif_resolution / fig_size_inches
    num_frames = steps.shape[1]
    images = []

    print(f"    - Step 1/3: Rendering {num_frames} frames into memory (with fading tail)...")
    for i in range(num_frames):
        fig, ax = plt.subplots(figsize=(fig_size_inches, fig_size_inches), dpi=dpi)
        
        if func_name == "rosenbrock":
            ax.contour(X, Y, Z, np.logspace(-0.5, 3.5, 20, base=10), cmap="jet")
        else:
            ax.contour(X, Y, Z, 20, cmap="jet")
        ax.plot(*minimum, "gD", markersize=10, label="Global Minimum")

        start_solid = max(0, i - tail_length)
        solid_path = steps[:, start_solid : i + 1]
        ax.plot(solid_path[0], solid_path[1], "r-", lw=1.5)
        ax.plot(solid_path[0], solid_path[1], "rx", markersize=4)

        start_fade = max(0, start_solid - fade_length)
        for j in range(start_solid - 1, start_fade - 1, -1):
            age = start_solid - j
            alpha = 1.0 - (age / fade_length)
            
            segment = steps[:, j : j + 2]
            ax.plot(segment[0], segment[1], color='red', lw=1.5, alpha=alpha)
        
        ax.plot(steps[0, i], steps[1, i], "rD", markersize=8, label="Current Position")
        
        ax.legend()
        ax.set_title(f"{func_name.capitalize()} Function: {optimizer_name}\nIteration: {i}/{num_frames-1}, lr={lr:.6f}")

        fig.canvas.draw()
        argb_buffer = fig.canvas.tostring_argb()
        image_argb = np.frombuffer(argb_buffer, dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_rgb = image_argb[:, :, 1:]
        images.append(image_rgb)
        plt.close(fig)

        print(f"\r      Rendered frame {i + 1}/{num_frames}", end="")
    print()

    output_path = f"gifs/{func_name}_{optimizer_name}.gif"
    
    print(f"    - Step 2/3: Creating initial GIF with imageio...")
    imageio.mimsave(output_path, images, fps=25)
    size_before = os.path.getsize(output_path) / 1024
    print(f"      Initial GIF saved ({size_before:.1f} KB).")

    print(f"    - Step 3/3: Compressing GIF with gifsicle...")
    try:
        subprocess.run(
            ["gifsicle", "-O2", "--colors", "256", "-o", output_path, output_path],
            check=True, capture_output=True, text=True
        )
        size_after = os.path.getsize(output_path) / 1024
        reduction = (1 - size_after / size_before) * 100 if size_before > 0 else 0
        print(f"      GIF compressed successfully. Size reduced by {reduction:.1f}% to {size_after:.1f} KB.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n      [WARNING] Gifsicle compression failed.")
        print("      Please ensure 'gifsicle' is installed and in your system's PATH.")
        if isinstance(e, subprocess.CalledProcessError):
            print(f"      Gifsicle stderr: {e.stderr}")


def execute_experiments(optimizers, objective, func, func_name, plot_params, initial_state, gif_config, seed=1):
    total_optimizers = len(optimizers)
    print("=" * 60)
    print(f"STARTING EXPERIMENTS FOR: {func_name.capitalize()} Function")
    print(f"Total optimizers to test: {total_optimizers}")
    print("=" * 60)
    
    if not os.path.exists("docs"): os.makedirs("docs")
    if not os.path.exists("gifs"): os.makedirs("gifs")
        
    x = np.linspace(plot_params['xlim'][0], plot_params['xlim'][1], 250)
    y = np.linspace(plot_params['ylim'][0], plot_params['ylim'][1], 250)
    X, Y = np.meshgrid(x, y)
    Z = func([X, Y], lib=np) if func_name == 'rastrigin' else func([X, Y])

    for i, item in enumerate(optimizers):
        optimizer_class, lr_low, lr_hi = item
        optimizer_name = optimizer_class.__name__
        
        print(f"\n[{i + 1}/{total_optimizers}] PROCESSING: {optimizer_name}")
        print("-" * 40)
        
        print("  1. Finding best learning rate with Hyperopt...")
        start_time = time.time()
        space = {"optimizer_class": hp.choice("optimizer_class", [optimizer_class]), "lr": hp.loguniform("lr", lr_low, lr_hi)}
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, rstate=np.random.default_rng(seed), verbose=0)
        end_time = time.time()
        print(f"    - Best LR found: {best['lr']:.6f} (search took {end_time - start_time:.2f}s)")

        print("  2. Generating full optimization path...")
        steps = execute_steps(func, initial_state, optimizer_class, {"lr": best["lr"]}, num_iter=500)
        print("    - Path generated.")

        print("  3. Creating and saving static image...")
        plot_static_image(steps, func_name, optimizer_name, best['lr'], X, Y, Z, plot_params['minimum'])
        print(f"    - Static image saved to docs/{func_name}_{optimizer_name}.png")

        print("  4. Creating and saving animated GIF with fading tail...")
        start_time = time.time()
        create_animation_with_fading_tail(
            steps, func_name, optimizer_name, best['lr'], X, Y, Z, plot_params['minimum'],
            gif_resolution=gif_config['resolution'],
            tail_length=gif_config['tail_length'],
            fade_length=gif_config['fade_length']
        )
        end_time = time.time()
        print(f"    - Animation created successfully in {end_time - start_time:.2f} seconds.")
        
        print(f"--- Finished processing {optimizer_name} ---")


def LookaheadYogi(*a, **kw):
    base = optim.Yogi(*a, **kw)
    return optim.Lookahead(base)

if __name__ == "__main__":
    GIF_CONFIG = {
        "resolution": 800,
        "tail_length": 20,
        "fade_length": 30
    }

    optimizers_to_test = [
        (torch.optim.Adamax, -8, 0.5), (torch.optim.Adagrad, -8, 0.5),
        (torch.optim.Adadelta, -8, 0.5), (torch.optim.RMSprop, -8, -2),
        (torch.optim.Rprop, -8, 0.5), (torch.optim.NAdam, -8, -1)
    ]

    plot_params_rastrigin = {'xlim': (-4.5, 4.5), 'ylim': (-4.5, 4.5), 'minimum': (0, 0)}
    execute_experiments(
        optimizers_to_test, objective_rastrigin, rastrigin, 'rastrigin', 
        plot_params_rastrigin, initial_state=(-2.0, 3.5), gif_config=GIF_CONFIG
    )

    plot_params_rosenbrock = {'xlim': (-2, 2), 'ylim': (-1, 3), 'minimum': (1.0, 1.0)}
    execute_experiments(
        optimizers_to_test, objective_rosenbrock, rosenbrock, 'rosenbrock', 
        plot_params_rosenbrock, initial_state=(-2.0, 2.0), gif_config=GIF_CONFIG
    )

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE!")
    print("Check the 'docs' directory for static images and 'gifs' for animations.")
    print("="*60)
```

</details>

同时，我也在 Fashion-MNIST 上面利用文中提到的各个优化器训练了一个简单的 CNN 模型，并画出了随 batch 的损失曲线，验证集准确率曲线等，还可视化了损失地形。代码放在[这个 Kaggle notebook](https://www.kaggle.com/code/liyanfromwhu/notebook3901731912) 上面了。

## 何以优化

神经网络的目的是在训练数据集上实现**结构风险最小化**以获得良好的拟合和泛化能力。简单说，如果我们在训练集 $X$ 上有一个定义明确的损失函数 $\mathcal{L}(X;\theta)$（表示我们的结构风险），那么所有优化器的目的都是设计一个算法来寻找合适的 $\theta$ 以获得 $\mathrm{argmin}_\theta\ \mathcal{L}(X;\theta)$。

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
