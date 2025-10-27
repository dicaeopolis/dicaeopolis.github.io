---
password: UUID('1afe080f-b280-11f0-902e-00155dff0634')
---

# MISC-AI 方向 WriteUp

## Problem 1

操作指南：本题为签到题，按下面指南操作即可。本题目鼓励 Vibe coding（人工智能辅助编程），但在提交 WP 时请仔细阅读相关规定！

1. 在下发的 mnist_cnn_weights.pth 文件里面找到密钥。
2. 分析这个模型的结构，基于 modified_mnist 里面的数据，训练一个新模型。
3. 使用这个新模型对 imgs/ 目录下的图片按顺序进行推理，推理的结果拼接成一个字符串。
4. 将这个字符串作为一个十进制数，转成 16 进制数，这就是密文。
5. 根据密钥解密密文，得到 flag。加密方式是一种非常简单的加密。

### 出题人吐槽

本题目设计出来就是让大家用 ai 梭的，不过不管大伙用什么大模型，总是能遇到一些共性问题呢：

- 不知道利用我修改后的数据。网页端的 Chat 模型肯定只能猜，**喜欢猜是 RL 大幅提升推理性能带来的反噬**，猜的后果就是很多人做出来的结果是 42…… 开头，这就撞到我预设的卡点了，因为修改后的数据是做了标签置换后的 MNIST。如果用 CLINE 等 Agent 插件会好一些。
- 顺序问题。**我觉得人看到“按顺序进行推理”几个字，结合 imgs/ 文件夹下的 img_index_0, img_index_1, ... 肯定会理解成是自然数顺序从 0 数到 100。** 但是很可惜有许多 GPT 们选择一个简单的字典序，也就是 0, 1, 10, 100, 11, ... 不过我后面也澄清了一下，帮大家写出更好的 prompts，毕竟这不应该成为一个卡点。

不过这其实体现了 LLM 的好几个局限性：

- 由于“不具身”无法访问数据，而 RL 又鼓励其过度猜测（其实是 RL 指标没有搞好 **exploration vs. exploitation**，万一猜能猜对呢，这就有利于刷 SOTA 了，参考 [OpenAI 的这个研究](https://arxiv.org/abs/2509.04664)），因此结果只能收敛到最平滑或者说最常见的分布，因此如果要在这个点上面卡人的话，换一个不常见的架构，比如 RBF 网络等，就行了。
- 不同 LLM 之间共用数据和相互蒸馏现象严重。以上提到的两个问题在不同厂家的 LLM 都出现了，或许以后我们甚至无需考虑 LLM 攻击的可迁移性，因为它们之间已经没有什么差别了，甚至我们现在已经可以看到这个趋势了。

### EXP

基本上就是文字描述的代码化：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image

# --- 环境配置与模型定义 ---
device = torch.device("cpu")
num_epochs = 1 

class PyTorch_MNIST_CNN(nn.Module):
    def __init__(self):
        super(PyTorch_MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout(0.25)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x) 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x) 
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x) 
        x = self.fc2(x)
        return x

from torchvision import transforms

# 用torchvision原生方法加载修改后的数据集
SAVE_DIR = './modified_mnist'
modified_train = datasets.MNIST(
    root=SAVE_DIR,  # 这里用我们保存的目录
    train=True,
    download=False,  # 已手动生成，无需下载
    transform=transforms.ToTensor()
)
modified_test = datasets.MNIST(
    root=SAVE_DIR,
    train=False,
    download=False,
    transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(modified_train, batch_size=64, shuffle=True)

# --- 训练逻辑 ---

model = PyTorch_MNIST_CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

print("\n开始模型训练 (1 Epoch)...")
train(model, device, train_loader, optimizer, criterion, num_epochs)
print("模型训练完成!")

def predict_label_from_path(
    model: nn.Module, 
    image_path: str, 
    device: torch.device
) -> int:
    """
    加载本地图片文件，进行预处理，并通过训练好的模型进行推理，
    返回模型预测的置换后标签。

    Args:
        model: 训练好的 PyTorch 模型 (已使用置换标签训练)。
        image_path: 本地图片文件的完整路径 (str)。
        device: 模型所在的设备 (e.g., torch.device("cpu"))。

    Returns:
        模型预测的标签 (int)，如果失败则返回 -1。
    """
    
    if not os.path.exists(image_path):
        print(f"错误：文件路径不存在: {image_path}")
        return -1
    
    # 1. 定义与训练时相同的预处理步骤
    # 注意：MNIST 使用的是 (0.1307,) 和 (0.3081,)
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 确保输入是单通道
        transforms.Resize((28, 28)),                # 确保尺寸匹配
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        # 2. 加载图片
        # 使用 L 模式加载灰度图
        image = Image.open(image_path).convert('L')
        
        # 3. 预处理
        image_tensor = preprocess(image)
        
        # 4. 增加 Batch 维度并移动到设备
        # 模型的输入需要是 [Batch_size, Channels, Height, Width]
        image_input = image_tensor.unsqueeze(0).to(device)
        
        # 5. 模型推理
        model.eval()
        with torch.no_grad():
            output = model(image_input)
            
            # 获取概率最大的索引，即预测标签
            predicted_label = output.argmax(dim=1).item()
            
            return predicted_label
            
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        return -1

# 对imgs/下面的所有图片进行推理，路径格式是 imgs/img_index_<i>.png
li = []
for i in range(101):
    # 一堆 GPT 把这个理解成 sort 了，没想到一个 for-range 循环就能搞定……
    image_path = f'imgs/img_index_{i}.png'
    predicted_label = predict_label_from_path(model, image_path, device)
    li.append(predicted_label)

num = 0
for i in li:
    num = num * 10 + i
print(hex(num)[2:].upper())
text_hex = hex(num)[2:].upper() # 十六进制密文

def xor_crypt(data, key, is_encrypt=True):
    """
    异或加密/解密函数
    :param data: 输入数据（加密时为明文字符串；解密时为Hex格式密文字符串）
    :param key: 异或密钥（字符串）
    :param is_encrypt: True为加密，False为解密
    :return: 加密返回Hex字符串，解密返回明文字符串
    """
    key_bytes = key.encode('utf-8')  # 密钥转字节
    key_len = len(key_bytes)
    result = bytearray()  # 存储异或结果的字节数组

    if is_encrypt:
        # 【加密】输入是明文字符串，转字节
        data_bytes = data.encode('utf-8')
    else:
        # 【解密】输入是Hex字符串，转字节
        data_bytes = bytes.fromhex(data)

    # 循环异或：数据每个字节与密钥对应字节（密钥循环复用）异或
    for i, b in enumerate(data_bytes):
        key_byte = key_bytes[i % key_len]  # 循环取密钥的字节
        result.append(b ^ key_byte)       # 异或运算

    if is_encrypt:
        # 【加密输出】字节转Hex字符串
        return result.hex()
    else:
        # 【解密输出】字节转UTF-8字符串
        return result.decode('utf-8', errors='ignore')

#异或解密
key = 'CIALLO0d000721998244353ISAPRIME' # 文本密钥
flag = xor_crypt(text_hex, key, is_encrypt=False)
print(f"Flag: {flag}")
```

没错我也是 vibe 的，但是 vibe 的前提是，得看得懂逻辑，以及出现张量形状不匹配或者给非叶子张量做操作等报错时，知道怎么处理。

## Problem 2

实测真正做这个题的人应该在 10 个人左右，不过我寻思这个题很简单啊。

不过让 AI 来伪装 shell 的想法很好玩，我是受到[这个文章](https://arxiv.org/abs/2409.08234)的启发。

### 考察点

本题就是大家最喜欢玩的**提示词攻击**，本来是也是想搜集一下大家有没有什么可玩性比较高的提示词，结果太可惜了没几个人做……

如果大家有想玩 LLM 安全的，下面的题解算是初步提供一个小小武器库吧，**请务必在合规范围内使用**。

### 提示词大赏

目前最简单的 prompt 由 misay 师傅做出：

![alt text](b7e5f15adbc44836b6073c8191808c45.png)

PJA 师傅用 Gemini 搓了个绕过，也很不错：

![alt text](image-5.png)

<details class = "warning">
<summary>有害内容警告</summary>

<p></p>
<p>下面的内容涉及到对大语言模型的提示词越狱攻击，可能包含不恰当内容，如暴力危险言论等。这些内容仅供学术交流使用。</p>
<p></p>

</details>

从 [这里](https://linux.do/t/topic/403782) 摘录的一段提示词，Deepseek-DAN：

![alt text](image-3.png)

经典猫娘提示词：

![alt text](9b3180ebc13a7e6a37593379232006af.png)

我自己很喜欢的一篇工作，[FlipAttack](https://arxiv.org/abs/2410.02832)，一年了这个方法还能奏效，说明 LLM 范式是有固有弱点的：

![alt text](image-4.png)

经典的 Untammed Writing Assistant 提示词，虽然模型拒绝了，但是还是泄露了内容。

![alt text](image-6.png)

[还挺好用的万能套取 prompt](https://linux.do/t/topic/209856)：

![alt text](image-7.png)

## Problem 3

本来这个位置应该放另外一道题的，但是由于 push 到比赛环境上面可能出了些问题导致没有回显一直修不好，就在放题当天凌晨加班到 3 点过花了 2h 出出来了，结果导致当天白天院运会志愿者直接睡过头……

题目结合背景其实很简单，考的是**后门攻击**的防御，也就是对后门触发器的逆向。考虑到题目说法可能还是有点隐晦，反正我自己试了几个 LLM 发现它们都没能读出后门攻击的味道反而一直在纠结题目背景的自动驾驶。不过最后我已经把关键词贴出来了，搜索一下照着做就行嘛。**为什么这些题出在 misc? 就是考验搜索能力。如果一道 misc 题目可以轻松利用随波逐流等工具“梭”出来，那么这样的题目（入门教学题目不算哈）质量肯定不高。**

顺着我给的提示，或者让 GPT5-high 等聪明模型来做就会找到 [这篇文章](https://ieeexplore.ieee.org/document/8835365)，然后喂给 LLM 直接 vibe 即可。

下面的 WP 就是 deepseek 帮我写的：

### WP

每个模型都被植入了后门（backdoor），当输入图像包含特定触发器（trigger）时，模型会错误分类到目标数字。触发器是一个小的扰动模式，通常人眼难以察觉。

Neural Cleanse是一种后门检测方法，其核心思想是：对于一个被植入后门的模型，将其他类别的输入误分类到目标类别所需的触发器是最小的。因此，我们可以通过逆向工程为每个候选数字（0-9）生成触发器，并比较触发器的大小，从而找出目标数字。

**关键步骤**：

1. 对于每个候选数字 \( c \)（0到9），我们优化一个触发器 \( \delta \)，使得当触发器添加到干净图像上时，模型将图像分类为 \( c \)。
2. 优化目标是最小化分类损失和触发器的L1范数（鼓励触发器稀疏）。
3. 目标数字 \( c^* \) 是所需触发器L1范数最小的那个。

#### 步骤1: 环境准备

确保安装以下Python库：

```bash
pip install torch torchvision pycryptodome
```

#### 步骤2: 确定模型数量

从题目附件中，查看模型文件的数量。假设你收到多个模型文件（如 `model_0.pth`, `model_1.pth`, ...），模型数量就是文件的数量。记模型数量为 N。

#### 步骤3: 加载模型并分析架构

模型是PyTorch保存的权重文件（.pth）。由于架构未知，我们需要从模型中推断。通常，MNIST分类模型是CNN。以下代码演示如何加载模型并检查架构：

```python
import torch
import torch.nn as nn

# 尝试加载第一个模型以推断架构
state_dict = torch.load('model_0.pth', map_location='cpu')

# 打印状态字典的键以了解层结构
print("State dict keys:", state_dict.keys())

# 这里提供一个通用MNIST CNN架构
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 加载模型权重
model = MNISTClassifier()
model.load_state_dict(state_dict)
model.eval()
```

注意：如果架构不匹配，可能需要调整模型定义。你可以通过查看状态字典的形状来推断层大小。

#### 步骤4: 准备测试数据

使用MNIST测试集作为干净图像来源：

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
```

#### 步骤5: 实现Neural Cleanse检测

对于每个模型，执行以下代码来检测目标数字：

```python
import torch.optim as optim
import numpy as np

def detect_target_class(model, test_loader, device='cpu'):
    """
    使用Neural Cleanse检测模型的目标数字
    返回目标数字（0-9）
    """
    model.to(device)
    model.eval()
    l1_norms = []  # 存储每个候选数字的触发器L1范数
    
    # 选择5个干净图像作为基础（来自不同类别）
    clean_images = []
    selected_classes = set()
    with torch.no_grad():
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                if lbl.item() not in selected_classes and len(clean_images) < 5:
                    clean_images.append(img.to(device))
                    selected_classes.add(lbl.item())
                if len(clean_images) >= 5:
                    break
            if len(clean_images) >= 5:
                break
    
    # 对每个候选数字（0-9）反演触发器
    for candidate in range(10):
        triggers = []  # 存储每个基础图像的触发器
        for base_img in clean_images:
            base_img = base_img.unsqueeze(0)  # 添加batch维度
            # 初始化触发器（可训练参数）
            trigger = torch.zeros_like(base_img, requires_grad=True, device=device)
            # 添加小幅噪声初始化
            trigger.data = torch.randn_like(trigger) * 0.01
            
            optimizer = optim.Adam([trigger], lr=0.05)
            # 优化循环
            for step in range(200):  # 200次迭代
                # 添加触发器并裁剪到有效范围
                poisoned_img = torch.clamp(base_img + trigger, 0, 1)
                outputs = model(poisoned_img)
                # 损失函数：分类损失 + L1正则化
                loss = nn.CrossEntropyLoss()(outputs, torch.tensor([candidate], device=device))
                loss += 0.1 * torch.norm(trigger, p=1)  # L1正则化系数0.1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            triggers.append(trigger.detach().cpu())
        
        # 计算平均触发器的L1范数
        avg_trigger = torch.mean(torch.stack(triggers), dim=0)
        l1_norm = torch.norm(avg_trigger, p=1).item()
        l1_norms.append(l1_norm)
        print(f"候选数字 {candidate} 的触发器L1范数: {l1_norm:.4f}")
    
    # 目标数字是L1范数最小的候选
    target_class = np.argmin(l1_norms)
    print(f"检测到的目标数字: {target_class}")
    return target_class

# 遍历所有模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_sequence = []
for i in range(N):  # N是模型数量
    print(f"处理模型 {i}...")
    model = MNISTClassifier()
    model.load_state_dict(torch.load(f'model_{i}.pth'))  # 根据实际文件名调整
    target_digit = detect_target_class(model, test_loader, device)
    target_sequence.append(str(target_digit))

# 拼接数字序列
digit_string = ''.join(target_sequence)
print(f"目标数字序列: {digit_string}")
```

#### 步骤6: 计算SHA256并解密

使用目标数字序列计算SHA256哈希，并解密密文，这里选一个在线网站解密即可。

注意事项：

1. **模型架构**：如果提供的模型架构与代码中的`MNISTClassifier`不匹配，你需要根据状态字典调整模型定义。查看状态字典的键和形状来构建正确架构。
2. **计算时间**：Neural Cleanse检测每个模型可能需要几分钟（取决于硬件）。使用GPU可以加速。
3. **触发器优化**：优化参数（学习率、迭代次数、L1系数）可能需要微调以获得最佳结果。如果检测不准，可以增加迭代次数或调整L1系数。
4. **模型数量**：确保正确统计模型文件数量，顺序通常按文件名排序（如`model_0.pth`, `model_1.pth`, ...）。

通过以上步骤，你可以找出每个模型的目标数字，拼接后计算SHA256密钥，并解密密文获得flag。本题展示了后门攻击的检测方法，强调了AI模型安全的重要性。如果你在实施过程中遇到问题，可以检查模型架构或调整优化参数。

## 碎碎念

从迎新赛开始就有了“梭”的风气。不可避免地，我们应该承认 LLM 越来越聪明了。但似乎 LLM 的智力和用 LLM 的人的智力在很多时候是守恒的。我看到好多人对着 AI 1 的指示咔咔一顿炼丹出 flag，然后看着 AI 2 的一个**真实世界的复杂问题**：nc 连不上，而发呆求助。**这并不是一个好现象**。固然我们不可能反对 LLM 浪潮开历史倒车，但我们的比赛是面向**人**的。AI 会写出结构非常清晰，注释丰富逻辑完整的代码，但至少现在的 AI 看到 base100 编码时仍然会选择老是在 emoji 的具体内容上面钻牛角尖，编造一个警察抓情侣的故事——而如果**人**能够设计一个合理的 prompt，并辅以搜索功能，是可以正常做出的。我们希望的是这样的 out-of-distribution：用 LLM 的人能够和 LLM 一起智力进步，而不是**电脑控制大脑，LLM 代替思考**。

扯到这了，还得再提一个点。**使用者即负责人**——人要对自己利用 AI 生成的内容负责。怎么个负责法呢？AI 写了段脚本，得清楚它都干了些啥，你才**敢**运行吧？如果你不懂，大可以再开一个对话问问吧。AI 梭题固然快，但你提交之前**能不能先看看这玩意是不是瞎编的啊？？？**我们后台看到这些富有想象力的 flag 们快笑疯了……

这两点总结起来还是那句话：**你尽力了吗？**或者说，**人类**尽力了吗？总之，重申一遍：不要让**电脑控制大脑，LLM 代替思考**。
