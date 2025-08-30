// app.js

// --- 核心参数 (必须和Python导出时一致) ---
const LATENT_DIM = 128;
const MODEL_URL = './web_model/model.json'; // 相对路径指向模型文件

// --- 获取DOM元素 ---
const generateBtn = document.getElementById('generateBtn');
const canvas = document.getElementById('ganCanvas');
const statusDiv = document.getElementById('status');
const ctx = canvas.getContext('2d');

let model = null;

/**
 * 异步加载 TensorFlow.js 模型
 */
async function loadModel() {
    try {
        model = await tf.loadGraphModel(MODEL_URL, {
            onProgress: (fraction) => {
                const percentage = (fraction * 100).toFixed(0);
                statusDiv.textContent = `模型文件下载中... ${percentage}%`;
            }
        });

        // 预热模型，让第一次生成更快
        tf.tidy(() => {
            model.predict(tf.zeros([1, LATENT_DIM, 1, 1]));
        });
        
        console.log("模型加载并预热成功！");
        statusDiv.textContent = '模型已就绪！';
        generateBtn.disabled = false;
        generateBtn.textContent = '生成新图片';

    } catch (error) {
        console.error("模型加载失败:", error);
        statusDiv.textContent = '模型加载失败，请检查浏览器控制台。';
        generateBtn.textContent = '加载失败';
    }
}

/**
 * 使用模型生成图片并渲染到 Canvas
 */
async function generateImage() {
    if (!model) {
        alert("模型尚未加载完成！");
        return;
    }
    
    statusDiv.textContent = '正在生成图片...';
    generateBtn.disabled = true;

    // 异步执行以避免UI阻塞
    await new Promise(resolve => setTimeout(resolve, 10));

    // 使用 tf.tidy() 自动管理内存，防止 WebGL 内存泄漏
    tf.tidy(() => {
        // 1. 生成随机噪声输入张量
        // 形状为 [batch_size, latent_dim, 1, 1]
        const noise = tf.randomNormal([1, LATENT_DIM, 1, 1]);

        // 2. 模型推理
        // outputTensor 的形状是 [1, 3, 64, 64] (batch, channels, height, width)
        const outputTensor = model.predict(noise);

        // 3. 处理输出张量
        // a. 将数值范围从 [-1, 1] (Tanh 输出) 转换到 [0, 1]
        const normalizedTensor = outputTensor.add(1).div(2);

        // b. 调整维度顺序以适配浏览器渲染
        // tf.browser.toPixels 需要的格式是 [height, width, channels]
        // 所以我们需要从 [1, 3, 64, 64] -> [3, 64, 64] -> [64, 64, 3]
        const imageTensor = normalizedTensor.squeeze([0]).transpose([1, 2, 0]);

        // 4. 将张量绘制到 Canvas
        tf.browser.toPixels(imageTensor, canvas);
        
        console.log("图片生成完毕！");
    });
    
    statusDiv.textContent = '图片已生成！点击按钮可再次生成。';
    generateBtn.disabled = false;
}

// --- 程序入口 ---
document.addEventListener('DOMContentLoaded', () => {
    generateBtn.addEventListener('click', generateImage);
    loadModel();
});