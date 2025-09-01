---
comments: true
noinfo: true
short_title: DC-GAN 老婆生成器
title: DC-GAN 老婆生成器
---
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DC-GAN</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>DC-GAN 老婆生成器</h1>
        
        <p>使用 PyTorch 在 Anime Face Dataset 上训练的 DC-GAN，并迁移到 TensorFlow.js 上（这个迁移任务会遇到依赖地狱，可费了我一番功夫），实现网页端的实时推理。向 Gwern 的 <a href = "https://gwern.net/twdne">This Waifu Does Not Exist</a> 项目致敬。生成的图片有概率很歪瓜裂枣，请务必多抽几发。</p>

        <div class="canvases-container">
            <div class="canvas-wrapper">
                <canvas id="ganCanvas1" width="64" height="64"></canvas>
            </div>
            <div class="canvas-wrapper">
                <canvas id="ganCanvas2" width="64" height="64"></canvas>
            </div>
        </div>
        
        <button id="generateBtn" disabled>正在加载模型...</button>
        <br>

        <div class="progress-container">
            <h3>模型加载进度（这个进度条逗你玩的哈哈，反正等着就对了）</h3>
            <div class="progress-bar">
                <div id="modelLoadProgress" class="progress-fill"></div>
            </div>
            <span id="modelLoadPercent">0%</span>
            <div id="modelLoadTime" class="time-display">耗时: -</div>
        </div>
        
        <div class="progress-container">
            <h3>图像生成进度（没错这个进度条也是逗你玩的，不过出图还是很快的）</h3>
            <div class="progress-bar">
                <div id="imageGenProgress" class="progress-fill"></div>
            </div>
            <span id="imageGenPercent">0%</span>
            <div id="imageGenTime" class="time-display">耗时: -</div>
        </div>
        
        <div id="status">模型初始化中，也就只能等着咯...</div>
    </div>

    <!-- 引入 TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    
    <!-- 引入我们自己的脚本 -->
    <script src="app.js"></script>
</body>
</html>