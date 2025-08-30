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
        <h1>DC-GAN 图像生成器</h1>
        <button id="generateBtn" disabled>正在加载模型...</button>
        <br>
        <div class="canvas-wrapper">
            <canvas id="ganCanvas" width="64" height="64"></canvas>
        </div>

        <div class="progress-container">
            <h3>模型加载进度</h3>
            <div class="progress-bar">
                <div id="modelLoadProgress" class="progress-fill"></div>
            </div>
            <span id="modelLoadPercent">0%</span>
            <div id="modelLoadTime" class="time-display">耗时: -</div>
        </div>
        
        <div class="progress-container">
            <h3>图像生成进度</h3>
            <div class="progress-bar">
                <div id="imageGenProgress" class="progress-fill"></div>
            </div>
            <span id="imageGenPercent">0%</span>
            <div id="imageGenTime" class="time-display">耗时: -</div>
        </div>
        
        <div id="status">模型初始化中，请稍候...</div>
    </div>

    <!-- 引入 TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    
    <!-- 引入我们自己的脚本 -->
    <script src="app.js"></script>
</body>
</html>