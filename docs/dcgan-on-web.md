<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>浏览器中的 DC-GAN</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>DC-GAN 图像生成器</h1>
        <p>所有计算都在你的浏览器中完成！点击下面的按钮生成一张新的图片。</p>
        
        <button id="generateBtn" disabled>正在加载模型...</button>
        
        <div class="canvas-wrapper">
            <canvas id="ganCanvas" width="64" height="64"></canvas>
        </div>
        
        <p id="status">模型初始化中，请稍候...</p>
    </div>

    <!-- 引入 TensorFlow.js -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    
    <!-- 引入我们自己的脚本 -->
    <script src="app.js"></script>
</body>
</html>