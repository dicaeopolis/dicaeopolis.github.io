<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DCGAN in Browser</title>
    <style>
        body { font-family: sans-serif; text-align: center; margin-top: 50px; }
        #generateBtn { font-size: 1.2em; padding: 10px 20px; cursor: pointer; }
        #imageContainer { margin-top: 20px; border: 1px solid #ccc; display: inline-block; }
        #status { color: #888; }
        canvas { display: block; }
    </style>
</head>
<body>
    <h1>DCGAN Image Generator</h1>
    <p>点击按钮，在浏览器中生成一张新图片！</p>
    <button id="generateBtn">生成新图片</button>
    <p id="status">正在加载模型...</p>
    <div id="imageContainer">
        <canvas id="ganCanvas" width="64" height="64"></canvas>
    </div>

    <!-- 引入 TensorFlow.js 库 -->
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
    
    <!-- 引入我们自己的脚本 -->
    <script src="app.js"></script>
</body>
</html>