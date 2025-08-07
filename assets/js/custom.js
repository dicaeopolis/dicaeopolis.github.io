document.addEventListener('DOMContentLoaded', function() {
  const players = document.querySelectorAll('.gif-player');

  players.forEach(function(player) {
    const img = player.querySelector('img');
    const button = player.querySelector('.gif-player-button');
    const staticSrc = player.dataset.staticSrc;
    const animatedSrc = player.dataset.animatedSrc;

    // 点击事件处理
    player.addEventListener('click', function() {
      if (player.classList.contains('is-playing')) {
        // 如果当前正在播放，则停止播放并显示重播按钮
        stopGif();
      } else {
        // 如果是静止状态，则开始播放
        playGif();
      }
    });

    function playGif() {
      player.classList.add('is-playing');
      button.dataset.state = 'replay'; // 准备好重播按钮

      // 切换到动画 GIF
      // 通过在 URL 后添加时间戳来强制浏览器重新加载 GIF
      img.src = animatedSrc + '?t=' + new Date().getTime();

      // 监听图片加载完成事件
      img.onload = function() {
        // GIF 理论上加载完就开始播放了
        // 我们可以在这里做一个延迟，模拟播放结束后显示重播按钮
        // 但更简单的做法是让用户再次点击来“停止”并准备重播
      };
    }

    function stopGif() {
      player.classList.remove('is-playing');
      // 切换回静态图
      img.src = staticSrc;
    }

  });
});