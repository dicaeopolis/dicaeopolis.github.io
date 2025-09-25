<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>二进制乘法可视化工具</title>
<style> .md-sidebar.md-sidebar--secondary { display: none !important; } </style>
<style>
  :root{
    --primary-color:#3498db;
    --secondary-color:#2980b9;
    --accent-color:#ff7e5f;
    --dark-color:#2c3e50;
    --success-color:#2ecc71;
    --warning-color:#f39c12;
    --danger-color:#e74c3c;
    --bit-size:28px;
    --gap-size:4px;
    --steps-height:320px;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;
    max-width:1200px;margin:0 auto;padding:20px;
    background:#f9f9f9;color:#333;line-height:1.6;
  }
  h1,h2,h3{color:var(--dark-color);margin-bottom:15px}
  .header{
    text-align:center;margin-bottom:30px;padding:20px;
    background:linear-gradient(135deg,var(--primary-color),var(--secondary-color));
    color:#fff;border-radius:10px;box-shadow:0 4px 15px rgba(0,0,0,.1)
  }
  .header p{max-width:800px;margin:0 auto;font-size:1.05em}
  .container{
    background:#fff;border-radius:10px;padding:18px;
    box-shadow:0 2px 15px rgba(0,0,0,.05);margin-bottom:18px
  }
  .control-panel{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
    gap:14px;margin-bottom:10px
  }
  .input-group{margin-bottom:8px}
  label{display:block;margin-bottom:6px;font-weight:600;color:var(--dark-color)}
  select,input{
    width:100%;padding:10px 12px;border:2px solid #ddd;border-radius:6px;font-size:16px;transition:border-color .3s
  }
  select:focus,input:focus{border-color:var(--primary-color);outline:none}
  .btn-group{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
  button{
    padding:10px 16px;border:none;border-radius:6px;font-size:15px;font-weight:600;cursor:pointer;
    transition:all .3s;display:inline-flex;align-items:center;justify-content:center
  }
  button i{margin-right:8px}
  .btn-primary{background:var(--primary-color);color:#fff}.btn-primary:hover{background:var(--secondary-color)}
  .btn-success{background:var(--success-color);color:#fff}.btn-success:hover{background:#27ae60}
  .btn-warning{background:var(--warning-color);color:#fff}.btn-warning:hover{background:#e67e22}
  .btn-danger{background:var(--danger-color);color:#fff}.btn-danger:hover{background:#c0392b}
  .btn-light{background:#eef2f7;color:#2c3e50;border:1px solid #d0d7de}.btn-light:hover{background:#e4e9ef}
  button:disabled{background:#bdc3c7;cursor:not-allowed;opacity:.7}
  .explanation{
    background:#e8f4f8;border-left:4px solid var(--primary-color);padding:14px;margin:12px 0;border-radius:0 8px 8px 0;font-size:15px
  }
  .process-layout{
    display:grid;grid-template-columns:minmax(0,1.7fr) minmax(280px,1fr);
    gap:16px;align-items:start
  }
  .left-pane,.right-pane{min-width:0}
  .registers-container{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));
    gap:12px;margin-bottom:12px
  }
  .register-group{background:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;padding:10px}
  .register-group.wide{grid-column:1 / -1}
  .register-title{
    font-weight:bold;margin-bottom:10px;color:var(--dark-color);font-size:16px;display:flex;align-items:center
  }
  .register-title i{margin-right:8px;color:var(--primary-color)}
  .bits-container{display:flex;flex-wrap:wrap;gap:var(--gap-size);margin-bottom:6px}
  .bits-line{display:flex;gap:var(--gap-size)}
  .bit{
    width:var(--bit-size);height:var(--bit-size);display:inline-flex;align-items:center;justify-content:center;
    border:1px solid #e0e0e0;border-radius:6px;font-family:monospace;font-weight:700;font-size:14px;background:#fff;transition:opacity .35s ease, transform .35s ease;user-select:none
  }
  .bit-0{background:#f0f8ff;color:#3498db}
  .bit-1{background:#e3f2fd;color:#e74c3c;font-weight:800}

  /* 加法后变动位：淡入动画 */
  .bit-fade-in{opacity:0;transform:scale(0.96)}
  .bit-fade-in.ready{opacity:1;transform:scale(1)}

  /* 结果寄存器：两半着色，位宽变化自动对应 */
  .result-row{
    position:relative;padding:6px;border:1px solid #dfe7f1;border-radius:6px;overflow:hidden;
  }
  .result-bits{display:flex;gap:0}
  .half{display:flex;gap:var(--gap-size)}
  .half-A{background:#eef6ff}
  .half-Q{background:#fff2ef;margin-left:var(--gap-size)}
  /* 让半区背景紧贴比特块 */
  .half .bit{background:transparent}

  /* 移位动画覆盖层 */
  .shift-overlay{position:absolute;left:6px;right:6px;top:6px;bottom:6px;pointer-events:none}
  .ghost-line{display:flex;gap:var(--gap-size)}
  .shift-ghost{transform:translateX(0);transition:transform .45s ease;opacity:.9}
  .shift-ghost.move{transform:translateX(calc(var(--bit-size) + var(--gap-size)))}
  .shift-enter{
    position:absolute;left:6px;top:6px;width:var(--bit-size);height:var(--bit-size);
    opacity:0;transform:translateX(calc(-1 * (var(--bit-size) + var(--gap-size))));
    animation:enterFromLeft .45s ease-out .05s forwards
  }
  .shift-exit{animation:fadeOut .40s linear .35s forwards}
  @keyframes enterFromLeft{to{transform:translateX(0);opacity:1}}
  @keyframes fadeOut{to{opacity:0}}

  .register-value{
    text-align:left;font-family:monospace;font-size:14px;margin-top:6px;padding:6px;background:#f1f8e9;border-radius:4px;color:#33691e
  }
  .register-labels{display:flex;justify-content:space-between;margin:2px 0 6px;font-size:13px;color:#7f8c8d}

  .inline-chips{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
  .chip{
    display:inline-flex;align-items:center;justify-content:center;padding:4px 10px;border-radius:999px;background:#eef2f7;border:1px solid #d0d7de;
    font-family:monospace;font-weight:700;font-size:14px;color:#2c3e50
  }
  .chip.small{padding:3px 8px;font-size:13px}

  /* 右侧步骤：固定高度，底部对齐视口底部 */
  .steps-card{
    background:#f8f9fa;border:1px solid #e9ecef;border-radius:8px;padding:12px;height:var(--steps-height);
    position:sticky;bottom:16px;display:flex;flex-direction:column
  }
  .steps-title{font-weight:bold;color:var(--dark-color);display:flex;align-items:center;margin-bottom:8px;font-size:16px}
  .steps-title i{margin-right:8px;color:var(--primary-color)}
  .steps-container{flex:1;overflow:auto}
  .step{padding:10px 12px;border-left:4px solid var(--primary-color);background:#fff;border-radius:0 8px 8px 0;font-size:15px;line-height:1.65}
  .step h4{margin-bottom:8px;font-size:16px}
  .step p{margin:6px 0}

  .result{
    padding:14px;background:#e8f5e9;border-radius:8px;margin-top:14px;text-align:center;font-size:16px;font-weight:bold;color:#2e7d32;
    box-shadow:0 2px 10px rgba(0,0,0,.05)
  }
  .result-error{background:#ffebee;color:#c62828}
  .animation-controls{display:flex;align-items:center;gap:12px;margin-top:8px}
  .speed-control{display:flex;align-items:center}
  .speed-control label{margin-right:8px;margin-bottom:0}
  .btn-group-compact{display:flex;gap:8px;flex-wrap:wrap}
  .btn-group-compact button{flex:1 1 120px}
  @media (max-width:992px){.process-layout{grid-template-columns:1fr}}
  @media (max-width:768px){
    .control-panel{grid-template-columns:1fr}
    .btn-group{flex-direction:column}
    button{width:100%}
    .bits-container{gap:3px}
    .bit{width:26px;height:26px;font-size:13px}
  }
</style>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"/>
</head>
<body>
  <div class="header">
    <h1><i class="fas fa-calculator"></i> 二进制乘法可视化工具</h1>
    <p>支持无符号乘法、原码乘法与 Booth（补码）乘法。支持单步、回退与自动播放。</p>
  </div>

  <div class="container">
    <h2><i class="fas fa-cog"></i> 参数设置</h2>
    <div class="control-panel">
      <div class="input-group">
        <label for="multiplicationType"><i class="fas fa-clipboard-list"></i> 乘法类型:</label>
        <select id="multiplicationType">
          <option value="unsigned">无符号乘法</option>
          <option value="signMagnitude">原码乘法</option>
          <option value="booth">补码乘法 (Booth 算法)</option>
        </select>
      </div>
      <div class="input-group">
        <label for="bitLength"><i class="fas fa-ruler-combined"></i> 位数:</label>
        <select id="bitLength">
          <option value="4">4位</option>
          <option value="8" selected>8位</option>
          <option value="16">16位</option>
        </select>
      </div>
      <div class="input-group">
        <label for="operand1"><i class="fas fa-hashtag"></i> 被乘数 (十进制):</label>
        <input type="number" id="operand1" value="5"/>
      </div>
      <div class="input-group">
        <label for="operand2"><i class="fas fa-hashtag"></i> 乘数 (十进制):</label>
        <input type="number" id="operand2" value="3"/>
      </div>
    </div>

    <div class="btn-group">
      <button id="initializeBtn" class="btn-primary"><i class="fas fa-play"></i> 初始化计算</button>
      <button id="prevBtn" class="btn-light" disabled><i class="fas fa-step-backward"></i> 上一步</button>
      <button id="stepBtn" class="btn-success" disabled><i class="fas fa-step-forward"></i> 下一步</button>
      <button id="autoBtn" class="btn-warning" disabled><i class="fas fa-play-circle"></i> 自动播放</button>
      <button id="resetBtn" class="btn-danger"><i class="fas fa-redo"></i> 重置</button>
    </div>

    <div class="animation-controls">
      <div class="speed-control">
        <label for="animationSpeed"><i class="fas fa-tachometer-alt"></i> 动画速度:</label>
        <select id="animationSpeed">
          <option value="1000">慢</option>
          <option value="500" selected>中</option>
          <option value="200">快</option>
        </select>
      </div>
    </div>
  </div>

  <div class="container">
    <h2><i class="fas fa-calculator"></i> 计算过程</h2>
    <div class="process-layout">
      <div class="left-pane">
        <div class="explanation" id="explanation">
          <p><i class="fas fa-info-circle"></i> 点击“初始化计算”开始。使用“下一步/上一步”逐步查看。右侧只显示当前一步的说明。</p>
        </div>

        <h3><i class="fas fa-microchip"></i> 寄存器状态</h3>
        <div class="registers-container" id="registers"></div>

        <div class="result" id="result"></div>
      </div>

      <div class="right-pane">
        <div class="steps-card">
          <div class="steps-title"><i class="fas fa-list-ol"></i> 步骤说明（仅当前一步）</div>
          <div class="steps-container" id="steps"></div>
        </div>
      </div>
    </div>
  </div>

<script>
  // 全局状态
  let currentStep = 0;
  let multiplication = null;
  let autoInterval = null;

  // DOM
  const multiplicationTypeEl = document.getElementById('multiplicationType');
  const bitLengthEl = document.getElementById('bitLength');
  const operand1El = document.getElementById('operand1');
  const operand2El = document.getElementById('operand2');
  const initializeBtn = document.getElementById('initializeBtn');
  const prevBtn = document.getElementById('prevBtn');
  const stepBtn = document.getElementById('stepBtn');
  const autoBtn = document.getElementById('autoBtn');
  const resetBtn = document.getElementById('resetBtn');
  const explanationEl = document.getElementById('explanation');
  const registersEl = document.getElementById('registers');
  const stepsEl = document.getElementById('steps');
  const resultEl = document.getElementById('result');
  const animationSpeedEl = document.getElementById('animationSpeed');

  class BinaryMultiplication {
    constructor(type, bitLength, operand1, operand2) {
      this.type = type;
      this.n = parseInt(bitLength);
      this.operand1 = parseInt(operand1);
      this.operand2 = parseInt(operand2);
      this.steps = [];
      this.valid = true;

      this.initializeRegisters();

      if (!this.validateInputs()) {
        this.valid = false;
        this.addStep("系统检测到参数超出当前位宽可表示的范围。请调整位宽或数值后重新初始化。");
        return;
      }

      if (type === 'unsigned') this.unsignedMultiplication();
      else if (type === 'signMagnitude') this.signMagnitudeMultiplication();
      else if (type === 'booth') this.boothMultiplication();
    }

    validateInputs() {
      const maxUnsigned = Math.pow(2, this.n) - 1;
      const minSigned = -Math.pow(2, this.n - 1);
      const maxSigned = Math.pow(2, this.n - 1) - 1;
      if (this.type === 'unsigned') {
        if (this.operand1 < 0 || this.operand1 > maxUnsigned || this.operand2 < 0 || this.operand2 > maxUnsigned) {
          alert(`对于 ${this.n} 位无符号乘法，操作数必须在 0 到 ${maxUnsigned} 之间`);
          return false;
        }
      } else {
        if (this.operand1 < minSigned || this.operand1 > maxSigned || this.operand2 < minSigned || this.operand2 > maxSigned) {
          alert(`对于 ${this.n} 位有符号乘法，操作数必须在 ${minSigned} 到 ${maxSigned} 之间`);
          return false;
        }
      }
      return true;
    }

    initializeRegisters() {
      if (this.type === 'unsigned') {
        this.A = Array(this.n).fill(0);
        this.Q = this.toUnsignedBinary(this.operand2);
        this.M = this.toUnsignedBinary(this.operand1);
        this.C = 0;
        this.counter = this.n;
        this.addStep(
          "系统把 A 置为全 0，把 Q 写入乘数的二进制表示，把 M 写入被乘数的二进制表示，把 C 置为 0，并把计数器置为位宽 n。",
          { A: this.A.join(''), Q: this.Q.join(''), M: this.M.join(''), C: this.C, counter: this.counter }
        );
      } else if (this.type === 'signMagnitude') {
        this.sign = (this.operand1 < 0 ? 1 : 0) ^ (this.operand2 < 0 ? 1 : 0);
        this.A = Array(this.n).fill(0);
        this.Q = this.toUnsignedBinary(Math.abs(this.operand2));
        this.M = this.toUnsignedBinary(Math.abs(this.operand1));
        this.counter = this.n;
        this.addStep(
          "系统把 A 置为全 0，把 Q 写入 |乘数| 的二进制表示，把 M 写入 |被乘数| 的二进制表示，并把计数器置为 n。系统同时计算结果的符号位（被乘数符号 XOR 乘数符号）。",
          { A: this.A.join(''), Q: this.Q.join(''), M: this.M.join(''), sign: this.sign, counter: this.counter }
        );
        this.addStep(`系统计算结果符号位：被乘数符号 ${this.operand1 < 0 ? 1 : 0} XOR 乘数符号 ${this.operand2 < 0 ? 1 : 0}，得到 ${this.sign}。`);
      } else if (this.type === 'booth') {
        this.A = Array(this.n).fill(0);
        this.Q = this.toTwosComplement(this.operand2);
        this.M = this.toTwosComplement(this.operand1);
        this.Q_minus1 = 0;
        this.counter = this.n;
        this.M_minus = this.toTwosComplement(-this.operand1);
        this.addStep(
          "系统把 A 置为全 0，把 Q 写入乘数的补码表示，把 M 写入被乘数的补码表示，把 Q_{-1} 置为 0，并把计数器置为 n。系统预先计算 −M 的补码以便使用。",
          { A: this.A.join(''), Q: this.Q.join(''), M: this.M.join(''), "M_minus": this.M_minus.join(''), "Q_-1": this.Q_minus1, counter: this.counter }
        );
      }
    }

    // 无符号乘法
    unsignedMultiplication() {
      this.addStep("系统开始无符号乘法。每一轮中，系统读取 Q 的最低位 Q0；如果 Q0 为 1，系统把 M 加到 A，并把产生的进位写入 C。随后，系统把 (C, A, Q) 看作连续寄存器并整体右移一位。");

      while (this.counter > 0) {
        const stepNum = this.n - this.counter + 1;
        this.addStep(`第 ${stepNum} 轮：系统读取 Q 的最低位 Q0 = ${this.Q[this.n - 1]}。如果 Q0 为 1，系统把 M 加到 A；如果 Q0 为 0，系统跳过加法。`);

        if (this.Q[this.n - 1] === 1) {
          this.addStep("系统执行加法：A ← A + M。系统把溢出的进位写入 C，并只保留 A 的低 n 位。");
          this.addBinaryArrays();
          this.addStep(`加法结束：此时 A = ${this.A.join('')}，C = ${this.C}。`);
        }

        this.addStep("系统把 (C, A, Q) 作为一个连续寄存器整体右移一位。系统把 A 的最低位写入 Q 的最高位，并把 C 写入 A 的最高位。");
        this.shiftRight();
        this.addStep(`右移完成：此时 C = ${this.C}，A = ${this.A.join('')}，Q = ${this.Q.join('')}。`);

        this.counter--;
        this.addStep(`系统把计数器减一：counter ← ${this.counter}。`);
      }

      const product = this.A.concat(this.Q).join('');
      const decimalProduct = parseInt(product, 2);
      this.addStep(`计算结束：系统给出最终乘积。二进制结果为 ${product}，十进制结果为 ${decimalProduct}。`);
    }

    // 原码乘法
    signMagnitudeMultiplication() {
      this.addStep("系统开始原码乘法。每一轮中，系统读取 Q 的最低位 Q0；如果 Q0 为 1，系统把 M 加到 A 并忽略进位。随后，系统对 (A, Q) 做右移：系统把 A 的最低位写入 Q 的最高位，并把 A 的最高位补 0。最终的符号位在初始化阶段已确定。");

      while (this.counter > 0) {
        const stepNum = this.n - this.counter + 1;
        this.addStep(`第 ${stepNum} 轮：系统读取 Q 的最低位 Q0 = ${this.Q[this.n - 1]}。如果 Q0 为 1，系统把 M 加到 A；如果 Q0 为 0，系统跳过加法。`);

        if (this.Q[this.n - 1] === 1) {
          this.addStep("系统执行加法：A ← A + M。系统忽略加法产生的进位，并只保留 A 的低 n 位。");
          this.addBinaryArraysNoCarry();
          this.addStep(`加法结束：此时 A = ${this.A.join('')}。`);
        }

        this.addStep("系统对 (A, Q) 各右移一位。系统把 A 的最低位写入 Q 的最高位，并把 A 的最高位补 0。");
        this.shiftRightNoCarry();
        this.addStep(`右移完成：此时 A = ${this.A.join('')}，Q = ${this.Q.join('')}。`);

        this.counter--;
        this.addStep(`系统把计数器减一：counter ← ${this.counter}。`);
      }

      const magnitude = this.A.concat(this.Q).join('');
      const product = (this.sign ? '1' : '0') + magnitude;
      const decimalMagnitude = parseInt(magnitude, 2);
      const decimalProduct = this.sign ? -decimalMagnitude : decimalMagnitude;
      this.addStep(`计算结束：系统根据已确定的符号位与数值部分给出结果。二进制结果为 ${product}，十进制结果为 ${decimalProduct}。`);
    }

    // Booth 乘法（补码）
    boothMultiplication() {
      this.addStep("系统开始 Booth 乘法。每一轮中，系统读取 Q 的最低位 Q0 和 Q_{-1}。当 (Q0, Q_{-1}) 为 (0, 1) 时，系统把 M 加到 A；当 (1, 0) 时，系统把 −M 加到 A；当两位相同（0,0 或 1,1）时，系统不进行加法。随后，系统对 (A, Q, Q_{-1}) 做一次算术右移：系统保持 A 的最高位不变，把 A 的最低位写入 Q 的最高位，并把 Q 的最低位写入 Q_{-1}。");

      while (this.counter > 0) {
        const stepNum = this.n - this.counter + 1;
        const q0 = this.Q[this.n - 1];
        const qm1 = this.Q_minus1;

        this.addStep(`第 ${stepNum} 轮：系统读取 Q0 = ${q0} 与 Q_{-1} = ${qm1}。当它们为 (0, 1) 时，系统把 M 加到 A；当它们为 (1, 0) 时，系统把 −M 加到 A；当它们相同，系统保持 A 不变。`);

        if (q0 === 0 && qm1 === 1) {
          this.addStep("系统执行加法：A ← A + M（补码加法，系统忽略最终溢出）。");
          this.addBinaryArraysTwosComplement(this.M);
        } else if (q0 === 1 && qm1 === 0) {
          this.addStep("系统执行加法：A ← A + (−M)（补码加法，系统忽略最终溢出）。");
          this.addBinaryArraysTwosComplement(this.M_minus);
        } else {
          this.addStep("系统不执行加法：A 保持不变。");
        }
        this.addStep(`加法结束：此时 A = ${this.A.join('')}。`);

        this.addStep("系统对 (A, Q, Q_{-1}) 执行算术右移一位。系统保持 A 的最高位不变，把 A 的最低位写入 Q 的最高位，并把 Q 的最低位写入 Q_{-1}。");
        this.arithmeticShiftRight();
        this.addStep(`右移完成：此时 A = ${this.A.join('')}，Q = ${this.Q.join('')}，Q_{-1} = ${this.Q_minus1}。`);

        this.counter--;
        this.addStep(`系统把计数器减一：counter ← ${this.counter}。`);
      }

      const product = this.A.concat(this.Q).join('');
      const decimalProduct = this.fromTwosComplement(product);
      this.addStep(`计算结束：系统给出结果。二进制结果为 ${product}，十进制结果为 ${decimalProduct}。`);
    }

    // 工具
    toUnsignedBinary(number){
      let b = number.toString(2);
      while (b.length < this.n) b = '0' + b;
      return b.split('').map(x => parseInt(x));
    }
    toTwosComplement(number){
      if (number >= 0) return this.toUnsignedBinary(number);
      const p = this.toUnsignedBinary(-number);
      const inv = p.map(x => x ? 0 : 1);
      let carry = 1, res = [];
      for (let i = inv.length - 1; i >= 0; i--){
        const s = inv[i] + carry;
        res.unshift(s & 1);
        carry = s >> 1;
      }
      return res;
    }
    fromTwosComplement(str){
      const bits = str.split('').map(x => parseInt(x));
      if (bits[0] === 1){
        // 负数：减 1 再取反
        let borrow = 1, sub = [];
        for (let i = bits.length - 1; i >= 0; i--){
          let v = bits[i] - borrow;
          if (v < 0){ v = 1; borrow = 1; } else { borrow = 0; }
          sub.unshift(v);
        }
        const inv = sub.map(x => x ? 0 : 1);
        const pos = parseInt(inv.join(''), 2);
        return -pos;
      }
      return parseInt(str, 2);
    }

    addBinaryArrays(){
      let carry = 0;
      for (let i = this.n - 1; i >= 0; i--){
        const sum = this.A[i] + this.M[i] + carry;
        this.A[i] = sum & 1;
        carry = sum >> 1;
      }
      this.C = carry;
    }
    addBinaryArraysNoCarry(){
      let carry = 0;
      for (let i = this.n - 1; i >= 0; i--){
        const sum = this.A[i] + this.M[i] + carry;
        this.A[i] = sum & 1;
        carry = sum >> 1;
      }
    }
    addBinaryArraysTwosComplement(addend){
      let carry = 0;
      for (let i = this.n - 1; i >= 0; i--){
        const sum = this.A[i] + addend[i] + carry;
        this.A[i] = sum & 1;
        carry = sum >> 1;
      }
    }

    shiftRight(){
      // (C, A, Q) 整体右移
      for (let i = this.n - 1; i > 0; i--) this.Q[i] = this.Q[i - 1];
      this.Q[0] = this.A[this.n - 1];
      for (let i = this.n - 1; i > 0; i--) this.A[i] = this.A[i - 1];
      this.A[0] = this.C;
      this.C = 0;
    }
    shiftRightNoCarry(){
      // (A, Q) 右移；A 最高位补 0
      for (let i = this.n - 1; i > 0; i--) this.Q[i] = this.Q[i - 1];
      this.Q[0] = this.A[this.n - 1];
      for (let i = this.n - 1; i > 0; i--) this.A[i] = this.A[i - 1];
      this.A[0] = 0;
    }
    arithmeticShiftRight(){
      // (A, Q, Q_-1) 算术右移；A[MSB] 保持
      const aMsb = this.A[0];
      this.Q_minus1 = this.Q[this.n - 1];
      for (let i = this.n - 1; i > 0; i--) this.Q[i] = this.Q[i - 1];
      this.Q[0] = this.A[this.n - 1];
      for (let i = this.n - 1; i > 0; i--) this.A[i] = this.A[i - 1];
      this.A[0] = aMsb;
    }

    addStep(description, registers = {}){
      const snapshot = {};
      if (this.A) snapshot.A = this.A.join('');
      if (this.Q) snapshot.Q = this.Q.join('');
      if (this.M) snapshot.M = this.M.join('');
      if (typeof this.C !== 'undefined') snapshot.C = this.C;
      if (typeof this.counter !== 'undefined') snapshot.counter = this.counter;
      if (typeof this.sign !== 'undefined') snapshot.sign = this.sign;
      if (typeof this.Q_minus1 !== 'undefined') snapshot["Q_-1"] = this.Q_minus1;
      if (this.M_minus) snapshot["M_minus"] = this.M_minus.join('');
      this.steps.push({ description, registers: { ...snapshot, ...registers } });
    }
  }

  // 控制
  function stopAutoIfRunning(){
    if (autoInterval){
      clearInterval(autoInterval);
      autoInterval = null;
      autoBtn.innerHTML = '<i class="fas fa-play-circle"></i> 自动播放';
      autoBtn.className = 'btn-warning';
    }
  }
  function goPrev(){
    if (multiplication && currentStep > 0){
      currentStep--;
      stopAutoIfRunning();
      updateDisplay();
    }
  }
  function goNext(){
    if (multiplication && currentStep < multiplication.steps.length - 1){
      currentStep++;
      stopAutoIfRunning();
      updateDisplay();
    }
  }

  // 事件
  initializeBtn.addEventListener('click', () => {
    const type = multiplicationTypeEl.value;
    const bitLength = bitLengthEl.value;
    const operand1 = operand1El.value;
    const operand2 = operand2El.value;
    multiplication = new BinaryMultiplication(type, bitLength, operand1, operand2);
    currentStep = 0;
    if (!multiplication.valid || !multiplication.steps.length){
      stepBtn.disabled = true; prevBtn.disabled = true; autoBtn.disabled = true;
    }else{
      stepBtn.disabled = false; prevBtn.disabled = true; autoBtn.disabled = false;
    }
    stopAutoIfRunning();
    updateDisplay();
  });
  prevBtn.addEventListener('click', goPrev);
  stepBtn.addEventListener('click', goNext);
  autoBtn.addEventListener('click', () => {
    if (autoInterval){ stopAutoIfRunning(); }
    else{
      autoBtn.innerHTML = '<i class="fas fa-pause-circle"></i> 停止';
      autoBtn.className = 'btn-danger';
      const speed = parseInt(animationSpeedEl.value);
      autoInterval = setInterval(() => {
        if (multiplication && currentStep < multiplication.steps.length - 1){
          currentStep++; updateDisplay();
        }else{ stopAutoIfRunning(); }
      }, speed);
    }
  });
  resetBtn.addEventListener('click', () => {
    stopAutoIfRunning();
    multiplication = null; currentStep = 0;
    stepBtn.disabled = true; prevBtn.disabled = true; autoBtn.disabled = true;
    explanationEl.innerHTML = "<p><i class='fas fa-info-circle'></i> 点击“初始化计算”开始。使用“下一步/上一步”逐步查看。右侧只显示当前一步的说明。</p>";
    registersEl.innerHTML = ""; stepsEl.innerHTML = ""; resultEl.textContent = ""; resultEl.className = 'result';
  });

  // 辅助
  function combinedDecimal(A, Q){
    if (!multiplication) return 0;
    const full = (A || '') + (Q || '');
    if (!full) return 0;
    if (multiplication.type === 'booth') return multiplication.fromTwosComplement(full);
    if (multiplication.type === 'signMagnitude'){
      const val = parseInt(full, 2); return multiplication.sign ? -val : val;
    }
    return parseInt(full, 2);
  }
  function labelType(type){
    if (type === 'unsigned') return '无符号乘法';
    if (type === 'signMagnitude') return '原码乘法';
    if (type === 'booth') return 'Booth（补码）乘法';
    return type;
  }
  function formatStep(description, idx){
    const total = multiplication ? multiplication.steps.length : 0;
    return `<div class="step"><h4>步骤 ${idx + 1} / ${total}</h4><p>${description}</p></div>`;
  }
  function isShiftCompletion(desc){ return desc.includes("右移完成"); }
  function isAdditionResultStep(desc){ return desc.includes("加法结束"); }

  function insertedBitForShift(prevRegs){
    if (!multiplication || !prevRegs) return 0;
    if (multiplication.type === 'unsigned') return Number(prevRegs.C || 0);
    if (multiplication.type === 'signMagnitude') return 0;
    if (multiplication.type === 'booth') return Number((prevRegs.A || '0')[0] || 0); // A 的最高位
    return 0;
  }
  function createBitEl(bit, extra=''){
    const el = document.createElement('div');
    el.className = `bit bit-${bit} ${extra}`.trim();
    el.textContent = bit;
    return el;
  }

  // 渲染
  function updateDisplay(){
    if (!multiplication || !multiplication.steps.length){
      explanationEl.innerHTML = `<p><i class="fas fa-info-circle"></i> 点击“初始化计算”开始。初始化后会显示第一步。</p>`;
      registersEl.innerHTML = ""; stepsEl.innerHTML = ""; resultEl.textContent = ""; resultEl.className = 'result'; return;
    }

    const step = multiplication.steps[currentStep];
    const prevStep = currentStep > 0 ? multiplication.steps[currentStep - 1] : null;
    const rs = step.registers;
    const prev = prevStep ? prevStep.registers : null;

    explanationEl.innerHTML = `<p><i class="fas fa-info-circle"></i> 算法：${labelType(multiplication.type)}｜位宽：${multiplication.n}｜进度：第 ${currentStep + 1} / ${multiplication.steps.length} 步</p>`;
    stepsEl.innerHTML = formatStep(step.description, currentStep);

    registersEl.innerHTML = "";

    // 结果寄存器 A+Q（两段背景，位宽改变自动对应）
    if (rs.A && rs.Q){
      const group = document.createElement('div'); group.className = 'register-group wide';
      const title = document.createElement('div'); title.className = 'register-title'; title.innerHTML = '<i class="fas fa-tasks"></i> 结果寄存器 (A + Q)';
      group.appendChild(title);

      const labels = document.createElement('div'); labels.className = 'register-labels'; labels.innerHTML = `<span>高位 (A)</span><span>低位 (Q)</span>`;
      group.appendChild(labels);

      const row = document.createElement('div'); row.className = 'result-row';
      const resultBits = document.createElement('div'); resultBits.className = 'result-bits';

      // A 半
      const halfA = document.createElement('div'); halfA.className = 'half half-A';
      const aBits = rs.A.split('');
      aBits.forEach((b,i) => {
        const el = createBitEl(b);
        // 加法结束步骤：把发生变化的 A 位做淡入
        if (prev && prev.A && isAdditionResultStep(step.description) && prev.A[i] !== b){
          el.classList.add('bit-fade-in');
          requestAnimationFrame(() => el.classList.add('ready'));
        }
        halfA.appendChild(el);
      });

      // Q 半
      const halfQ = document.createElement('div'); halfQ.className = 'half half-Q';
      rs.Q.split('').forEach((b) => {
        const el = createBitEl(b);
        halfQ.appendChild(el);
      });

      resultBits.appendChild(halfA);
      resultBits.appendChild(halfQ);
      row.appendChild(resultBits);

      // 移位动画：整体右移一位 → 左侧补入 → 最右位淡出
      if (prev && prev.A && prev.Q && isShiftCompletion(step.description)){
        const overlay = document.createElement('div'); overlay.className = 'shift-overlay';
        const ghostLine = document.createElement('div'); ghostLine.className = 'ghost-line';
        const prevFull = (prev.A + prev.Q).split('');
        prevFull.forEach((b, idx) => {
          const ghost = createBitEl(b, 'shift-ghost');
          if (idx === prevFull.length - 1) ghost.classList.add('shift-exit');
          ghostLine.appendChild(ghost);
        });
        overlay.appendChild(ghostLine);

        const insBit = insertedBitForShift(prev);
        const enter = createBitEl(insBit, 'shift-enter');
        overlay.appendChild(enter);

        row.appendChild(overlay);

        requestAnimationFrame(() => {
          overlay.querySelectorAll('.shift-ghost').forEach(el => el.classList.add('move'));
        });

        const speed = parseInt(animationSpeedEl.value);
        const totalMs = Math.max(300, Math.min(800, speed + 100));
        setTimeout(() => { if (overlay.parentNode) overlay.parentNode.removeChild(overlay); }, totalMs);
      }

      const val = document.createElement('div'); val.className = 'register-value';
      val.textContent = `十进制: ${combinedDecimal(rs.A, rs.Q)}`;

      group.appendChild(row);
      group.appendChild(val);
      registersEl.appendChild(group);
    }

    // 其他寄存器；合并 counter 与 Q_-1
    let counterVal = null, qMinus1Val = null;
    for (const [name, value] of Object.entries(rs)){
      if (name === 'A' || name === 'Q') continue;
      if (name === 'counter'){ counterVal = value; continue; }
      if (name === 'Q_-1'){ qMinus1Val = value; continue; }

      const group = document.createElement('div'); group.className = 'register-group';
      const title = document.createElement('div'); title.className = 'register-title';
      let icon = 'fas fa-memory';
      if (name === 'M') icon = 'fas fa-xmark';
      if (name === 'C') icon = 'fas fa-arrow-up';
      if (name === 'sign') icon = 'fas fa-adjust';
      if (name === 'M_minus') icon = 'fas fa-minus';
      title.innerHTML = `<i class="${icon}"></i> ${name} 寄存器`;
      group.appendChild(title);

      const box = document.createElement('div'); box.className = 'bits-container';
      if (typeof value === 'number' || (typeof value === 'string' && value.length === 1 && name !== 'M' && name !== 'M_minus')){
        const chip = document.createElement('div'); chip.className = 'chip small'; chip.textContent = value.toString();
        box.appendChild(chip);
      }else{
        const bits = value.toString().split('');
        const prevStr = prev && prev[name] ? prev[name].toString() : null;
        bits.forEach((b,i) => {
          const el = createBitEl(b);
          // A 寄存器：加法结束步骤对变动位淡入
          if (name === 'A' && prevStr && isAdditionResultStep(step.description) && prevStr[i] !== b){
            el.classList.add('bit-fade-in');
            requestAnimationFrame(() => el.classList.add('ready'));
          }
          box.appendChild(el);
        });
      }
      group.appendChild(box);
      registersEl.appendChild(group);
    }

    // 合并显示 counter 与 Q_-1
    if (counterVal !== null || qMinus1Val !== null){
      const group = document.createElement('div'); group.className = 'register-group';
      const title = document.createElement('div'); title.className = 'register-title';
      title.innerHTML = `<i class="fas fa-sliders-h"></i> 控制位（合并）`;
      group.appendChild(title);

      const chips = document.createElement('div'); chips.className = 'inline-chips';
      if (qMinus1Val !== null){
        const c1 = document.createElement('div'); c1.className = 'chip small'; c1.textContent = `Q_-1 = ${qMinus1Val}`;
        chips.appendChild(c1);
      }
      if (counterVal !== null){
        const c2 = document.createElement('div'); c2.className = 'chip small'; c2.textContent = `counter = ${counterVal}`;
        chips.appendChild(c2);
      }
      group.appendChild(chips);
      registersEl.appendChild(group);
    }

    // 底部便捷控制
    const ctrl = document.createElement('div'); ctrl.className = 'register-group';
    const ctrlTitle = document.createElement('div'); ctrlTitle.className = 'register-title'; ctrlTitle.innerHTML = `<i class="fas fa-gamepad"></i> 快捷控制`;
    ctrl.appendChild(ctrlTitle);
    const compact = document.createElement('div'); compact.className = 'btn-group-compact';
    const bPrev = document.createElement('button'); bPrev.className = 'btn-light'; bPrev.innerHTML = '<i class="fas fa-step-backward"></i> 上一步'; bPrev.disabled = currentStep === 0; bPrev.addEventListener('click', goPrev);
    const bNext = document.createElement('button'); bNext.className = 'btn-success'; bNext.innerHTML = '<i class="fas fa-step-forward"></i> 下一步'; bNext.disabled = currentStep >= multiplication.steps.length - 1; bNext.addEventListener('click', goNext);
    compact.appendChild(bPrev); compact.appendChild(bNext); ctrl.appendChild(compact); registersEl.appendChild(ctrl);

    // 结果
    if (currentStep === multiplication.steps.length - 1){
      const last = rs;
      if (last && last.A && last.Q){
        const dec = combinedDecimal(last.A, last.Q);
        const expected = multiplication.operand1 * multiplication.operand2;
        resultEl.textContent = `最终乘积：二进制 ${last.A + last.Q}，十进制 ${dec}`;
        resultEl.className = dec === expected ? 'result' : 'result result-error';
        if (dec !== expected) resultEl.textContent += `（期望: ${expected}）`;
      }
    }else{
      resultEl.textContent = ""; resultEl.className = 'result';
    }

    // 顶部按钮状态
    prevBtn.disabled = currentStep === 0;
    stepBtn.disabled = currentStep >= multiplication.steps.length - 1;
    autoBtn.disabled = multiplication.steps.length <= 1 || currentStep >= multiplication.steps.length - 1;
  }
</script>
</body>
</html>