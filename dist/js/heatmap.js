/**
 * OCULUS-AI // High-Performance Real-Time Heatmap & Scanpath Engine
 * Gera mapas de calor de densidade de atenção visual em tempo real, detecção
 * de fixações (algoritmo I-VT), scanpaths sacádicos e exportação de alta resolução.
 */

export class HeatmapEngine {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = this.canvas.getContext('2d');

    // Canvas auxiliar para acúmulo de densidade (escala de cinza / alfa)
    this.shadowCanvas = document.createElement('canvas');
    this.shadowCtx = this.shadowCanvas.getContext('2d', { willReadFrequently: true });

    // Configurações padrão
    this.radius = options.radius || 42;
    this.opacity = options.opacity !== undefined ? options.opacity : 0.75;
    this.decay = options.decay !== undefined ? options.decay : 0.005; // 0 para cumulativo
    this.activePalette = options.palette || 'turbo';
    this.showScanpath = true;
    this.showHeatmap = true;

    // Detecção de Fixações (I-VT: Velocity-Threshold Identification)
    this.velocityThreshold = 40; // px por quadro
    this.minFixationDuration = 100; // ms
    this.lastPoint = null;
    this.lastTimestamp = 0;
    this.currentFixation = null;
    this.fixations = []; // Histórico de fixações identificadas
    this.saccades = [];

    // Paletas de Cores Pré-computadas (Lookup Tables 256x4)
    this.paletteGradients = this.initColorPalettes();

    // Estímulo de fundo (imagem personalizada ou padrão)
    this.backgroundImage = null;

    this.resize();
  }

  resize() {
    const parent = this.canvas.parentElement;
    const width = parent ? parent.clientWidth : window.innerWidth;
    const height = parent ? parent.clientHeight : window.innerHeight;

    if (width === 0 || height === 0) return;

    this.canvas.width = width;
    this.canvas.height = height;
    this.shadowCanvas.width = width;
    this.shadowCanvas.height = height;

    this.clearShadow();
  }

  clearShadow() {
    this.shadowCtx.clearRect(0, 0, this.shadowCanvas.width, this.shadowCanvas.height);
  }

  reset() {
    this.clearShadow();
    this.fixations = [];
    this.saccades = [];
    this.lastPoint = null;
    this.currentFixation = null;
    this.render();
  }

  /**
   * Registra um novo ponto de olhar predito na tela
   */
  addPoint(screenX, screenY, timestamp = performance.now()) {
    if (isNaN(screenX) || isNaN(screenY)) return;

    // Normaliza para as dimensões deste canvas
    const rect = this.canvas.getBoundingClientRect();
    const x = screenX - rect.left;
    const y = screenY - rect.top;

    if (x < 0 || y < 0 || x > this.canvas.width || y > this.canvas.height) {
      return;
    }

    // 1. Carimba o gradiente radial na camada de densidade
    this.stampDensity(x, y);

    // 2. Classificação I-VT (Fixação vs Sacada)
    this.processIVT(x, y, timestamp);
  }

  /**
   * Deposita intensidade de calor (radial blur)
   */
  stampDensity(x, y) {
    const rad = this.radius;
    const grad = this.shadowCtx.createRadialGradient(x, y, rad * 0.1, x, y, rad);
    grad.addColorStop(0, 'rgba(0, 0, 0, 0.16)');
    grad.addColorStop(0.5, 'rgba(0, 0, 0, 0.08)');
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    this.shadowCtx.fillStyle = grad;
    this.shadowCtx.beginPath();
    this.shadowCtx.arc(x, y, rad, 0, Math.PI * 2);
    this.shadowCtx.fill();
  }

  /**
   * Algoritmo I-VT (Velocity-Threshold Identification)
   */
  processIVT(x, y, timestamp) {
    if (!this.lastPoint) {
      this.lastPoint = { x, y, t: timestamp };
      this.currentFixation = {
        id: 1,
        x,
        y,
        startTime: timestamp,
        endTime: timestamp,
        duration: 0,
        samples: [{ x, y }]
      };
      return;
    }

    const dt = Math.max(1, timestamp - this.lastPoint.t);
    const dist = Math.hypot(x - this.lastPoint.x, y - this.lastPoint.y);
    const velocity = (dist / dt) * 16.6; // Normalizado para ~60fps

    if (velocity < this.velocityThreshold) {
      // Fixação contínua
      if (this.currentFixation) {
        this.currentFixation.samples.push({ x, y });
        this.currentFixation.endTime = timestamp;
        this.currentFixation.duration = timestamp - this.currentFixation.startTime;

        // Recalcula o centroide da fixação
        let sumX = 0, sumY = 0;
        for (const s of this.currentFixation.samples) {
          sumX += s.x;
          sumY += s.y;
        }
        this.currentFixation.x = sumX / this.currentFixation.samples.length;
        this.currentFixation.y = sumY / this.currentFixation.samples.length;
      } else {
        this.currentFixation = {
          id: this.fixations.length + 1,
          x,
          y,
          startTime: timestamp,
          endTime: timestamp,
          duration: 0,
          samples: [{ x, y }]
        };
      }
    } else {
      // Sacada (movimento rápido entre fixações)
      if (this.currentFixation && this.currentFixation.duration >= this.minFixationDuration) {
        this.fixations.push({ ...this.currentFixation });
        if (this.fixations.length > 60) this.fixations.shift();

        // Registra sacada
        if (this.fixations.length >= 2) {
          const prev = this.fixations[this.fixations.length - 2];
          const curr = this.fixations[this.fixations.length - 1];
          this.saccades.push({ from: prev, to: curr });
          if (this.saccades.length > 59) this.saccades.shift();
        }
      }
      this.currentFixation = null;
    }

    this.lastPoint = { x, y, t: timestamp };
  }

  /**
   * Aplica decaimento temporal se configurado
   */
  applyDecay() {
    if (this.decay > 0) {
      this.shadowCtx.globalCompositeOperation = 'destination-out';
      this.shadowCtx.fillStyle = `rgba(0, 0, 0, ${this.decay})`;
      this.shadowCtx.fillRect(0, 0, this.shadowCanvas.width, this.shadowCanvas.height);
      this.shadowCtx.globalCompositeOperation = 'source-over';
    }
  }

  /**
   * Renderiza a composição do mapa de calor com gradiente cromático
   */
  render() {
    this.applyDecay();

    // 1. Limpa o canvas de exibição
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 2. Fundo (Estímulo ou Grid)
    if (this.backgroundImage) {
      this.ctx.drawImage(this.backgroundImage, 0, 0, this.canvas.width, this.canvas.height);
    } else {
      this.renderBiometricGrid();
    }

    // 3. Colorização do Mapa de Calor (Image Data Shader)
    if (this.showHeatmap) {
      this.colorizeHeatmap();
    }

    // 4. Scanpath Sacádico
    if (this.showScanpath) {
      this.renderScanpath();
    }
  }

  colorizeHeatmap() {
    const width = this.shadowCanvas.width;
    const height = this.shadowCanvas.height;
    if (width === 0 || height === 0) return;

    const shadowData = this.shadowCtx.getImageData(0, 0, width, height);
    const pixels = shadowData.data;
    const outData = this.ctx.createImageData(width, height);
    const out = outData.data;

    const palette = this.paletteGradients[this.activePalette] || this.paletteGradients.turbo;
    const opFactor = this.opacity;

    // Mapeamento canal Alpha -> Cor na LUT
    for (let i = 3; i < pixels.length; i += 4) {
      const alphaVal = pixels[i];
      if (alphaVal > 5) {
        const lutIdx = Math.min(255, alphaVal);
        out[i - 3] = palette[lutIdx * 4];     // R
        out[i - 2] = palette[lutIdx * 4 + 1]; // G
        out[i - 1] = palette[lutIdx * 4 + 2]; // B
        out[i]     = Math.min(255, Math.floor(palette[lutIdx * 4 + 3] * opFactor * (alphaVal / 255))); // A
      }
    }

    // Renderiza dados coloridos via canvas temporário
    const tmp = document.createElement('canvas');
    tmp.width = width;
    tmp.height = height;
    tmp.getContext('2d').putImageData(outData, 0, 0);

    this.ctx.save();
    this.ctx.globalAlpha = this.opacity;
    this.ctx.drawImage(tmp, 0, 0);
    this.ctx.restore();
  }

  renderBiometricGrid() {
    const w = this.canvas.width;
    const h = this.canvas.height;

    this.ctx.fillStyle = '#080d1a';
    this.ctx.fillRect(0, 0, w, h);

    // Linhas de grade sutis
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.04)';
    this.ctx.lineWidth = 1;
    const step = 40;

    this.ctx.beginPath();
    for (let x = 0; x < w; x += step) {
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += step) {
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(w, y);
    }
    this.ctx.stroke();

    // Cruz central de alinhamento
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
    this.ctx.beginPath();
    this.ctx.moveTo(w / 2 - 20, h / 2);
    this.ctx.lineTo(w / 2 + 20, h / 2);
    this.ctx.moveTo(w / 2, h / 2 - 20);
    this.ctx.lineTo(w / 2, h / 2 + 20);
    this.ctx.stroke();
  }

  renderScanpath() {
    if (this.fixations.length === 0 && !this.currentFixation) return;

    const allFixations = [...this.fixations];
    if (this.currentFixation && this.currentFixation.duration >= 50) {
      allFixations.push(this.currentFixation);
    }

    // Linhas de Sacadas com Setas
    this.ctx.save();
    this.ctx.strokeStyle = 'rgba(0, 240, 255, 0.7)';
    this.ctx.lineWidth = 2.5;
    this.ctx.setLineDash([4, 4]);

    for (let i = 1; i < allFixations.length; i++) {
      const p1 = allFixations[i - 1];
      const p2 = allFixations[i];
      this.ctx.beginPath();
      this.ctx.moveTo(p1.x, p1.y);
      this.ctx.lineTo(p2.x, p2.y);
      this.ctx.stroke();
    }
    this.ctx.setLineDash([]);

    // Círculos de Fixação com Numeração Sequencial
    allFixations.forEach((fix, index) => {
      // Raio proporcional à duração da fixação (12px a 36px)
      const rad = Math.min(36, Math.max(12, Math.sqrt(fix.duration) * 1.5));

      // Glow
      this.ctx.fillStyle = 'rgba(255, 0, 102, 0.35)';
      this.ctx.beginPath();
      this.ctx.arc(fix.x, fix.y, rad + 4, 0, Math.PI * 2);
      this.ctx.fill();

      // Círculo Principal
      this.ctx.fillStyle = 'rgba(255, 0, 102, 0.85)';
      this.ctx.strokeStyle = '#ffffff';
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.arc(fix.x, fix.y, rad, 0, Math.PI * 2);
      this.ctx.fill();
      this.ctx.stroke();

      // Número de Ordem
      this.ctx.fillStyle = '#ffffff';
      this.ctx.font = 'bold 11px Inter, sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(index + 1, fix.x, fix.y);
    });

    this.ctx.restore();
  }

  /**
   * Pré-calcula as tabelas de cores RGB (Lookup Tables)
   */
  initColorPalettes() {
    return {
      turbo: this.generateLUT([
        { stop: 0.0, r: 48, g: 18, b: 59, a: 0 },
        { stop: 0.2, r: 70, g: 134, b: 251, a: 120 },
        { stop: 0.4, r: 27, g: 229, b: 181, a: 180 },
        { stop: 0.6, r: 164, g: 252, b: 60, a: 220 },
        { stop: 0.8, r: 251, g: 185, b: 56, a: 240 },
        { stop: 1.0, r: 217, g: 30, b: 24, a: 255 }
      ]),
      inferno: this.generateLUT([
        { stop: 0.0, r: 0, g: 0, b: 4, a: 0 },
        { stop: 0.25, r: 87, g: 16, b: 110, a: 130 },
        { stop: 0.5, r: 187, g: 55, b: 84, a: 190 },
        { stop: 0.75, r: 249, g: 142, b: 9, a: 230 },
        { stop: 1.0, r: 252, g: 255, b: 164, a: 255 }
      ]),
      viridis: this.generateLUT([
        { stop: 0.0, r: 68, g: 1, b: 84, a: 0 },
        { stop: 0.25, r: 59, g: 82, b: 139, a: 120 },
        { stop: 0.5, r: 33, g: 145, b: 140, a: 180 },
        { stop: 0.75, r: 94, g: 201, b: 98, a: 220 },
        { stop: 1.0, r: 253, g: 231, b: 37, a: 255 }
      ]),
      jet: this.generateLUT([
        { stop: 0.0, r: 0, g: 0, b: 128, a: 0 },
        { stop: 0.25, r: 0, g: 255, b: 255, a: 140 },
        { stop: 0.5, r: 0, g: 255, b: 0, a: 190 },
        { stop: 0.75, r: 255, g: 255, b: 0, a: 230 },
        { stop: 1.0, r: 255, g: 0, b: 0, a: 255 }
      ]),
      neon: this.generateLUT([
        { stop: 0.0, r: 10, g: 0, b: 30, a: 0 },
        { stop: 0.3, r: 0, g: 240, b: 255, a: 150 },
        { stop: 0.7, r: 150, g: 0, b: 255, a: 210 },
        { stop: 1.0, r: 255, g: 0, b: 120, a: 255 }
      ])
    };
  }

  generateLUT(stops) {
    const lutCanvas = document.createElement('canvas');
    lutCanvas.width = 256;
    lutCanvas.height = 1;
    const ctx = lutCanvas.getContext('2d');

    const grad = ctx.createLinearGradient(0, 0, 256, 0);
    for (const s of stops) {
      grad.addColorStop(s.stop, `rgba(${s.r}, ${s.g}, ${s.b}, ${s.a / 255})`);
    }

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 256, 1);

    const imgData = ctx.getImageData(0, 0, 256, 1);
    return imgData.data;
  }

  setPalette(paletteName) {
    if (this.paletteGradients[paletteName]) {
      this.activePalette = paletteName;
      this.render();
    }
  }

  setBackground(imageElement) {
    this.backgroundImage = imageElement;
    this.render();
  }

  exportImage() {
    this.render();
    return this.canvas.toDataURL('image/png');
  }
}
