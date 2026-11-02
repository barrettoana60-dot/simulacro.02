/**
 * OCULUS-AI // Physiological Eye Tracking Simulator v2.0
 * Gera cinemática ocular 100% precisa no sistema de coordenadas MediaPipe.
 * Os landmarks simulados são idênticos em estrutura aos do tracker real,
 * para garantir que o pipeline de rendering funcione do mesmo jeito.
 */

export class EyeSimulator {
  constructor(onResultsCallback) {
    this.onResults = onResultsCallback;
    this.isActive  = false;
    this.animId    = null;

    // Estado fisiológico
    this.targetGaze  = { x: 0.5, y: 0.5 };
    this.currentGaze = { x: 0.5, y: 0.5 };
    this.headPose    = { pitch: 0, yaw: 0, roll: 0 };
    this.targetHead  = { pitch: 0, yaw: 0, roll: 0 };

    // Piscadas
    this.earProgress  = 0;
    this.isBlinking   = false;
    this.blinkCount   = 0;
    this.lastBlink    = performance.now();
    this.blinkInterval = 3800;
    this.currentEAR   = 0.28;
    this.blinkTimes   = [];

    // Sacadas — pontos de interesse no ecrã
    this.interestPoints = [
      { x: 0.50, y: 0.25 },
      { x: 0.25, y: 0.18 },
      { x: 0.75, y: 0.18 },
      { x: 0.35, y: 0.50 },
      { x: 0.65, y: 0.50 },
      { x: 0.50, y: 0.80 },
      { x: 0.20, y: 0.72 },
      { x: 0.80, y: 0.72 }
    ];
    this.interestIdx = 0;
    this.lastSaccade = performance.now();
    this.saccadeInterval = 2200;

    this.fps = 60;
    this.frameCount = 0;
    this.fpsTimer = performance.now();
  }

  start() {
    this.isActive = true;
    this._loop();
  }

  stop() {
    this.isActive = false;
    if (this.animId) cancelAnimationFrame(this.animId);
    this.animId = null;
  }

  _loop() {
    if (!this.isActive) return;
    this._update();
    this._emit();
    this.animId = requestAnimationFrame(() => this._loop());
  }

  _update() {
    const now = performance.now();

    // ── Sacada Planejada ─────────────────────────────────────────
    if (now - this.lastSaccade > this.saccadeInterval) {
      this.interestIdx = (this.interestIdx + 1) % this.interestPoints.length;
      const p = this.interestPoints[this.interestIdx];
      this.targetGaze = {
        x: p.x + (Math.random() - 0.5) * 0.06,
        y: p.y + (Math.random() - 0.5) * 0.06
      };
      this.targetHead = {
        yaw:   (this.targetGaze.x - 0.5) * 18,
        pitch: -(this.targetGaze.y - 0.5) * 14,
        roll:  (Math.random() - 0.5) * 3
      };
      this.lastSaccade = now;
      this.saccadeInterval = 900 + Math.random() * 2200;
    }

    // ── Interpolação EMA do Olhar ────────────────────────────────
    const k = 0.18;
    this.currentGaze.x += (this.targetGaze.x - this.currentGaze.x) * k;
    this.currentGaze.y += (this.targetGaze.y - this.currentGaze.y) * k;

    // Microssacadas e drift browniano fixacional
    this.currentGaze.x += (Math.random() - 0.5) * 0.0025;
    this.currentGaze.y += (Math.random() - 0.5) * 0.0025;
    this.currentGaze.x = Math.max(0.05, Math.min(0.95, this.currentGaze.x));
    this.currentGaze.y = Math.max(0.05, Math.min(0.95, this.currentGaze.y));

    // ── Pose da Cabeça ───────────────────────────────────────────
    const hk = 0.04;
    this.headPose.yaw   += (this.targetHead.yaw   - this.headPose.yaw)   * hk;
    this.headPose.pitch += (this.targetHead.pitch - this.headPose.pitch) * hk;
    this.headPose.roll  += (this.targetHead.roll  - this.headPose.roll)  * hk;

    // ── Piscada Fisiológica ──────────────────────────────────────
    if (!this.isBlinking && now - this.lastBlink > this.blinkInterval) {
      this.isBlinking   = true;
      this.earProgress  = 0;
      this.blinkCount++;
      this.lastBlink    = now;
      this.blinkInterval = 3000 + Math.random() * 3000;
      this.blinkTimes.push(now);
    }

    if (this.isBlinking) {
      this.earProgress += 0.18;
      this.currentEAR = 0.28 * (1 - Math.sin(this.earProgress * Math.PI));
      if (this.earProgress >= 1) { this.isBlinking = false; this.currentEAR = 0.28; }
    } else {
      this.currentEAR = 0.28 + Math.sin(now * 0.001) * 0.015;
    }

    this.blinkTimes = this.blinkTimes.filter(t => now - t < 60000);

    // FPS
    this.frameCount++;
    if (now - this.fpsTimer >= 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.fpsTimer = now;
    }
  }

  _emit() {
    const now = performance.now();
    const u = this.currentGaze.x;
    const v = this.currentGaze.y;

    // ── Posição anatômica da íris no espaço normalizado ──────────
    // IPD simulado: ~0.24 de largura (relativo ao frame 0-1)
    const ipdHalf = 0.12;
    const eyeY    = 0.40 + (v - 0.5) * 0.10;
    const eyeOffX = (u - 0.5) * 0.06; // movimento horizontal da íris

    // Olho direito (índice 468, câmera left = observador right)
    const rightIris = {
      x: 0.38 + eyeOffX + (Math.random() - 0.5) * 0.001,
      y: eyeY + (Math.random() - 0.5) * 0.001,
      z: -0.018
    };
    // Olho esquerdo (índice 473, câmera right = observador left)
    const leftIris = {
      x: 0.62 + eyeOffX + (Math.random() - 0.5) * 0.001,
      y: eyeY + (Math.random() - 0.5) * 0.001,
      z: -0.018
    };

    // ── Ratio da íris no globo ocular ────────────────────────────
    const rightRatios = { x: u, y: v };
    const leftRatios  = { x: u, y: v };

    // ── Polígonos dos olhos (elipse) ─────────────────────────────
    const earFactor = this.currentEAR / 0.28;
    const rightEyePoly = this._ellipse(rightIris.x, eyeY, 0.055, 0.028 * earFactor, 16);
    const leftEyePoly  = this._ellipse(leftIris.x,  eyeY, 0.055, 0.028 * earFactor, 16);
    const rightIrisPoly = this._ellipse(rightIris.x, eyeY, 0.018, 0.018, 8);
    const leftIrisPoly  = this._ellipse(leftIris.x,  eyeY, 0.018, 0.018, 8);

    // ── Landmarks faciais simulados (478 pontos) ─────────────────
    const fakeLM = this._buildFaceLandmarks(rightIris, leftIris, eyeY, earFactor);

    this.onResults({
      detected:    true,
      timestamp:   now,
      fps:         this.fps,
      latency:     4,
      confidence:  1.0,
      isSimulated: true,

      rightPupil: rightIris,
      leftPupil:  leftIris,

      polygons: {
        rightEye:  rightEyePoly,
        leftEye:   leftEyePoly,
        rightIris: rightIrisPoly,
        leftIris:  leftIrisPoly
      },

      rightRatios,
      leftRatios,
      combinedGazeRatio: { x: u, y: v },

      rightEAR:    this.currentEAR,
      leftEAR:     this.currentEAR,
      avgEAR:      this.currentEAR,
      earBaseline: 0.28,
      isBlinking:  this.isBlinking,
      blinkCount:  this.blinkCount,
      blinkRateBPM: this.blinkTimes.length,

      headPose:         this.headPose,
      metricDistanceCm: 58,
      allLandmarks:     fakeLM
    });
  }

  _ellipse(cx, cy, rx, ry, n = 16) {
    const pts = [];
    for (let i = 0; i < n; i++) {
      const a = (i / n) * Math.PI * 2;
      pts.push({ x: cx + Math.cos(a) * rx, y: cy + Math.sin(a) * ry, z: 0 });
    }
    return pts;
  }

  _buildFaceLandmarks(rightIris, leftIris, eyeY, earF) {
    // Constrói 478 landmarks sintéticos realistas
    const lm = new Array(478);
    const cx = 0.5, cy = 0.5;

    // Face oval
    for (let i = 0; i < 468; i++) {
      const t = i / 468;
      const angle = t * Math.PI * 2;
      const fx = cx + Math.cos(angle) * (0.22 + Math.sin(angle * 3) * 0.03);
      const fy = cy + Math.sin(angle) * (0.30 + Math.cos(angle * 2) * 0.02);
      lm[i] = { x: fx, y: fy, z: -Math.abs(Math.cos(angle * 2)) * 0.05 };
    }

    // Pontos-chave específicos (nose, chin, forehead, mouth, cheeks)
    lm[1]  = { x: 0.50, y: 0.53, z: -0.08 }; // nose tip
    lm[152]= { x: 0.50, y: 0.80, z: -0.02 }; // chin
    lm[10] = { x: 0.50, y: 0.20, z: -0.01 }; // forehead
    lm[291]= { x: 0.62, y: 0.67, z: -0.02 }; // mouth left
    lm[61] = { x: 0.38, y: 0.67, z: -0.02 }; // mouth right
    lm[234]= { x: 0.22, y: 0.50, z: -0.01 }; // cheek left
    lm[454]= { x: 0.78, y: 0.50, z: -0.01 }; // cheek right

    // Olho direito (contorno + EAR)
    const rEyeConts = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
    rEyeConts.forEach((idx, i) => {
      const a = (i / rEyeConts.length) * Math.PI * 2;
      lm[idx] = {
        x: rightIris.x + Math.cos(a) * 0.055,
        y: eyeY + Math.sin(a) * 0.028 * earF,
        z: -0.01
      };
    });

    // Olho esquerdo
    const lEyeConts = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
    lEyeConts.forEach((idx, i) => {
      const a = (i / lEyeConts.length) * Math.PI * 2;
      lm[idx] = {
        x: leftIris.x + Math.cos(a) * 0.055,
        y: eyeY + Math.sin(a) * 0.028 * earF,
        z: -0.01
      };
    });

    // Íris direita (marcos 468-472)
    lm[468] = { ...rightIris };
    [469, 470, 471, 472].forEach((idx, i) => {
      const a = (i / 4) * Math.PI * 2;
      lm[idx] = { x: rightIris.x + Math.cos(a) * 0.018, y: rightIris.y + Math.sin(a) * 0.018, z: -0.02 };
    });

    // Íris esquerda (marcos 473-477)
    lm[473] = { ...leftIris };
    [474, 475, 476, 477].forEach((idx, i) => {
      const a = (i / 4) * Math.PI * 2;
      lm[idx] = { x: leftIris.x + Math.cos(a) * 0.018, y: leftIris.y + Math.sin(a) * 0.018, z: -0.02 };
    });

    return lm;
  }
}
