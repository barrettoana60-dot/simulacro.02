/**
 * OCULUS-AI // Deep Learning Eye & Face Mesh Tracking Module v2.0
 * MediaPipe FaceMesh 478 landmarks com refinamento de íris, filtro Kalman-like,
 * recovery automático de perda de tracking e melhor estabilidade para câmera frontal.
 */

export class EyeTracker {
  constructor(options = {}) {
    this.videoElement      = options.videoElement;
    this.onResultsCallback = options.onResults || null;
    this.onErrorCallback   = options.onError   || null;
    this.onStatusCallback  = options.onStatus  || null;

    this.faceMesh  = null;
    this.isRunning = false;
    this.fps       = 0;
    this.frameCount  = 0;
    this.fpsTimer    = performance.now();
    this.lastLatency = 0;

    // ── Marcos Anatômicos MediaPipe Face Mesh 478 ─────────────────
    // Nota: "esquerdo/direito" = perspectiva do observador (não da câmera)
    // Marco 468 = íris DIREITA da câmera (olho esquerdo do observador)
    // Marco 473 = íris ESQUERDA da câmera (olho direito do observador)
    this.LM = {
      RIGHT_IRIS:   468,
      RIGHT_IRIS_P: [469, 470, 471, 472],
      RIGHT_EYE:    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
      RIGHT_INNER:  133,
      RIGHT_OUTER:  33,
      RIGHT_TOP:    159,
      RIGHT_BOT:    145,
      RIGHT_EAR:    [33, 160, 158, 133, 153, 144],

      LEFT_IRIS:    473,
      LEFT_IRIS_P:  [474, 475, 476, 477],
      LEFT_EYE:     [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
      LEFT_INNER:   362,
      LEFT_OUTER:   263,
      LEFT_TOP:     386,
      LEFT_BOT:     374,
      LEFT_EAR:     [362, 385, 387, 263, 373, 380],

      NOSE_TIP:    1,
      CHIN:        152,
      FOREHEAD:    10,
      MOUTH_L:     291,
      MOUTH_R:     61,
      CHEEK_L:     234,
      CHEEK_R:     454
    };

    // ── Filtro EMA Duplo (suavização adaptativa) ──────────────────
    // Alpha baixo = mais suave mas mais lento | Alto = mais responsivo mas trêmulo
    this.alphaFast = 0.45; // Para movimentos grandes (sacadas)
    this.alphaSlow = 0.15; // Para fixações e micromovimentos

    this._leftSmooth  = { x: 0.5, y: 0.5, z: 0, vx: 0, vy: 0 };
    this._rightSmooth = { x: 0.5, y: 0.5, z: 0, vx: 0, vy: 0 };
    this._headSmooth  = { pitch: 0, yaw: 0, roll: 0 };

    // ── Detecção de Piscada (EAR threshold adaptativo) ────────────
    this.earThreshold     = 0.21;
    this.earBaseline      = 0.28; // calibrado continuamente
    this.blinkCount       = 0;
    this.isBlinking       = false;
    this.blinkTimes       = [];
    this._noFaceFrames    = 0;
    this._maxNoFaceFrames = 8;  // Recovery após 8 frames sem rosto
    this._lastValidData   = null;
  }

  // ─────────────────────────────────────────────────────────────────
  async initialize() {
    this.notifyStatus('Carregando rede neural MediaPipe FaceMesh v2 (478 marcos)...');
    try {
      if (typeof window.FaceMesh === 'undefined') {
        throw new Error('MediaPipe FaceMesh não encontrado. Verifique sua conexão.');
      }

      this.faceMesh = new window.FaceMesh({
        locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`
      });

      this.faceMesh.setOptions({
        maxNumFaces:            1,
        refineLandmarks:        true,   // Ativa 478 marcos (inclui íris)
        minDetectionConfidence: 0.72,   // Mais alto = menos falsos positivos
        minTrackingConfidence:  0.68,   // Mais alto = tracking mais estável
        selfieMode:             false   // Não inverter internamente (fazemos via CSS)
      });

      this.faceMesh.onResults((r) => this._processResults(r));
      this.notifyStatus('Modelo de IA pronto. Iniciando câmera...');
      return true;
    } catch (err) {
      console.error('[EyeTracker] Falha na inicialização:', err);
      if (this.onErrorCallback) this.onErrorCallback(err);
      throw err;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  async startCamera(deviceId = null) {
    if (!this.faceMesh) await this.initialize();

    this.notifyStatus('Abrindo câmera...');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:      { ideal: 1280, min: 640 },
          height:     { ideal: 720,  min: 480 },
          frameRate:  { ideal: 60,   min: 30  },
          facingMode: 'user',
          ...(deviceId ? { deviceId: { exact: deviceId } } : {})
        }
      });

      this.videoElement.srcObject = stream;
      await this.videoElement.play();

      this.isRunning = true;
      this.notifyStatus('⚡ Deep Learning ativo // 60 FPS');
      this._runLoop();
      return true;
    } catch (err) {
      console.error('[EyeTracker] Câmera indisponível:', err);
      this.notifyStatus('Câmera não disponível — usando simulação.');
      if (this.onErrorCallback) this.onErrorCallback(err);
      return false;
    }
  }

  stop() {
    this.isRunning = false;
    if (this.videoElement?.srcObject) {
      this.videoElement.srcObject.getTracks().forEach(t => t.stop());
      this.videoElement.srcObject = null;
    }
    this.notifyStatus('Rastreamento parado.');
  }

  // ─────────────────────────────────────────────────────────────────
  _runLoop() {
    const tick = async () => {
      if (!this.isRunning) return;
      if (this.videoElement.readyState >= 2) {
        const t0 = performance.now();
        try { await this.faceMesh.send({ image: this.videoElement }); }
        catch (e) { console.warn('[EyeTracker] frame skip:', e.message); }
        this.lastLatency = Math.round(performance.now() - t0);
        this._tickFPS();
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  // ─────────────────────────────────────────────────────────────────
  _processResults(results) {
    if (!results.multiFaceLandmarks?.length) {
      this._noFaceFrames++;

      // Recovery: envia último dado válido por até N frames para evitar "flicker"
      if (this._noFaceFrames <= this._maxNoFaceFrames && this._lastValidData) {
        const d = this._lastValidData;
        d.detected = true;
        d.confidence = Math.max(0, 1 - this._noFaceFrames / this._maxNoFaceFrames);
        d.fps = this.fps;
        d.latency = this.lastLatency;
        if (this.onResultsCallback) this.onResultsCallback(d);
      } else {
        if (this.onResultsCallback) this.onResultsCallback({ detected: false, fps: this.fps, latency: this.lastLatency });
      }
      return;
    }

    this._noFaceFrames = 0;
    const lm = results.multiFaceLandmarks[0];

    // Verifica se os marcos de íris existem (refineLandmarks=true)
    if (!lm[468] || !lm[473]) return;

    // ── Íris brutas ───────────────────────────────────────────────
    const rIrisRaw = lm[this.LM.RIGHT_IRIS];
    const lIrisRaw = lm[this.LM.LEFT_IRIS];

    // ── Filtro EMA Adaptativo (velocidade determina alpha) ────────
    const rVelX = Math.abs(rIrisRaw.x - this._rightSmooth.x);
    const lVelX = Math.abs(lIrisRaw.x - this._leftSmooth.x);
    const maxVel = Math.max(rVelX, lVelX);
    // Alpha aumenta com a velocidade: sacada usa 0.45, fixação usa 0.15
    const alpha = this.alphaSlow + (this.alphaFast - this.alphaSlow) * Math.min(1, maxVel * 20);

    this._rightSmooth.x = this._rightSmooth.x * (1 - alpha) + rIrisRaw.x * alpha;
    this._rightSmooth.y = this._rightSmooth.y * (1 - alpha) + rIrisRaw.y * alpha;
    this._rightSmooth.z = this._rightSmooth.z * (1 - alpha) + (rIrisRaw.z || 0) * alpha;

    this._leftSmooth.x = this._leftSmooth.x * (1 - alpha) + lIrisRaw.x * alpha;
    this._leftSmooth.y = this._leftSmooth.y * (1 - alpha) + lIrisRaw.y * alpha;
    this._leftSmooth.z = this._leftSmooth.z * (1 - alpha) + (lIrisRaw.z || 0) * alpha;

    // ── Polígonos Anatômicos dos Olhos ────────────────────────────
    const rightEyePoly = this.LM.RIGHT_EYE.map(i => lm[i]);
    const leftEyePoly  = this.LM.LEFT_EYE.map(i => lm[i]);
    const rightIrisPoly = this.LM.RIGHT_IRIS_P.map(i => lm[i]);
    const leftIrisPoly  = this.LM.LEFT_IRIS_P.map(i => lm[i]);

    // ── Gaze Ratio (posição relativa da íris no olho) ─────────────
    const rightRatios = this._gazeRatio(this._rightSmooth, lm, this.LM.RIGHT_INNER, this.LM.RIGHT_OUTER, this.LM.RIGHT_TOP, this.LM.RIGHT_BOT);
    const leftRatios  = this._gazeRatio(this._leftSmooth, lm, this.LM.LEFT_INNER, this.LM.LEFT_OUTER, this.LM.LEFT_TOP, this.LM.LEFT_BOT);

    const combinedGazeRatio = {
      x: (rightRatios.x + leftRatios.x) / 2,
      y: (rightRatios.y + leftRatios.y) / 2
    };

    // ── EAR (Eye Aspect Ratio) — Detecção de Piscada ─────────────
    const rightEAR = this._calcEAR(lm, this.LM.RIGHT_EAR);
    const leftEAR  = this._calcEAR(lm, this.LM.LEFT_EAR);
    const avgEAR   = (rightEAR + leftEAR) / 2;

    // Calibração contínua da linha de base EAR (olho aberto médio)
    if (avgEAR > 0.22) {
      this.earBaseline = this.earBaseline * 0.998 + avgEAR * 0.002;
    }
    const dynThreshold = this.earBaseline * 0.72; // 72% da abertura normal

    const now = performance.now();
    if (avgEAR < dynThreshold) {
      if (!this.isBlinking) {
        this.isBlinking = true;
        this.blinkCount++;
        this.blinkTimes.push(now);
      }
    } else {
      this.isBlinking = false;
    }
    this.blinkTimes = this.blinkTimes.filter(t => now - t < 60000);

    // ── Pose da Cabeça (Euler angles aproximados) ─────────────────
    const headPose = this._headPose(lm);
    const hAlpha = 0.12;
    this._headSmooth.pitch = this._headSmooth.pitch * (1 - hAlpha) + headPose.pitch * hAlpha;
    this._headSmooth.yaw   = this._headSmooth.yaw   * (1 - hAlpha) + headPose.yaw   * hAlpha;
    this._headSmooth.roll  = this._headSmooth.roll  * (1 - hAlpha) + headPose.roll  * hAlpha;

    // ── Distância estimada (baseada em largura ocular pixel) ──────
    const eyeW = Math.abs(lm[this.LM.RIGHT_OUTER].x - lm[this.LM.LEFT_OUTER].x);
    const metricDistCm = eyeW > 0.01
      ? Math.round(Math.max(30, Math.min(120, (0.23 / eyeW) * 60)))
      : 60;

    const payload = {
      detected:    true,
      timestamp:   now,
      fps:         this.fps,
      latency:     this.lastLatency,
      confidence:  1.0,
      isSimulated: false,

      // Coordenadas suavizadas das íris
      rightPupil: { ...this._rightSmooth },
      leftPupil:  { ...this._leftSmooth  },

      // Polígonos anatômicos
      polygons: {
        rightEye:  rightEyePoly,
        leftEye:   leftEyePoly,
        rightIris: rightIrisPoly,
        leftIris:  leftIrisPoly
      },

      rightRatios,
      leftRatios,
      combinedGazeRatio,

      rightEAR,
      leftEAR,
      avgEAR,
      earBaseline:  this.earBaseline,
      isBlinking:   this.isBlinking,
      blinkCount:   this.blinkCount,
      blinkRateBPM: this.blinkTimes.length,

      headPose:        this._headSmooth,
      metricDistanceCm: metricDistCm,
      allLandmarks:    lm
    };

    this._lastValidData = { ...payload };
    if (this.onResultsCallback) this.onResultsCallback(payload);
  }

  // ── Gaze Ratio: posição normalizada da íris no olho ──────────────
  _gazeRatio(iris, lm, innerIdx, outerIdx, topIdx, botIdx) {
    const inner = lm[innerIdx];
    const outer = lm[outerIdx];
    const top   = lm[topIdx];
    const bot   = lm[botIdx];

    const eyeW = Math.abs(outer.x - inner.x) || 0.001;
    const eyeH = Math.abs(bot.y  - top.y)    || 0.001;

    const minX = Math.min(inner.x, outer.x);
    const minY = Math.min(top.y, bot.y);

    return {
      x: Math.max(0, Math.min(1, (iris.x - minX) / eyeW)),
      y: Math.max(0, Math.min(1, (iris.y - minY) / eyeH))
    };
  }

  // ── Eye Aspect Ratio ──────────────────────────────────────────────
  _calcEAR(lm, pts) {
    const [p1, p2, p3, p4, p5, p6] = pts.map(i => lm[i]);
    const v1 = Math.hypot(p2.x - p6.x, p2.y - p6.y);
    const v2 = Math.hypot(p3.x - p5.x, p3.y - p5.y);
    const h  = Math.hypot(p1.x - p4.x, p1.y - p4.y);
    return h < 1e-6 ? 0.28 : (v1 + v2) / (2 * h);
  }

  // ── Estimativa de Pose da Cabeça ──────────────────────────────────
  _headPose(lm) {
    const nose   = lm[this.LM.NOSE_TIP];
    const chin   = lm[this.LM.CHIN];
    const fore   = lm[this.LM.FOREHEAD];
    const mouthL = lm[this.LM.MOUTH_L];
    const mouthR = lm[this.LM.MOUTH_R];
    const cheekL = lm[this.LM.CHEEK_L];
    const cheekR = lm[this.LM.CHEEK_R];

    const faceH = Math.abs(chin.y - fore.y) || 0.001;
    const faceW = Math.abs(cheekL.x - cheekR.x) || 0.001;

    const mouthCX = (mouthL.x + mouthR.x) / 2;
    const yaw   = Math.max(-50, Math.min(50, (nose.x - mouthCX) * 160));
    const pitch = Math.max(-50, Math.min(50, ((nose.y - fore.y) / faceH - 0.55) * 110));
    const roll  = Math.atan2(mouthL.y - mouthR.y, mouthL.x - mouthR.x) * (180 / Math.PI);

    return { pitch, yaw, roll };
  }

  _tickFPS() {
    this.frameCount++;
    const now = performance.now();
    if (now - this.fpsTimer >= 1000) {
      this.fps = this.frameCount;
      this.frameCount = 0;
      this.fpsTimer = now;
    }
  }

  notifyStatus(msg) {
    if (this.onStatusCallback) this.onStatusCallback(msg);
  }
}
