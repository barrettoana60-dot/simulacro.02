/**
 * OCULUS-AI // Calibration & Polynomial Gaze Mapping Engine
 * Mapeia vetores de íris + pose cefálica em coordenadas de tela (X, Y)
 * através de regressão polinomial bivariada regularizada (Ridge Regression).
 */

export class CalibrationEngine {
  constructor(options = {}) {
    this.pointsCount = options.pointsCount || 9;
    this.samplesPerPoint = options.samplesPerPoint || 25;
    this.onProgress = options.onProgress || null;
    this.onComplete = options.onComplete || null;

    this.isCalibrating = false;
    this.isCalibrated = false;
    this.currentPointIndex = 0;
    this.currentSamples = [];

    // Grid de calibração padrão de 9 pontos (normalizado 0.0 - 1.0)
    this.grid9Points = [
      { x: 0.1, y: 0.1 }, { x: 0.5, y: 0.1 }, { x: 0.9, y: 0.1 },
      { x: 0.1, y: 0.5 }, { x: 0.5, y: 0.5 }, { x: 0.9, y: 0.5 },
      { x: 0.1, y: 0.9 }, { x: 0.5, y: 0.9 }, { x: 0.9, y: 0.9 }
    ];

    // Grid de calibração de 5 pontos rápidos
    this.grid5Points = [
      { x: 0.5, y: 0.5 }, // Centro
      { x: 0.15, y: 0.15 }, // Topo Esquerdo
      { x: 0.85, y: 0.15 }, // Topo Direito
      { x: 0.15, y: 0.85 }, // Fundo Esquerdo
      { x: 0.85, y: 0.85 }  // Fundo Direito
    ];

    this.activeGrid = this.grid9Points;

    // Coeficientes do modelo polinomial treinado
    this.modelX = null;
    this.modelY = null;
    this.accuracyScore = 0; // %
    this.rmsePixels = 0;

    // Filtro Kalman 1D simples para coordenadas finais de tela
    this.smoothScreenX = window.innerWidth / 2;
    this.smoothScreenY = window.innerHeight / 2;
    this.filterAlpha = 0.28;
  }

  /**
   * Inicia o processo guiado de calibração
   */
  startCalibration(points = 9) {
    this.pointsCount = points;
    this.activeGrid = points === 5 ? this.grid5Points : this.grid9Points;
    this.isCalibrating = true;
    this.isCalibrated = false;
    this.currentPointIndex = 0;
    this.currentSamples = [];
    this.allTrainingData = [];

    this.notifyPointUpdate();
  }

  /**
   * Coleta uma amostra de olhar durante a calibração
   */
  collectSample(eyeData) {
    if (!this.isCalibrating || !eyeData.detected) return;

    const target = this.activeGrid[this.currentPointIndex];
    const targetPx = {
      x: target.x * window.innerWidth,
      y: target.y * window.innerHeight
    };

    // Vetor de features biométricas
    const feature = this.extractFeatures(eyeData);

    this.currentSamples.push({
      features: feature,
      target: targetPx
    });

    if (this.onProgress) {
      this.onProgress({
        pointIndex: this.currentPointIndex,
        totalPoints: this.activeGrid.length,
        samplesCount: this.currentSamples.length,
        targetSamples: this.samplesPerPoint,
        point: target
      });
    }

    if (this.currentSamples.length >= this.samplesPerPoint) {
      this.allTrainingData.push(...this.currentSamples);
      this.currentSamples = [];
      this.currentPointIndex++;

      if (this.currentPointIndex >= this.activeGrid.length) {
        this.finishCalibration();
      } else {
        this.notifyPointUpdate();
      }
    }
  }

  notifyPointUpdate() {
    if (this.onProgress) {
      this.onProgress({
        pointIndex: this.currentPointIndex,
        totalPoints: this.activeGrid.length,
        samplesCount: 0,
        targetSamples: this.samplesPerPoint,
        point: this.activeGrid[this.currentPointIndex]
      });
    }
  }

  /**
   * Extrai vetor de características:
   * [1, u, v, u^2, v^2, u*v, headYaw, headPitch]
   */
  extractFeatures(eyeData) {
    const u = eyeData.combinedGazeRatio ? eyeData.combinedGazeRatio.x : 0.5;
    const v = eyeData.combinedGazeRatio ? eyeData.combinedGazeRatio.y : 0.5;
    const yaw = eyeData.headPose ? (eyeData.headPose.yaw / 45.0) : 0;
    const pitch = eyeData.headPose ? (eyeData.headPose.pitch / 45.0) : 0;

    return [
      1.0,
      u,
      v,
      u * u,
      v * v,
      u * v,
      yaw,
      pitch
    ];
  }

  /**
   * Treina os modelos de regressão polinomial regularizada para X e Y
   */
  finishCalibration() {
    this.isCalibrating = false;

    if (this.allTrainingData.length < 10) {
      console.warn('[Calibration] Amostras insuficientes para treinamento.');
      return;
    }

    const n = this.allTrainingData.length;
    const p = this.allTrainingData[0].features.length;

    // Matriz de features X_mat (n x p) e vetores alvo Y_x e Y_y
    const X_mat = this.allTrainingData.map(d => d.features);
    const Y_x = this.allTrainingData.map(d => d.target.x);
    const Y_y = this.allTrainingData.map(d => d.target.y);

    // Ridge Regression: W = (X^T * X + lambda * I)^(-1) * X^T * Y
    const lambda = 0.001;
    this.modelX = this.solveRidge(X_mat, Y_x, lambda);
    this.modelY = this.solveRidge(X_mat, Y_y, lambda);

    // Avaliação de Erro Quadrático Médio (RMSE)
    let totalErrorSq = 0;
    for (const d of this.allTrainingData) {
      const predX = this.dot(d.features, this.modelX);
      const predY = this.dot(d.features, this.modelY);
      const err = Math.hypot(predX - d.target.x, predY - d.target.y);
      totalErrorSq += err * err;
    }

    this.rmsePixels = Math.sqrt(totalErrorSq / n);
    const diagonal = Math.hypot(window.innerWidth, window.innerHeight);
    this.accuracyScore = Math.max(70, Math.min(99.5, 100 - (this.rmsePixels / diagonal) * 100 * 4));

    this.isCalibrated = true;

    if (this.onComplete) {
      this.onComplete({
        accuracyScore: Number(this.accuracyScore.toFixed(1)),
        rmsePixels: Number(this.rmsePixels.toFixed(1)),
        totalSamples: n
      });
    }
  }

  /**
   * Prediz as coordenadas de tela a partir dos dados do olho
   */
  predict(eyeData) {
    if (!eyeData || !eyeData.detected) {
      return {
        screenX: this.smoothScreenX,
        screenY: this.smoothScreenY,
        normX: this.smoothScreenX / window.innerWidth,
        normY: this.smoothScreenY / window.innerHeight,
        calibrated: this.isCalibrated
      };
    }

    let targetX, targetY;

    if (this.isCalibrated && this.modelX && this.modelY) {
      const feats = this.extractFeatures(eyeData);
      targetX = this.dot(feats, this.modelX);
      targetY = this.dot(feats, this.modelY);
    } else {
      // Mapeamento linear heurístico pré-calibração com compensação de cabeça
      const ratioX = eyeData.combinedGazeRatio ? eyeData.combinedGazeRatio.x : 0.5;
      const ratioY = eyeData.combinedGazeRatio ? eyeData.combinedGazeRatio.y : 0.5;
      const yaw = eyeData.headPose ? eyeData.headPose.yaw : 0;
      const pitch = eyeData.headPose ? eyeData.headPose.pitch : 0;

      // Inversão natural do eixo da câmera frontal
      const mappedX = (1.0 - ratioX) * 1.6 - 0.3 - (yaw * 0.015);
      const mappedY = ratioY * 1.6 - 0.3 + (pitch * 0.012);

      targetX = mappedX * window.innerWidth;
      targetY = mappedY * window.innerHeight;
    }

    // Clamping dentro da tela estendida
    targetX = Math.max(0, Math.min(window.innerWidth, targetX));
    targetY = Math.max(0, Math.min(window.innerHeight, targetY));

    // Suavização adaptativa (filtro passa-baixas exponencial)
    this.smoothScreenX = this.smoothScreenX * (1 - this.filterAlpha) + targetX * this.filterAlpha;
    this.smoothScreenY = this.smoothScreenY * (1 - this.filterAlpha) + targetY * this.filterAlpha;

    return {
      screenX: Math.round(this.smoothScreenX),
      screenY: Math.round(this.smoothScreenY),
      normX: Math.max(0, Math.min(1, this.smoothScreenX / window.innerWidth)),
      normY: Math.max(0, Math.min(1, this.smoothScreenY / window.innerHeight)),
      calibrated: this.isCalibrated,
      accuracyScore: this.accuracyScore,
      rmsePixels: this.rmsePixels
    };
  }

  // --- Álgebra Linear para Ridge Regression ---

  solveRidge(X, y, lambda) {
    const n = X.length;
    const p = X[0].length;

    // XtX = X^T * X
    const XtX = Array.from({ length: p }, () => new Float64Array(p));
    for (let i = 0; i < p; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += X[k][i] * X[k][j];
        }
        XtX[i][j] = sum + (i === j ? lambda : 0);
      }
    }

    // Xty = X^T * y
    const Xty = new Float64Array(p);
    for (let i = 0; i < p; i++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += X[k][i] * y[k];
      }
      Xty[i] = sum;
    }

    // Eliminação Gaussiana com Pivoteamento Parcial
    return this.gaussianElimination(XtX, Xty);
  }

  gaussianElimination(A, b) {
    const n = b.length;
    for (let i = 0; i < n; i++) {
      let maxEl = Math.abs(A[i][i]);
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(A[k][i]) > maxEl) {
          maxEl = Math.abs(A[k][i]);
          maxRow = k;
        }
      }

      for (let k = i; k < n; k++) {
        const tmp = A[maxRow][k];
        A[maxRow][k] = A[i][k];
        A[i][k] = tmp;
      }
      const tmp = b[maxRow];
      b[maxRow] = b[i];
      b[i] = tmp;

      if (Math.abs(A[i][i]) < 1e-12) {
        A[i][i] = 1e-6; // Regularização para evitar divisão por zero
      }

      for (let k = i + 1; k < n; k++) {
        const c = -A[k][i] / A[i][i];
        for (let j = i; j < n; j++) {
          if (i === j) {
            A[k][j] = 0;
          } else {
            A[k][j] += c * A[i][j];
          }
        }
        b[k] += c * b[i];
      }
    }

    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      x[i] = b[i] / A[i][i];
      for (let k = i - 1; k >= 0; k--) {
        b[k] -= A[k][i] * x[i];
      }
    }
    return Array.from(x);
  }

  dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  reset() {
    this.isCalibrated = false;
    this.isCalibrating = false;
    this.modelX = null;
    this.modelY = null;
    this.accuracyScore = 0;
    this.rmsePixels = 0;
  }
}
