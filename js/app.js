/**
 * OCULUS-AI // Master Application Orchestration Module
 * Integração completa dos módulos de Deep Learning, 3D FOV, Heatmap, Polígonos e Telemetria.
 * Mirror-fix: canvas context is flipped to match mirrored video display.
 */

import { EyeTracker } from './tracker.js';
import { CalibrationEngine } from './calibration.js';
import { ThreeVisualizer } from './three-fov.js';
import { HeatmapEngine } from './heatmap.js';
import { PolygonEngine } from './polygons.js';
import { TelemetryEngine } from './telemetry.js';
import { EyeSimulator } from './simulator.js';

class OculusApplication {
  constructor() {
    this.tracker = null;
    this.calibration = null;
    this.threeViz = null;
    this.heatmap = null;
    this.polygons = null;
    this.telemetry = null;
    this.simulator = null;

    this.activeMode = 'camera'; // 'camera' | 'simulation'
    this.activeTab = 'heatmap'; // default tab

    // Track last overlay dimensions for resize-on-change
    this._lastOverlayW = 0;
    this._lastOverlayH = 0;

    this.init();
  }

  async init() {
    this.setupDOMElements();
    this.setupEngines();
    this.setupEventListeners();
    this.setupResizeHandlers();

    // Default to simulation mode for Vercel demo
    this.toggleSimulationMode(true);
  }

  setupDOMElements() {
    this.videoElement   = document.getElementById('webcam-video');
    this.videoOverlay   = document.getElementById('video-overlay');
    this.videoOverlayCtx = this.videoOverlay ? this.videoOverlay.getContext('2d') : null;

    this.threeContainer      = document.getElementById('three-container');
    this.heatmapCanvas       = document.getElementById('heatmap-canvas');
    this.aoiCanvas           = document.getElementById('aoi-canvas');
    this.oscilloscopeCanvas  = document.getElementById('oscilloscope-canvas');

    this.calibrationModal    = document.getElementById('calibration-modal');
    this.calibrationTarget   = document.getElementById('calibration-target');
    this.calibrationProgress = document.getElementById('calibration-progress');

    this.badgeStatus   = document.getElementById('badge-status');
    this.badgeFps      = document.getElementById('badge-fps');
    this.badgeLatency  = document.getElementById('badge-latency');
    this.badgeAccuracy = document.getElementById('badge-accuracy');
  }

  setupEngines() {
    // 1. Three.js 3D Visualizer
    if (this.threeContainer) {
      this.threeViz = new ThreeVisualizer(this.threeContainer);
    }

    // 2. Heatmap Engine
    if (this.heatmapCanvas) {
      this.heatmap = new HeatmapEngine(this.heatmapCanvas, {
        radius: 44, opacity: 0.75, decay: 0.003, palette: 'turbo'
      });
    }

    // 3. Polygon / AOI Engine
    if (this.aoiCanvas) {
      this.polygons = new PolygonEngine(this.aoiCanvas);
    }

    // 4. Telemetry / Oscilloscope Engine
    if (this.oscilloscopeCanvas) {
      this.telemetry = new TelemetryEngine(this.oscilloscopeCanvas);
    }

    // 5. Calibration Engine
    this.calibration = new CalibrationEngine({
      pointsCount: 9,
      samplesPerPoint: 20,
      onProgress: (info) => this.handleCalibrationProgress(info),
      onComplete: (res)  => this.handleCalibrationComplete(res)
    });

    // 6. Eye Tracker (MediaPipe Face Mesh + Iris)
    this.tracker = new EyeTracker({
      videoElement: this.videoElement,
      onResults: (data) => this.handleTrackingResults(data),
      onError:   (err)  => this.handleTrackingError(err),
      onStatus:  (msg)  => this.updateStatusBadge(msg)
    });

    // 7. Physiological Simulator
    this.simulator = new EyeSimulator((data) => this.handleTrackingResults(data));
  }

  setupEventListeners() {
    // Tab navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const tab = e.currentTarget.dataset.tab;
        this.switchTab(tab);
      });
    });

    // Camera button
    const btnStartCamera = document.getElementById('btn-start-camera');
    if (btnStartCamera) {
      btnStartCamera.addEventListener('click', () => this.toggleCamera());
    }

    // Simulation button
    const btnSimulation = document.getElementById('btn-toggle-simulation');
    if (btnSimulation) {
      btnSimulation.addEventListener('click', () => {
        const nextState = this.activeMode !== 'simulation';
        this.toggleSimulationMode(nextState);
      });
    }

    // Calibrate button
    const btnCalibrate = document.getElementById('btn-calibrate');
    if (btnCalibrate) {
      btnCalibrate.addEventListener('click', () => this.startCalibrationWizard());
    }

    // Export buttons
    const btnExportCSV = document.getElementById('btn-export-csv');
    if (btnExportCSV) btnExportCSV.addEventListener('click', () => this.exportCSVFile());

    const btnExportJSON = document.getElementById('btn-export-json');
    if (btnExportJSON) btnExportJSON.addEventListener('click', () => this.exportJSONFile());

    const btnSaveHeatmap = document.getElementById('btn-save-heatmap');
    if (btnSaveHeatmap) btnSaveHeatmap.addEventListener('click', () => this.saveHeatmapImage());

    const btnResetHeatmap = document.getElementById('btn-reset-heatmap');
    if (btnResetHeatmap) {
      btnResetHeatmap.addEventListener('click', () => {
        if (this.heatmap)   this.heatmap.reset();
        if (this.polygons)  this.polygons.reset();
      });
    }

    // Heatmap controls
    const paletteSelect = document.getElementById('heatmap-palette');
    if (paletteSelect) {
      paletteSelect.addEventListener('change', (e) => {
        if (this.heatmap) this.heatmap.setPalette(e.target.value);
      });
    }
    const radiusInput = document.getElementById('heatmap-radius');
    if (radiusInput) {
      radiusInput.addEventListener('input', (e) => {
        if (this.heatmap) this.heatmap.radius = parseInt(e.target.value);
      });
    }
    const opacityInput = document.getElementById('heatmap-opacity');
    if (opacityInput) {
      opacityInput.addEventListener('input', (e) => {
        if (this.heatmap) this.heatmap.opacity = parseFloat(e.target.value);
      });
    }
    const decayToggle = document.getElementById('heatmap-decay');
    if (decayToggle) {
      decayToggle.addEventListener('change', (e) => {
        if (this.heatmap) this.heatmap.decay = e.target.checked ? 0.005 : 0;
      });
    }
    const scanpathToggle = document.getElementById('heatmap-scanpath');
    if (scanpathToggle) {
      scanpathToggle.addEventListener('change', (e) => {
        if (this.heatmap) this.heatmap.showScanpath = e.target.checked;
      });
    }
    const bgUpload = document.getElementById('bg-image-upload');
    if (bgUpload) {
      bgUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
          const reader = new FileReader();
          reader.onload = (ev) => {
            const img = new Image();
            img.onload = () => { if (this.heatmap) this.heatmap.setBackground(img); };
            img.src = ev.target.result;
          };
          reader.readAsDataURL(file);
        }
      });
    }

    // 3D controls
    const btnReset3D = document.getElementById('btn-reset-3d');
    if (btnReset3D && this.threeViz) {
      btnReset3D.addEventListener('click', () => this.threeViz.resetCamera());
    }
    const toggleCones = document.getElementById('toggle-fov-cones');
    if (toggleCones && this.threeViz) {
      toggleCones.addEventListener('change', (e) => this.threeViz.toggleFOVCones(e.target.checked));
    }
    const toggleFaceMesh = document.getElementById('toggle-face-mesh');
    if (toggleFaceMesh && this.threeViz) {
      toggleFaceMesh.addEventListener('change', (e) => this.threeViz.toggleFaceCloud(e.target.checked));
    }

    // Calibration cancel
    const btnCancelCalib = document.getElementById('btn-cancel-calibration');
    if (btnCancelCalib) {
      btnCancelCalib.addEventListener('click', () => this.cancelCalibration());
    }
  }

  setupResizeHandlers() {
    window.addEventListener('resize', () => {
      if (this.threeViz) this.threeViz.onResize();
      if (this.heatmap)  this.heatmap.resize();
      if (this.polygons) this.polygons.resize();
      if (this.telemetry) this.telemetry.resize();
      this.resizeVideoOverlay();
    });
    this.resizeVideoOverlay();
  }

  /**
   * MIRROR FIX: Size the overlay canvas to match the container element,
   * not the raw video element — so coordinates align correctly with the
   * CSS-mirrored video (transform: scaleX(-1)).
   */
  resizeVideoOverlay() {
    const video = this.videoElement;
    const container = video ? video.parentElement : null;
    if (!container || !this.videoOverlay) return;
    const rect = container.getBoundingClientRect();
    this.videoOverlay.width  = rect.width  || 640;
    this.videoOverlay.height = rect.height || 480;
  }

  switchTab(tabId) {
    this.activeTab = tabId;

    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === tabId);
    });
    document.querySelectorAll('.tab-content').forEach(c => {
      c.classList.toggle('active', c.id === `tab-${tabId}`);
    });

    if (tabId === 'heatmap'   && this.heatmap)  setTimeout(() => { this.heatmap.resize(); this.heatmap.render(); }, 50);
    if (tabId === 'aoi'       && this.polygons)  setTimeout(() => this.polygons.resize(), 50);
    if (tabId === 'telemetry' && this.telemetry) setTimeout(() => this.telemetry.resize(), 50);
  }

  async toggleCamera() {
    const btn = document.getElementById('btn-start-camera');

    if (this.tracker && this.tracker.isRunning) {
      this.tracker.stop();
      if (btn) btn.innerHTML = `<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3" stroke-width="2" stroke-linejoin="round"/></svg> Câmera`;
      this.updateStatusBadge('Câmera parada.');
    } else {
      if (this.simulator && this.simulator.isActive) {
        this.simulator.stop();
        this.activeMode = 'camera';
        const simBtn = document.getElementById('btn-toggle-simulation');
        if (simBtn) simBtn.classList.remove('active');
      }
      this.updateStatusBadge('Iniciando webcam...');
      const success = await this.tracker.startCamera();
      if (success) {
        this.activeMode = 'camera';
        if (btn) btn.innerHTML = `<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg> Parar`;
        this.updateStatusBadge('Deep Learning ativo // 60 FPS');
      } else {
        this.toggleSimulationMode(true);
      }
    }
  }

  toggleSimulationMode(enable) {
    const simBtn = document.getElementById('btn-toggle-simulation');
    const camBtn = document.getElementById('btn-start-camera');

    if (enable) {
      if (this.tracker && this.tracker.isRunning) {
        this.tracker.stop();
        if (camBtn) camBtn.innerHTML = `<svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><polygon points="5 3 19 12 5 21 5 3" stroke-width="2" stroke-linejoin="round"/></svg> Câmera`;
      }
      this.activeMode = 'simulation';
      this.simulator.start();
      if (simBtn) {
        simBtn.classList.add('active');
        simBtn.innerHTML = `<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Simulando`;
      }
      this.updateStatusBadge('Simulação 3D Ativa // Dados Fisiológicos');
    } else {
      this.simulator.stop();
      this.activeMode = 'idle';
      if (simBtn) {
        simBtn.classList.remove('active');
        simBtn.innerHTML = `<svg width="12" height="12" fill="currentColor" viewBox="0 0 24 24"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg> Simular`;
      }
      this.updateStatusBadge('Pronto para iniciar.');
    }
  }

  /**
   * Master tracking results handler — called every inference frame.
   * MIRROR FIX: the 2D canvas context is translated+scaled to mirror X
   * so polygon coordinates (from raw MediaPipe) align with the
   * CSS-mirrored video element (transform: scaleX(-1)).
   */
  handleTrackingResults(data) {
    if (!data || !data.detected) return;

    // Resize overlay if container dimensions changed
    const video     = this.videoElement;
    const container = video ? video.parentElement : null;
    if (container) {
      const rect = container.getBoundingClientRect();
      const cw = rect.width  || 640;
      const ch = rect.height || 480;
      if (cw !== this._lastOverlayW || ch !== this._lastOverlayH) {
        this._lastOverlayW = cw;
        this._lastOverlayH = ch;
        if (this.videoOverlay) {
          this.videoOverlay.width  = cw;
          this.videoOverlay.height = ch;
        }
      }
    }

    // 1. Gaze prediction (polynomial calibration or heuristic)
    const gazePrediction = this.calibration.predict(data);
    if (this.calibration.isCalibrating) {
      this.calibration.collectSample(data);
    }

    // 2. Three.js 3D Update
    if (this.threeViz) {
      this.threeViz.update(data, gazePrediction);
    }

    // 3. Heatmap & Scanpath
    if (this.heatmap) {
      this.heatmap.addPoint(gazePrediction.screenX, gazePrediction.screenY, data.timestamp);
      if (this.activeTab === 'heatmap') this.heatmap.render();
    }

    // 4. Polygon AOI update
    if (this.polygons) {
      this.polygons.update(gazePrediction.screenX, gazePrediction.screenY);
      if (this.activeTab === 'aoi') {
        this.polygons.renderAOIOverlay();
        this.updateAOITable();
      }
    }

    // 5. MIRROR FIX — draw eye anatomy with flipped canvas context
    if (this.videoOverlayCtx && this.videoOverlay) {
      const vw = this.videoOverlay.width;
      const vh = this.videoOverlay.height;
      this.videoOverlayCtx.clearRect(0, 0, vw, vh);
      this.videoOverlayCtx.save();
      // Flip X so that raw MediaPipe coords align with the CSS-mirrored video
      this.videoOverlayCtx.translate(vw, 0);
      this.videoOverlayCtx.scale(-1, 1);
      if (this.polygons) {
        this.polygons.renderEyeAnatomy(this.videoOverlayCtx, data, vw, vh);
      }
      this.videoOverlayCtx.restore();
    }

    // 6. Telemetry
    if (this.telemetry) {
      const teleData = this.telemetry.recordFrame(data, gazePrediction);
      this.updateTelemetryHUD(teleData, data);
    }

    // 7. Top badges
    if (this.badgeFps)     this.badgeFps.textContent     = `${data.fps || 60}`;
    if (this.badgeLatency) this.badgeLatency.textContent  = `${data.latency || 4}ms`;
    if (this.badgeAccuracy && gazePrediction.accuracyScore) {
      this.badgeAccuracy.textContent = `${gazePrediction.accuracyScore}%`;
    }

    // 8. Camera panel quick stats
    const csEar    = document.getElementById('cs-ear');
    const csBlinks = document.getElementById('cs-blinks');
    const csFace   = document.getElementById('cs-face-status');
    if (csEar && data.ear) csEar.textContent = (data.ear.avg || 0.28).toFixed(2);
    if (csBlinks) csBlinks.textContent = data.blinks ? data.blinks.count || 0 : 0;
    if (csFace) {
      csFace.textContent = data.detected ? 'FACE ✓' : 'NO FACE';
      csFace.style.color = data.detected ? '#00ff88' : '#ff2d78';
    }
  }

  updateTelemetryHUD(tele, rawData) {
    const elLeftMm = document.getElementById('hud-left-mm');
    if (elLeftMm && tele.leftIris3D.mm) {
      elLeftMm.textContent = `X:${tele.leftIris3D.mm.x} Y:${tele.leftIris3D.mm.y} Z:${tele.leftIris3D.mm.z}`;
    }

    const elRightMm = document.getElementById('hud-right-mm');
    if (elRightMm && tele.rightIris3D.mm) {
      elRightMm.textContent = `X:${tele.rightIris3D.mm.x} Y:${tele.rightIris3D.mm.y} Z:${tele.rightIris3D.mm.z}`;
    }

    const elGazePx = document.getElementById('hud-gaze-px');
    if (elGazePx) {
      elGazePx.textContent = `X:${tele.screenGaze.x}px Y:${tele.screenGaze.y}px (${tele.screenGaze.normX}, ${tele.screenGaze.normY})`;
    }

    const elVisualAngle = document.getElementById('hud-visual-angle');
    if (elVisualAngle) elVisualAngle.textContent = `${tele.visualAngleDeg}°`;

    const elIPD = document.getElementById('hud-ipd');
    if (elIPD) elIPD.textContent = `${tele.ipd_mm}mm`;

    const elHeadPose = document.getElementById('hud-head-pose');
    if (elHeadPose) {
      elHeadPose.textContent = `P:${Math.round(tele.headPose.pitch)}° Y:${Math.round(tele.headPose.yaw)}° R:${Math.round(tele.headPose.roll)}°`;
    }

    const elBlinks = document.getElementById('hud-blinks');
    if (elBlinks) {
      elBlinks.textContent = `${tele.blinks.count} (${tele.blinks.bpm}BPM) EAR:${tele.ear.avg}`;
    }

    // RMSE (not displayed in new bottom bar, kept for compat)
    const elRmse = document.getElementById('hud-rmse');
    if (elRmse) elRmse.textContent = `${tele.precision.rmse}px (${tele.precision.confidencePct}%)`;
  }

  updateAOITable() {
    const tableBody = document.getElementById('aoi-table-body');
    if (!tableBody || !this.polygons) return;

    const stats = this.polygons.getAOIStats();
    tableBody.innerHTML = stats.map(s => `
      <tr>
        <td style="display:flex;align-items:center;gap:5px">
          <span style="width:8px;height:8px;border-radius:2px;background:${s.color};display:inline-block"></span>
          ${s.name}
        </td>
        <td style="color:#00f5ff">${s.dwellTimeSec}s</td>
        <td style="color:#94a3b8">${s.attentionPct}%</td>
        <td style="color:#94a3b8">${s.fixationCount}</td>
        <td style="color:#64748b">${s.ttffMs}ms</td>
      </tr>
    `).join('');
  }

  // ─── Calibration Wizard ──────────────────────────────────────────

  startCalibrationWizard() {
    if (!this.calibrationModal || !this.calibrationTarget) return;
    this.calibrationModal.classList.add('open');
    this.calibration.startCalibration(9);
  }

  handleCalibrationProgress(info) {
    if (!this.calibrationTarget || !this.calibrationProgress) return;
    const pt = info.point;
    this.calibrationTarget.style.left = `${pt.x * 100}%`;
    this.calibrationTarget.style.top  = `${pt.y * 100}%`;
    const pct = Math.round(
      ((info.pointIndex * info.targetSamples + info.samplesCount)
       / (info.totalPoints * info.targetSamples)) * 100
    );
    this.calibrationProgress.textContent =
      `Ponto ${info.pointIndex + 1}/${info.totalPoints} — ${pct}% concluído.`;
  }

  handleCalibrationComplete(res) {
    if (this.calibrationModal) this.calibrationModal.classList.remove('open');
    this.updateStatusBadge(`Calibração: ${res.accuracyScore}% precisão (RMSE: ${res.rmsePixels}px)`);
    if (this.badgeAccuracy) this.badgeAccuracy.textContent = `${res.accuracyScore}%`;
  }

  cancelCalibration() {
    if (this.calibrationModal) this.calibrationModal.classList.remove('open');
    this.calibration.reset();
    this.updateStatusBadge('Calibração cancelada.');
  }

  // ─── Exports ─────────────────────────────────────────────────────

  exportCSVFile() {
    if (!this.telemetry) return;
    this.downloadFile(this.telemetry.exportCSV(), `oculus_telemetry_${Date.now()}.csv`, 'text/csv');
  }

  exportJSONFile() {
    if (!this.telemetry) return;
    const aoiStats = this.polygons ? this.polygons.getAOIStats() : [];
    this.downloadFile(this.telemetry.exportJSON(aoiStats), `oculus_dataset_${Date.now()}.json`, 'application/json');
  }

  saveHeatmapImage() {
    if (!this.heatmap) return;
    const link = document.createElement('a');
    link.download = `oculus_heatmap_${Date.now()}.png`;
    link.href = this.heatmap.exportImage();
    link.click();
  }

  downloadFile(content, fileName, contentType) {
    const blob = new Blob([content], { type: contentType });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  handleTrackingError(err) {
    console.warn('[App] Erro de rastreamento:', err);
    this.updateStatusBadge('Webcam não disponível — Simulação 3D iniciada');
    this.toggleSimulationMode(true);
  }

  updateStatusBadge(text) {
    if (this.badgeStatus) this.badgeStatus.textContent = text;
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.oculusApp = new OculusApplication();
});
