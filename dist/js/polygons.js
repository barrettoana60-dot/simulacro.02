/**
 * OCULUS-AI // Biometric Polygon & AOI Engine v2.0
 * Polígonos anatômicos com coordenadas MediaPipe, Convex Hull, AOIs com Ray-Casting.
 */

export class PolygonEngine {
  constructor(overlayCanvas) {
    this.canvas = overlayCanvas;
    this.ctx    = this.canvas.getContext('2d');

    // AOIs pré-definidas (coords normalizadas 0-1)
    this.aois = [
      { id: 'A', name: 'Zona A // Header',       color: '#00f5ff', points: [{x:0.12,y:0.06},{x:0.88,y:0.06},{x:0.88,y:0.24},{x:0.12,y:0.24}], dwellMs:0, fixes:0, ttff:null, looking:false },
      { id: 'B', name: 'Zona B // Centro',        color: '#ff2d78', points: [{x:0.28,y:0.32},{x:0.72,y:0.32},{x:0.75,y:0.72},{x:0.25,y:0.72}], dwellMs:0, fixes:0, ttff:null, looking:false },
      { id: 'C', name: 'Zona C // CTA / Ação',   color: '#00ff88', points: [{x:0.33,y:0.78},{x:0.67,y:0.78},{x:0.67,y:0.94},{x:0.33,y:0.94}], dwellMs:0, fixes:0, ttff:null, looking:false }
    ];

    this.gazeHistory = [];
    this.maxHistory  = 150;
    this.convexHull  = [];

    this.totalMs = 0;
    this.lastT   = performance.now();
  }

  resize() {
    const p = this.canvas.parentElement;
    if (p) { this.canvas.width = p.clientWidth; this.canvas.height = p.clientHeight; }
  }

  // ── Atualização por Frame ──────────────────────────────────────────
  update(gx, gy) {
    const now = performance.now();
    const dt  = now - this.lastT;
    this.lastT = now;
    this.totalMs += dt;

    if (isNaN(gx) || isNaN(gy)) return;

    const r    = this.canvas.getBoundingClientRect();
    const normX = (gx - r.left) / this.canvas.width;
    const normY = (gy - r.top)  / this.canvas.height;

    // Histórico para Convex Hull
    if (normX >= 0 && normX <= 1 && normY >= 0 && normY <= 1) {
      this.gazeHistory.push({ x: normX, y: normY });
      if (this.gazeHistory.length > this.maxHistory) this.gazeHistory.shift();
      this.convexHull = this._hull(this.gazeHistory);
    }

    // AOI hit-test
    this.aois.forEach(aoi => {
      const inside = this._pip({ x: normX, y: normY }, aoi.points);
      if (inside) {
        aoi.dwellMs += dt;
        if (!aoi.looking) { aoi.looking = true; aoi.fixes++; aoi.ttff = aoi.ttff ?? Math.round(this.totalMs); }
      } else {
        aoi.looking = false;
      }
    });
  }

  // ── Renderiza Polígonos Anatômicos dos Olhos ─────────────────────
  // ctx já virá com transform de espelho aplicado pelo app.js
  renderEyeAnatomy(ctx, eyeData, vw, vh) {
    if (!eyeData?.detected || !eyeData.polygons) return;
    const { rightEye, leftEye, rightIris, leftIris } = eyeData.polygons;

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Contorno dos olhos
    this._drawPoly(ctx, rightEye, vw, vh, { stroke: '#00f5ff', fill: 'rgba(0,245,255,0.10)', lw: 2.0 });
    this._drawPoly(ctx, leftEye,  vw, vh, { stroke: '#00f5ff', fill: 'rgba(0,245,255,0.10)', lw: 2.0 });

    // Íris
    this._drawPoly(ctx, rightIris, vw, vh, { stroke: '#ff2d78', fill: 'rgba(255,45,120,0.25)', lw: 2.2 });
    this._drawPoly(ctx, leftIris,  vw, vh, { stroke: '#ff2d78', fill: 'rgba(255,45,120,0.25)', lw: 2.2 });

    // Centro das pupilas — retículo de precisão
    if (eyeData.rightPupil) this._reticle(ctx, eyeData.rightPupil.x * vw, eyeData.rightPupil.y * vh, '#ffffff', 8);
    if (eyeData.leftPupil)  this._reticle(ctx, eyeData.leftPupil.x  * vw, eyeData.leftPupil.y  * vh, '#ffffff', 8);

    ctx.restore();
  }

  _drawPoly(ctx, pts, w, h, s) {
    if (!pts || pts.length < 3) return;
    ctx.beginPath();
    ctx.moveTo(pts[0].x * w, pts[0].y * h);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x * w, pts[i].y * h);
    ctx.closePath();
    if (s.fill)   { ctx.fillStyle = s.fill;   ctx.fill(); }
    if (s.stroke) { ctx.strokeStyle = s.stroke; ctx.lineWidth = s.lw || 1.5; ctx.stroke(); }
  }

  _reticle(ctx, x, y, color, r) {
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.8;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.moveTo(x - r * 1.7, y); ctx.lineTo(x + r * 1.7, y);
    ctx.moveTo(x, y - r * 1.7); ctx.lineTo(x, y + r * 1.7);
    ctx.stroke();
  }

  // ── Renderiza AOIs e Convex Hull ──────────────────────────────────
  renderAOIOverlay() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    // Convex hull (campo de exploração visual)
    if (this.convexHull.length >= 3) {
      this.ctx.save();
      this.ctx.beginPath();
      this.ctx.moveTo(this.convexHull[0].x * w, this.convexHull[0].y * h);
      this.convexHull.slice(1).forEach(p => this.ctx.lineTo(p.x * w, p.y * h));
      this.ctx.closePath();
      this.ctx.fillStyle = 'rgba(0,245,255,0.05)';
      this.ctx.fill();
      this.ctx.strokeStyle = 'rgba(0,245,255,0.45)';
      this.ctx.lineWidth = 1.5;
      this.ctx.setLineDash([5, 5]);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
      this.ctx.restore();
    }

    // AOIs
    this.aois.forEach(aoi => {
      if (aoi.points.length < 3) return;
      this.ctx.save();

      this.ctx.beginPath();
      this.ctx.moveTo(aoi.points[0].x * w, aoi.points[0].y * h);
      aoi.points.slice(1).forEach(p => this.ctx.lineTo(p.x * w, p.y * h));
      this.ctx.closePath();

      this.ctx.fillStyle   = aoi.looking ? this._rgba(aoi.color, 0.22) : this._rgba(aoi.color, 0.07);
      this.ctx.strokeStyle = aoi.looking ? aoi.color : this._rgba(aoi.color, 0.6);
      this.ctx.lineWidth   = aoi.looking ? 3.5 : 2.0;
      this.ctx.shadowColor = aoi.looking ? aoi.color : 'transparent';
      this.ctx.shadowBlur  = aoi.looking ? 14 : 0;
      this.ctx.fill();
      this.ctx.stroke();

      // Label
      const c = this._center(aoi.points);
      const cx = c.x * w, cy = c.y * h;
      const dw   = (aoi.dwellMs / 1000).toFixed(1);
      const pct  = this.totalMs > 0 ? Math.round(aoi.dwellMs / this.totalMs * 100) : 0;

      this.ctx.shadowBlur = 0;
      this.ctx.fillStyle  = 'rgba(4,8,18,0.88)';
      this.ctx.strokeStyle = aoi.color;
      this.ctx.lineWidth   = 1;
      this.ctx.beginPath();
      this.ctx.roundRect(cx - 76, cy - 20, 152, 40, 8);
      this.ctx.fill();
      this.ctx.stroke();

      this.ctx.fillStyle  = '#fff';
      this.ctx.font       = 'bold 11px Inter, sans-serif';
      this.ctx.textAlign  = 'center';
      this.ctx.fillText(aoi.name, cx, cy - 4);
      this.ctx.fillStyle = aoi.color;
      this.ctx.font      = '10px JetBrains Mono, monospace';
      this.ctx.fillText(`${dw}s (${pct}%) • ${aoi.fixes} fix`, cx, cy + 12);
      this.ctx.restore();
    });
  }

  getAOIStats() {
    return this.aois.map(a => ({
      id: a.id, name: a.name, color: a.color,
      dwellTimeSec: (a.dwellMs / 1000).toFixed(2),
      attentionPct: this.totalMs > 0 ? Number(((a.dwellMs / this.totalMs) * 100).toFixed(1)) : 0,
      fixationCount: a.fixes,
      ttffMs: a.ttff ?? 'N/A'
    }));
  }

  reset() {
    this.gazeHistory = []; this.convexHull = [];
    this.totalMs = 0;      this.lastT = performance.now();
    this.aois.forEach(a => { a.dwellMs = 0; a.fixes = 0; a.ttff = null; a.looking = false; });
  }

  // ── Algoritmos ────────────────────────────────────────────────────

  _pip(pt, vs) {
    // Ray-Casting (Jordan Curve Theorem)
    let inside = false;
    const { x, y } = pt;
    for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
      const { x: xi, y: yi } = vs[i];
      const { x: xj, y: yj } = vs[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / ((yj - yi) || 1e-9) + xi)
        inside = !inside;
    }
    return inside;
  }

  _hull(pts) {
    // Graham Scan O(N log N)
    if (pts.length < 3) return [...pts];
    const sorted = [...pts];
    let lo = 0;
    sorted.forEach((p, i) => { if (p.y < sorted[lo].y || (p.y === sorted[lo].y && p.x < sorted[lo].x)) lo = i; });
    const start = sorted.splice(lo, 1)[0];
    sorted.sort((a, b) => {
      const A = Math.atan2(a.y - start.y, a.x - start.x);
      const B = Math.atan2(b.y - start.y, b.x - start.x);
      return A - B;
    });
    const stack = [start, sorted[0], sorted[1]];
    for (let i = 2; i < sorted.length; i++) {
      while (stack.length >= 2) {
        const [p1, p2] = [stack[stack.length-2], stack[stack.length-1]];
        if ((p2.x-p1.x)*(sorted[i].y-p1.y) - (p2.y-p1.y)*(sorted[i].x-p1.x) <= 0) stack.pop();
        else break;
      }
      stack.push(sorted[i]);
    }
    return stack;
  }

  _center(pts) {
    const sx = pts.reduce((s, p) => s + p.x, 0);
    const sy = pts.reduce((s, p) => s + p.y, 0);
    return { x: sx / pts.length, y: sy / pts.length };
  }

  _rgba(hex, a) {
    const c = hex.replace('#', '');
    const r = parseInt(c.slice(0,2), 16);
    const g = parseInt(c.slice(2,4), 16);
    const b = parseInt(c.slice(4,6), 16);
    return `rgba(${r},${g},${b},${a})`;
  }
}
