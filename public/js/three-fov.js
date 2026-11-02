/**
 * OCULUS-AI // 3D Field of View (FOV) Engine v2.0 (Three.js)
 * Globos oculares anatômicos, cones FOV vistosos, anéis orbitais, nuvem facial.
 */

export class ThreeVisualizer {
  constructor(containerElement) {
    this.container = containerElement;

    this.scene    = null;
    this.camera   = null;
    this.renderer = null;
    this.controls = null;
    this.animId   = null;

    this.leftEyeball  = null;
    this.rightEyeball = null;
    this.fovGroup     = null;
    this.faceCloud    = null;
    this.screenPlane  = null;
    this.hitPoint     = null;
    this.leftRing     = null;
    this.rightRing    = null;

    this.showCones     = true;
    this.showFaceCloud = true;

    this._t = 0; // animation tick

    this.init();
  }

  init() {
    const W = this.container.clientWidth  || 500;
    const H = this.container.clientHeight || 400;

    // ── Scene ─────────────────────────────────────────────────────
    this.scene = new window.THREE.Scene();
    this.scene.background = new window.THREE.Color(0x040812);
    this.scene.fog = new window.THREE.FogExp2(0x040812, 0.0009);

    // ── Camera ────────────────────────────────────────────────────
    this.camera = new window.THREE.PerspectiveCamera(42, W / H, 0.5, 3000);
    this.camera.position.set(0, 25, 240);

    // ── Renderer ──────────────────────────────────────────────────
    this.renderer = new window.THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setSize(W, H);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = false;
    this.container.appendChild(this.renderer.domElement);

    // ── OrbitControls ─────────────────────────────────────────────
    if (window.THREE.OrbitControls) {
      this.controls = new window.THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping  = true;
      this.controls.dampingFactor  = 0.06;
      this.controls.maxDistance    = 1200;
      this.controls.minDistance    = 40;
      this.controls.target.set(0, 0, 0);
    }

    this._buildLighting();
    this._buildGrid();
    this._buildEyeballs();
    this._buildOrbitRings();
    this._buildFOVCones();
    this._buildFaceCloud();
    this._buildScreenPlane();

    this._animate();

    this._ro = new ResizeObserver(() => this.onResize());
    this._ro.observe(this.container);
  }

  // ── Iluminação ────────────────────────────────────────────────────
  _buildLighting() {
    this.scene.add(new window.THREE.AmbientLight(0x1a2540, 2.2));

    const key = new window.THREE.DirectionalLight(0x00f5ff, 3.0);
    key.position.set(80, 120, 180);
    this.scene.add(key);

    const fill = new window.THREE.DirectionalLight(0x9333ea, 2.0);
    fill.position.set(-80, -80, -80);
    this.scene.add(fill);

    const rim = new window.THREE.DirectionalLight(0xff2d78, 1.2);
    rim.position.set(0, -50, -150);
    this.scene.add(rim);

    const top = new window.THREE.PointLight(0xffffff, 0.8, 800);
    top.position.set(0, 180, 0);
    this.scene.add(top);
  }

  // ── Grid ──────────────────────────────────────────────────────────
  _buildGrid() {
    const g = new window.THREE.GridHelper(900, 70, 0x00f5ff, 0x0a1422);
    g.position.y = -110;
    this.scene.add(g);

    // Segundo grid mais grosso como referência
    const g2 = new window.THREE.GridHelper(900, 14, 0x00f5ff, 0x091220);
    g2.position.y = -110.5;
    this.scene.add(g2);
  }

  // ── Globo Ocular ─────────────────────────────────────────────────
  _makeEyeball() {
    const grp = new window.THREE.Group();
    const R = 22;

    // Esclera
    const sclGeo = new window.THREE.SphereGeometry(R, 64, 64);
    const sclMat = new window.THREE.MeshPhongMaterial({
      color:     0xeef4ff,
      shininess: 140,
      specular:  0x99bbdd
    });
    grp.add(new window.THREE.Mesh(sclGeo, sclMat));

    // Íris (disco colorido)
    const irisGeo = new window.THREE.CircleGeometry(8.8, 64);
    const irisMat = new window.THREE.MeshPhongMaterial({
      color:     0x1a8fff,
      shininess: 60,
      specular:  0x448899,
      side: window.THREE.DoubleSide
    });
    const iris = new window.THREE.Mesh(irisGeo, irisMat);
    iris.position.z = R - 0.4;
    grp.add(iris);

    // Anel límbico escuro
    const limGeo = new window.THREE.RingGeometry(8.2, 9.0, 64);
    const limMat = new window.THREE.MeshBasicMaterial({ color: 0x003366, side: window.THREE.DoubleSide });
    const lim = new window.THREE.Mesh(limGeo, limMat);
    lim.position.z = R - 0.35;
    grp.add(lim);

    // Pupila
    const pupGeo = new window.THREE.CircleGeometry(4.2, 48);
    const pupMat = new window.THREE.MeshBasicMaterial({ color: 0x020408, side: window.THREE.DoubleSide });
    const pup = new window.THREE.Mesh(pupGeo, pupMat);
    pup.position.z = R - 0.2;
    grp.add(pup);

    // Highlight de córnea
    const hlGeo = new window.THREE.CircleGeometry(1.1, 16);
    const hlMat = new window.THREE.MeshBasicMaterial({ color: 0xffffff, side: window.THREE.DoubleSide });
    const hl = new window.THREE.Mesh(hlGeo, hlMat);
    hl.position.set(2.5, 2.5, R + 0.12);
    grp.add(hl);

    // Córnea convexa transparente
    const corGeo = new window.THREE.SphereGeometry(10, 32, 32, 0, Math.PI * 2, 0, Math.PI * 0.22);
    const corMat = new window.THREE.MeshPhongMaterial({
      color: 0xffffff, transparent: true, opacity: 0.06,
      shininess: 300, specular: 0xffffff
    });
    const cor = new window.THREE.Mesh(corGeo, corMat);
    cor.position.z = R - 1.5;
    grp.add(cor);

    return grp;
  }

  _buildEyeballs() {
    const ipd = 40;

    this.leftEyeball = this._makeEyeball();
    this.leftEyeball.position.set(ipd, 0, 0);
    this.scene.add(this.leftEyeball);

    this.rightEyeball = this._makeEyeball();
    this.rightEyeball.position.set(-ipd, 0, 0);
    this.scene.add(this.rightEyeball);
  }

  // ── Anéis Orbitais ────────────────────────────────────────────────
  _buildOrbitRings() {
    const ringGeo = new window.THREE.TorusGeometry(50, 0.7, 8, 80);
    const ringMat = new window.THREE.MeshBasicMaterial({
      color: 0x00f5ff, transparent: true, opacity: 0.4
    });

    this.leftRing = new window.THREE.Mesh(ringGeo, ringMat);
    this.leftRing.position.set(40, 0, 0);
    this.leftRing.rotation.y = Math.PI / 3;
    this.scene.add(this.leftRing);

    this.rightRing = new window.THREE.Mesh(ringGeo.clone(), ringMat.clone());
    this.rightRing.position.set(-40, 0, 0);
    this.rightRing.rotation.y = -Math.PI / 3;
    this.scene.add(this.rightRing);
  }

  // ── Cones de Campo de Visão ───────────────────────────────────────
  _buildFOVCones() {
    this.fovGroup = new window.THREE.Group();
    const L = 270;

    const makeCone = (radius, color, opacity, wire = false) => {
      const geo = new window.THREE.ConeGeometry(radius, L, 48, 1, true);
      geo.translate(0, -L / 2, 0);
      geo.rotateX(-Math.PI / 2);
      const mat = new window.THREE.MeshBasicMaterial({
        color, transparent: true, opacity,
        wireframe: wire,
        side: window.THREE.DoubleSide
      });
      return new window.THREE.Mesh(geo, mat);
    };

    // Foveal 2° — verde neon brilhante
    const fovRad = Math.tan((1.5 * Math.PI) / 180) * L;
    this.fovGroup.add(makeCone(fovRad, 0x00ff88, 0.65));

    // Parafoveal 10° — ciano
    const paraRad = Math.tan((5.5 * Math.PI) / 180) * L;
    this.fovGroup.add(makeCone(paraRad, 0x00d2ff, 0.20));

    // Periférico 160° — wireframe roxo
    const periRad = Math.tan((36 * Math.PI) / 180) * (L * 0.72);
    const periGeo = new window.THREE.ConeGeometry(periRad, L * 0.72, 24, 4, true);
    periGeo.translate(0, -(L * 0.72) / 2, 0);
    periGeo.rotateX(-Math.PI / 2);
    const periMat = new window.THREE.MeshBasicMaterial({
      color: 0x9333ea, transparent: true, opacity: 0.22, wireframe: true, side: window.THREE.DoubleSide
    });
    this.fovGroup.add(new window.THREE.Mesh(periGeo, periMat));

    // Raio laser do olhar
    const rayPts = [new window.THREE.Vector3(0, 0, 0), new window.THREE.Vector3(0, 0, L + 30)];
    const rayGeo = new window.THREE.BufferGeometry().setFromPoints(rayPts);
    const rayMat = new window.THREE.LineBasicMaterial({ color: 0x00ffcc, linewidth: 2 });
    this.fovGroup.add(new window.THREE.Line(rayGeo, rayMat));

    this.scene.add(this.fovGroup);
  }

  // ── Nuvem de Pontos Facial ────────────────────────────────────────
  _buildFaceCloud() {
    const N = 478;
    const geo = new window.THREE.BufferGeometry();
    const pos = new Float32Array(N * 3);
    const col = new Float32Array(N * 3);
    const c   = new window.THREE.Color(0x00f5ff);

    for (let i = 0; i < N; i++) {
      pos[i*3]   = (Math.random() - 0.5) * 90;
      pos[i*3+1] = (Math.random() - 0.5) * 100;
      pos[i*3+2] = (Math.random() - 0.5) * 45;
      col[i*3]   = c.r; col[i*3+1] = c.g; col[i*3+2] = c.b;
    }

    geo.setAttribute('position', new window.THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color',    new window.THREE.BufferAttribute(col, 3));

    const mat = new window.THREE.PointsMaterial({
      size: 3.0, vertexColors: true, transparent: true, opacity: 0.6
    });
    this.faceCloud = new window.THREE.Points(geo, mat);
    this.scene.add(this.faceCloud);
  }

  // ── Plano de Tela Virtual ─────────────────────────────────────────
  _buildScreenPlane() {
    const planeGeo = new window.THREE.PlaneGeometry(380, 240);
    const planeMat = new window.THREE.MeshBasicMaterial({
      color: 0x0b1a2e, transparent: true, opacity: 0.45, side: window.THREE.DoubleSide
    });
    this.screenPlane = new window.THREE.Mesh(planeGeo, planeMat);
    this.screenPlane.position.set(0, 0, 270);

    const edgeGeo = new window.THREE.EdgesGeometry(planeGeo);
    const edgeMat = new window.THREE.LineBasicMaterial({ color: 0x00f5ff });
    this.screenPlane.add(new window.THREE.LineSegments(edgeGeo, edgeMat));

    // Ponto de impacto
    const hitGeo = new window.THREE.SphereGeometry(5, 16, 16);
    const hitMat = new window.THREE.MeshBasicMaterial({ color: 0xff2d78 });
    this.hitPoint = new window.THREE.Mesh(hitGeo, hitMat);
    this.hitPoint.position.set(0, 0, 2);
    this.screenPlane.add(this.hitPoint);

    this.scene.add(this.screenPlane);
  }

  // ── Update por frame ──────────────────────────────────────────────
  update(eyeData, gazePrediction) {
    if (!eyeData?.detected) return;
    this._t += 0.012;

    // Rotação dos anéis
    if (this.leftRing)  this.leftRing.rotation.z  = this._t;
    if (this.rightRing) this.rightRing.rotation.z = -this._t * 1.2;

    // Olhar das pupilas
    const lrx = eyeData.leftRatios?.x  ?? 0.5;
    const lry = eyeData.leftRatios?.y  ?? 0.5;
    const rrx = eyeData.rightRatios?.x ?? 0.5;
    const rry = eyeData.rightRatios?.y ?? 0.5;

    const lYaw   = (lrx - 0.5) *  0.85;
    const lPitch = -(lry - 0.5) * 0.65;
    const rYaw   = (rrx - 0.5) *  0.85;
    const rPitch = -(rry - 0.5) * 0.65;

    if (this.leftEyeball) {
      this.leftEyeball.rotation.y  = lYaw;
      this.leftEyeball.rotation.x  = lPitch;
    }
    if (this.rightEyeball) {
      this.rightEyeball.rotation.y = rYaw;
      this.rightEyeball.rotation.x = rPitch;
    }

    // Cone aponta para a média dos dois olhos
    const avgYaw   = (lYaw   + rYaw)   / 2;
    const avgPitch = (lPitch + rPitch) / 2;
    if (this.fovGroup) {
      this.fovGroup.rotation.y = avgYaw;
      this.fovGroup.rotation.x = avgPitch;
    }

    // Ponto de impacto
    if (this.hitPoint && gazePrediction) {
      const px = (gazePrediction.normX - 0.5) * 380;
      const py = -(gazePrediction.normY - 0.5) * 240;
      this.hitPoint.position.set(px, py, 2);
    }

    // Nuvem de pontos facial
    if (this.faceCloud && eyeData.allLandmarks && this.showFaceCloud) {
      const attr = this.faceCloud.geometry.attributes.position;
      const lms  = eyeData.allLandmarks;
      for (let i = 0; i < lms.length && i < 478; i++) {
        const l = lms[i];
        attr.setXYZ(i, (0.5 - l.x) * 220, (0.5 - l.y) * 220, -l.z * 200);
      }
      attr.needsUpdate = true;
    }
  }

  _animate() {
    this.animId = requestAnimationFrame(() => this._animate());
    this.controls?.update();
    this.renderer?.render(this.scene, this.camera);
  }

  onResize() {
    const W = this.container.clientWidth;
    const H = this.container.clientHeight;
    if (!W || !H || !this.renderer) return;
    this.camera.aspect = W / H;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(W, H);
  }

  resetCamera() {
    this.camera.position.set(0, 25, 240);
    this.controls?.target.set(0, 0, 0);
    this.controls?.update();
  }

  toggleFOVCones(v)    { this.showCones = v;     if (this.fovGroup)  this.fovGroup.visible  = v; }
  toggleFaceCloud(v)   { this.showFaceCloud = v;  if (this.faceCloud) this.faceCloud.visible = v; }
  toggleScreenPlane(v) { if (this.screenPlane) this.screenPlane.visible = v; }

  destroy() {
    if (this.animId) cancelAnimationFrame(this.animId);
    this._ro?.disconnect();
    if (this.renderer?.domElement) this.container.removeChild(this.renderer.domElement);
  }
}
