import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh, SplatFileType } from '@sparkjsdev/spark';

export class ViewerRenderer {
  constructor(canvas, viewport) {
    this.viewport = viewport;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: false, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setClearColor('#101817');
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(55, 1, 0.001, 1000);
    this.camera.position.set(3, 2, 4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.addEventListener('start', () => { this.captured = null; this.resize(); });
    this.spark = new SparkRenderer({ renderer: this.renderer, enableLod: false, accumExtSplats: true,
      preBlurAmount: 0, blurAmount: 0.3, minAlpha: 1 / 255 });
    this.scene.add(this.spark);
    this.pointSize = 2;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(viewport);
    this.resize();
    this.renderer.setAnimationLoop(() => { this.controls.update(); this.renderer.render(this.scene, this.camera); });
  }
  resize() {
    const width = Math.max(1, this.viewport.clientWidth), height = Math.max(1, this.viewport.clientHeight);
    this.renderer.setSize(width, height, false);
    let w = width, h = height;
    if (this.captured) {
      const c = this.captured, ratio = c.width / c.height;
      if (width / height > ratio) w = height * ratio; else h = width / ratio;
      this.camera.aspect = ratio * c.fy / c.fx;
      this.camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(c.height / (2 * c.fy)));
      this.camera.updateProjectionMatrix();
      this.camera.projectionMatrix.elements[8] = 1 - 2 * c.cx / c.width;
      this.camera.projectionMatrix.elements[9] = 2 * c.cy / c.height - 1;
      this.camera.projectionMatrixInverse.copy(this.camera.projectionMatrix).invert();
    } else {
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
    }
    this.renderer.setViewport((width - w) / 2, (height - h) / 2, w, h);
  }
  clear() {
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.dispose(); this.mesh = null; }
    if (this.points) { this.scene.remove(this.points); this.points.geometry.dispose(); this.points.material.dispose(); this.points = null; }
    this.bounds = null;
    this.captured = null;
  }
  createPoints(positions, colors) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    if (colors) {
      const linear = new Float32Array(colors.length), c = new THREE.Color();
      for (let i = 0; i < colors.length; i += 3) {
        c.setRGB(colors[i], colors[i + 1], colors[i + 2], THREE.SRGBColorSpace);
        linear.set([c.r, c.g, c.b], i);
      }
      geometry.setAttribute('color', new THREE.BufferAttribute(linear, 3));
    }
    geometry.computeBoundingBox();
    const material = new THREE.PointsMaterial({ size: this.pointSize, sizeAttenuation: false,
      vertexColors: !!colors, color: colors ? 0xffffff : 0xb8e8d0, toneMapped: false });
    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    this.scene.add(points);
    return points;
  }
  async loadGaussian(bytes, name) {
    this.clear();
    this.mesh = new SplatMesh({ fileBytes: bytes, fileName: name, fileType: SplatFileType.PLY,
      extSplats: true, lod: false, enableLod: false, nonLod: true });
    this.scene.add(this.mesh);
    await this.mesh.initialized;
    this.bounds = this.mesh.getBoundingBox(true);
    if (![...this.bounds.min, ...this.bounds.max].every(Number.isFinite)) throw Error('模型范围包含无效数值');
    this.fit();
    return this.mesh.splats.getNumSplats();
  }
  loadPoints(data) {
    this.clear();
    this.points = this.createPoints(data.positions, data.colors);
    this.bounds = this.points.geometry.boundingBox.clone();
    this.fit();
  }
  setMode(mode) {
    if (this.mesh && mode === 'points' && !this.points) {
      const count = this.mesh.splats.getNumSplats(), positions = new Float32Array(count * 3), colors = new Float32Array(count * 3);
      this.mesh.forEachSplat((i, p, _s, _q, _a, c) => { positions.set([p.x, p.y, p.z], i * 3); colors.set([c.r, c.g, c.b], i * 3); });
      this.points = this.createPoints(positions, colors);
    }
    if (this.mesh) this.mesh.visible = mode === 'gaussian';
    if (this.points) this.points.visible = mode === 'points' || !this.mesh;
  }
  fit() {
    if (!this.bounds) return;
    const center = this.bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(this.bounds.getSize(new THREE.Vector3()).length() / 2, 0.01);
    this.radius = radius;
    this.captured = null;
    this.camera.up.set(0, 1, 0);
    this.camera.fov = 55;
    this.camera.position.copy(center).add(new THREE.Vector3(0.35, 0.15, 1).normalize().multiplyScalar(radius * 2.8));
    this.camera.near = Math.max(radius / 10000, 1e-6);
    this.camera.far = Math.max(radius * 100, 100);
    this.controls.target.copy(center);
    this.camera.lookAt(center);
    this.controls.update();
    this.resize();
  }
  chooseCamera(c) {
    const r = c.rotation, p = c.position;
    const m = new THREE.Matrix4().set(r[0][0], -r[0][1], -r[0][2], p[0],
      r[1][0], -r[1][1], -r[1][2], p[1], r[2][0], -r[2][1], -r[2][2], p[2], 0, 0, 0, 1);
    this.camera.position.fromArray(p);
    this.camera.quaternion.setFromRotationMatrix(m);
    const direction = this.camera.getWorldDirection(new THREE.Vector3());
    this.controls.target.copy(this.camera.position).addScaledVector(direction, this.radius || 1);
    this.camera.up.setFromMatrixColumn(m, 1);
    this.controls.update();
    this.captured = c;
    this.resize();
  }
  setBackground(color) { this.renderer.setClearColor(color); }
  setPointSize(value) { this.pointSize = value; if (this.points) this.points.material.size = value; }
  screenshot() { this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL('image/png'); }
}
