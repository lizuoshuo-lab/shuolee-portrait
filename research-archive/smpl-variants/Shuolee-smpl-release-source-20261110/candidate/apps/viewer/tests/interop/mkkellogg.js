import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { Vector3, WebGLRenderer } from 'three';
import { createHarness } from './common.js';

createHarness('GaussianSplats3D', '0.4.7', async (source, update) => {
  const root = document.querySelector('#viewport');
  root.replaceChildren();
  // 0.4.7 removes rootElement from document.body when it owns the renderer.
  // Supply our own renderer so its disposal cannot delete the harness host.
  const renderer = new WebGLRenderer({ antialias: false, alpha: false, precision: 'highp' });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0x080b12, 1);
  root.append(renderer.domElement);
  let viewer;
  let disposed = false;
  const resize = () => {
    if (disposed) return;
    renderer.setSize(root.clientWidth, root.clientHeight);
    viewer?.forceRenderNextFrame();
  };
  const observer = new ResizeObserver(resize);
  const cleanup = async () => {
    if (disposed) return;
    disposed = true;
    observer.disconnect();
    viewer?.stop();
    try { await viewer?.dispose(); }
    finally {
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
    }
  };
  try {
    resize();
    viewer = new GaussianSplats3D.Viewer({
      renderer, rootElement: root, cameraUp: [0, 1, 0], initialCameraPosition: [0, 0, 3.2],
      initialCameraLookAt: [0, 0, 0], sphericalHarmonicsDegree: 2,
      sharedMemoryForWorkers: false, gpuAcceleratedSort: false,
      inMemoryCompressionLevel: 0, halfPrecisionCovariancesOnGPU: false,
      sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
    });
    observer.observe(root);
    await viewer.addSplatScene(source, {
      format: GaussianSplats3D.SceneFormat.Ply,
      splatAlphaRemovalThreshold: 0, showLoadingUI: false, progressiveLoad: false,
    });
    const count = viewer.getSplatMesh().getSplatCount();
    if (!Number.isSafeInteger(count) || count <= 0) throw new Error('Third-party decoder returned no Gaussians.');
    const scene = viewer.getSplatScene(0);
    const bounds = viewer.getSplatMesh().computeBoundingBox(true);
    const center = bounds.getCenter(new Vector3());
    const radius = Math.max(bounds.getSize(new Vector3()).length() * 1.75, 0.5);
    viewer.camera.position.copy(center).add(new Vector3(0, 0, radius));
    viewer.camera.near = Math.max(radius / 1000, 0.0001);
    viewer.camera.far = radius * 100;
    viewer.camera.lookAt(center);
    viewer.camera.updateProjectionMatrix();
    viewer.controls.target.copy(center);
    viewer.controls.update();
    update({ state: 'loaded', loadedCount: count,
      loadedShDegree: scene.splatBuffer.getMinSphericalHarmonicsDegree(),
      rendererShLimit: 2, center: center.toArray(), radius,
      cameraPosition: viewer.camera.position.toArray(), cameraLookAt: center.toArray() });
    let frames = 0;
    const originalRender = viewer.render.bind(viewer);
    viewer.render = () => {
      originalRender();
      if (!disposed && viewer.splatRenderReady && renderer.info.render.calls > 0 && frames < 3) {
        frames++;
        update({ state: 'rendered', renderFrames: frames });
      }
    };
    viewer.start();
    return cleanup;
  } catch (error) {
    await cleanup();
    throw error;
  }
});
