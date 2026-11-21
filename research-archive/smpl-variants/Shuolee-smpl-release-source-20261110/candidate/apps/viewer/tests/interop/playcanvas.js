import * as pc from 'playcanvas';
import { createHarness } from './common.js';

createHarness('PlayCanvas', '2.23.3', async (source, update) => {
  const root = document.querySelector('#viewport');
  root.replaceChildren();
  const canvas = document.createElement('canvas');
  root.append(canvas);
  const app = new pc.Application(canvas, {
    graphicsDeviceOptions: { antialias: false, alpha: false },
  });
  let disposed = false;
  const resize = () => { if (!disposed) app.resizeCanvas(root.clientWidth, root.clientHeight); };
  const observer = new ResizeObserver(resize);
  observer.observe(root);
  resize();
  const camera = new pc.Entity('Interop camera');
  camera.addComponent('camera', { clearColor: new pc.Color(0.025, 0.035, 0.06), fov: 45 });
  app.root.addChild(camera);
  const asset = new pc.Asset('Local standard PLY', 'gsplat', { url: source, filename: 'model.ply' }, { reorder: false });
  app.assets.add(asset);
  const cleanup = () => {
    if (disposed) return;
    disposed = true;
    observer.disconnect();
    app.destroy();
    canvas.remove();
  };
  try {
    await new Promise((resolve, reject) => {
      asset.once('load', resolve);
      asset.once('error', reject);
      app.assets.load(asset);
    });
    const data = asset.resource.gsplatData;
    if (!Number.isSafeInteger(data.numSplats) || data.numSplats <= 0) throw new Error('Third-party decoder returned no Gaussians.');
    const entity = new pc.Entity('Standard 3DGS');
    // Non-unified path uses the engine's own GSplatInstance with its SH material.
    entity.addComponent('gsplat', { asset, unified: false });
    app.root.addChild(entity);
    const bounds = asset.resource.aabb;
    const center = bounds.center.clone();
    const initialRadius = Math.max(bounds.halfExtents.length() * 3.5, 0.5);
    let radius = initialRadius;
    let azimuth = 0;
    let elevation = 0;
    const placeCamera = () => {
      camera.setPosition(center.x + radius * Math.sin(azimuth) * Math.cos(elevation),
        center.y + radius * Math.sin(elevation), center.z + radius * Math.cos(azimuth) * Math.cos(elevation));
      camera.lookAt(center);
      camera.camera.nearClip = Math.max(radius / 1000, 0.0001);
      camera.camera.farClip = Math.max(radius * 100, initialRadius * 100);
    };
    placeCamera();
    canvas.addEventListener('pointerdown', (event) => canvas.setPointerCapture(event.pointerId));
    canvas.addEventListener('pointermove', (event) => {
      if (!canvas.hasPointerCapture(event.pointerId)) return;
      azimuth -= event.movementX * 0.007;
      elevation = Math.max(-1.5, Math.min(1.5, elevation + event.movementY * 0.007));
      placeCamera();
    });
    canvas.addEventListener('pointerup', (event) => {
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    });
    canvas.addEventListener('wheel', (event) => {
      event.preventDefault();
      radius = Math.max(initialRadius / 100, Math.min(initialRadius * 100, radius * Math.exp(event.deltaY * 0.001)));
      placeCamera();
    }, { passive: false });
    update({ state: 'loaded', loadedCount: data.numSplats, loadedShDegree: data.shBands,
      rendererShLimit: 3, center: [center.x, center.y, center.z], radius,
      gaussianProperties: ['scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity']
        .filter((name) => !!data.getProp(name)),
    });
    let frames = 0;
    app.on('postrender', () => {
      // This scene contains only one model and one camera: no mesh, grid or UI draw calls.
      if (!disposed && app.stats.drawCalls.total > 0 && frames < 3) {
        frames++;
        update({ state: 'rendered', renderFrames: frames });
      }
    });
    app.start();
    return cleanup;
  } catch (error) { cleanup(); throw error; }
});
