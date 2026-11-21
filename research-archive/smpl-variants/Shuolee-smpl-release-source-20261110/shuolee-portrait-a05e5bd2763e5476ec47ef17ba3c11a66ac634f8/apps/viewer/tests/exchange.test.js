import test from 'node:test';
import assert from 'node:assert/strict';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { exportPointPly, exchangePackage, sha256 } from '../src/io/exchange-package.js';
import { normalizeCameras } from '../src/io/camera-io.js';
import { colmapCameraFiles, crc32 } from '../src/io/camera-formats.js';

const camera = { img_name: 'frame 01.png', width: 900, height: 1600,
  K: [[1100, 0, 430.25], [0, 1120, 780.75], [0, 0, 1]],
  w2c: [[0, -1, 0, 2], [1, 0, 0, -3], [0, 0, 1, 4], [0, 0, 0, 1]],
  pixelConvention: 'integer_centers', projectionModel: 'pinhole' };
function zipEntries(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), entries = {};
  let at = 0;
  while (view.getUint32(at, true) === 0x04034b50) {
    assert.equal(view.getUint16(at + 8, true), 0);
    const size = view.getUint32(at + 18, true), n = view.getUint16(at + 26, true), extra = view.getUint16(at + 28, true);
    const name = new TextDecoder().decode(bytes.subarray(at + 30, at + 30 + n)), body = at + 30 + n + extra;
    entries[name] = bytes.slice(body, body + size);
    assert.equal(crc32(entries[name]), view.getUint32(at + 14, true));
    at = body + size;
  }
  assert.equal(view.getUint32(at, true), 0x02014b50);
  assert.equal(view.getUint32(bytes.length - 22, true), 0x06054b50);
  assert.equal(view.getUint16(bytes.length - 12, true), Object.keys(entries).length);
  return entries;
}
test('point export is read by independent Three PLYLoader in original coordinates', () => {
  const points = { count: 2, positions: new Float32Array([-1.25, 0, 3, 4, 5, -6]),
    colors: new Float32Array([1, 0.5, 0, 0, 0.25, 1]) };
  const result = exportPointPly(points), geometry = new PLYLoader().parse(result.bytes.buffer);
  assert.deepEqual([...geometry.attributes.position.array], [...points.positions]);
  assert.equal(geometry.attributes.color.count, 2);
  assert.throws(() => exportPointPly({ ...points, colors: new Float32Array(3) }), /颜色数量/);
  assert.throws(() => exportPointPly({ ...points, positions: [Infinity, 0, 0, 0, 0, 0] }), /无效坐标/);
});
test('standard package has CRC-correct ZIP and no viewer-specific dependency', async () => {
  const result = exportPointPly({ count: 1, positions: new Float32Array([1, 2, 3]) });
  const cameras = normalizeCameras({ cameras: [camera] }).cameras;
  const bundle = await exchangePackage({ ...result, cameras });
  const files = zipEntries(bundle.bytes), text = name => new TextDecoder().decode(files[name]);
  assert.deepEqual(Object.keys(files).sort(), ['README.txt', 'camera_provenance.json', 'cameras.json', 'cameras.txt', 'images.txt', 'metadata.json', 'point_cloud.ply', 'points3D.txt'].sort());
  assert.deepEqual(files['point_cloud.ply'], result.bytes);
  assert.equal(JSON.parse(text('metadata.json')).model.sha256, await sha256(result.bytes));
  assert.ok(!Object.keys(files).some(name => /spark|\.html$|\.js$/i.test(name)));
  assert.match(text('cameras.txt'), /1 PINHOLE 900 1600 1100 1120 430.75 781.25/);
  const roundtrip = normalizeCameras(JSON.parse(text('cameras.json'))).cameras[0];
  assert.deepEqual(roundtrip.position, cameras[0].position);
  assert.deepEqual(roundtrip.rotation, cameras[0].rotation);
  assert.equal(roundtrip.cx, 430.75);
});
test('camera conversion retains w2c pose, independent off-center K and refuses wrong model or distortion', () => {
  const cameras = normalizeCameras({ cameras: [camera] }).cameras;
  const file = colmapCameraFiles(cameras)['images.txt'];
  const row = file.split('\n').find(x => /^1 /.test(x)).split(' ');
  const q = row.slice(1, 5).map(Number), t = row.slice(5, 8).map(Number);
  assert.deepEqual(t, [2, -3, 4]);
  assert.ok(Math.abs(q[0] - Math.SQRT1_2) < 1e-12);
  assert.ok(Math.abs(q[3] - Math.SQRT1_2) < 1e-12);
  assert.throws(() => normalizeCameras({ model: { sha256: 'abc' }, cameras: [camera] }, { expectedSha256: 'def' }), /SHA256/);
  assert.throws(() => normalizeCameras({ cameras: [{ ...camera, D: [0, 0.1] }] }), /畸变/);
  assert.throws(() => normalizeCameras({ cameras: [{ ...camera, projectionModel: 'fisheye' }] }), /针孔/);
});
