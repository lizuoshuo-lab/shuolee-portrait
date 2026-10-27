import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Vector3, Vector4, Quaternion } from 'three';
import { PlyParser as MkkPlyParser } from '@mkkellogg/gaussian-splats-3d';
import { Vec3, Quat, Color } from 'playcanvas';
import { createFixture } from './generate-fixture.mjs';
import { decodePlayCanvasCpu } from './playcanvas-cpu-reader.mjs';

const binary = readFileSync(new URL('./anisotropic-sh3.ply', import.meta.url));
const manifest = JSON.parse(readFileSync(new URL('./anisotropic-sh3.json', import.meta.url), 'utf8'));
const close = (actual, expected, tolerance = 1e-6) => assert.ok(Math.abs(actual - expected) < tolerance,
  `${actual} should be within ${tolerance} of ${expected}`);

test('committed synthetic binary is deterministic and contains all 62 standard fields', () => {
  assert.equal(createHash('sha256').update(binary).digest('hex'), manifest.sha256);
  assert.ok(binary.equals(createFixture().ply));
  assert.equal(manifest.count, 9);
  assert.equal(manifest.propertyCount, 62);
});

test('GaussianSplats3D independently decodes positions, anisotropy, wxyz rotations and opacity', () => {
  const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  const data = MkkPlyParser.parseToUncompressedSplatBuffer(arrayBuffer, 2);
  assert.equal(data.getSplatCount(), 9);
  assert.equal(data.getMinSphericalHarmonicsDegree(), 2);
  const position = new Vector3();
  const scale = new Vector3();
  const rotation = new Quaternion();
  const color = new Vector4();
  data.getSplatCenter(0, position);
  data.getSplatScaleAndRotation(0, scale, rotation);
  data.getSplatColor(0, color);
  close(position.x, -0.6); close(position.y, -0.6); close(position.z, 0);
  close(scale.x, 0.15); close(scale.y, 0.055); close(scale.z, 0.025);
  close(rotation.w, Math.cos(Math.PI / 24)); close(rotation.z, Math.sin(Math.PI / 24));
  assert.equal(color.w, 127); // upstream quantizes sigmoid(0)*255 with floor
  assert.ok(color.x > 235 && color.y < 40 && color.z < 30);
});

test('PlayCanvas independently decodes SH3 and activation semantics before GPU allocation', async () => {
  // The upstream PLY handler emits load:data after decoding and before allocating GPU textures.
  // Stop at that pinned-version event boundary: this CPU test makes no rendering claim.
  const decoded = await decodePlayCanvasCpu(binary);
  assert.equal(decoded.numSplats, 9);
  assert.equal(decoded.shBands, 3);
  const position = new Vec3();
  const scale = new Vec3();
  const rotation = new Quat();
  const color = new Color();
  decoded.createIter(position, rotation, scale, color).read(0);
  close(position.x, -0.6); close(position.y, -0.6); close(position.z, 0);
  close(scale.x, 0.15); close(scale.y, 0.055); close(scale.z, 0.025);
  close(rotation.w, Math.cos(Math.PI / 24)); close(rotation.z, Math.sin(Math.PI / 24));
  close(color.a, 0.5); close(color.r, 0.95);
  for (let coefficient = 0; coefficient < 45; coefficient++) {
    close(decoded.getProp(`f_rest_${coefficient}`)[0], 0.03 * Math.sin(coefficient + 1));
  }
});
