import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectGaussianPly, exportStandardGaussianPly } from '../src/io/gaussian-ply.js';

const encoder = new TextEncoder();
function fieldNames(degree = 0, withNormals = true) {
  return ['x', 'y', 'z', ...(withNormals ? ['nx', 'ny', 'nz'] : []),
    'f_dc_0', 'f_dc_1', 'f_dc_2',
    ...Array.from({ length: 3 * ((degree + 1) ** 2 - 1) }, (_, i) => `f_rest_${i}`),
    'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
}
function values() {
  // Non-axis-aligned, anisotropic and translucent. Raw quat need not be unit.
  return { x: -1.25, y: 2.5, z: 3.75, nx: 0, ny: 0, nz: 0,
    f_dc_0: -0.25, f_dc_1: 0.5, f_dc_2: 1.25, opacity: -2,
    scale_0: -4, scale_1: -2, scale_2: 0.5,
    rot_0: 2, rot_1: 0.25, rot_2: -0.5, rot_3: 0.75 };
}
function fixture({ degree = 0, names = fieldNames(degree), format = 'binary_little_endian',
  overrides = {}, count = 2, type = 'float' } = {}) {
  const header = encoder.encode(`ply\nformat ${format} 1.0\ncomment fixture\nelement vertex ${count}\n`
    + names.map(name => `property ${type} ${name}\n`).join('') + 'end_header\n');
  const rows = Array.from({ length: count }, (_, i) => names.map(name => overrides[name] ?? values()[name]
    ?? (name.startsWith('f_rest_') ? ((Number(name.slice(7)) + 1) * (i ? -1 : 1)) / 64 : 0)));
  if (format === 'ascii') {
    const body = encoder.encode(rows.map(row => row.join(' ')).join('\n') + '\n');
    const out = new Uint8Array(header.length + body.length); out.set(header); out.set(body, header.length); return out;
  }
  const width = type === 'double' ? 8 : 4;
  const out = new Uint8Array(header.length + count * names.length * width); out.set(header);
  const view = new DataView(out.buffer);
  rows.flat().forEach((value, i) => view[type === 'double' ? 'setFloat64' : 'setFloat32'](header.length + i * width, value, format === 'binary_little_endian'));
  return out;
}
function attributes(bytes) {
  const info = inspectGaussianPly(bytes), view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  assert.equal(info.format, 'binary_little_endian');
  return Array.from({ length: info.count }, (_, i) => Object.fromEntries(info.properties.map((name, j) =>
    [name, view.getFloat32(info.headerBytes + (i * info.properties.length + j) * 4, true)])));
}

for (const degree of [0, 1, 2, 3]) {
  test(`SH${degree}: standard export preserves all bytes and raw parameters`, () => {
    const original = fixture({ degree }), frozen = original.slice();
    const info = inspectGaussianPly(original), out = exportStandardGaussianPly(original);
    assert.equal(info.kind, 'gaussian'); assert.equal(info.shDegree, degree); assert.equal(info.count, 2);
    assert.equal(info.bodyValidated, true); assert.equal(out.metadata.mode, 'byte-preserved');
    assert.deepEqual(out.bytes, frozen); assert.deepEqual(original, frozen);
    assert.notEqual(out.bytes.buffer, original.buffer);
    const row = attributes(out.bytes)[0];
    for (const [name, value] of Object.entries(values())) assert.equal(row[name], value);
    if (degree) { assert.equal(row.f_rest_0, 1 / 64); assert.equal(row[`f_rest_${3 * ((degree + 1) ** 2 - 1) - 1}`], 3 * ((degree + 1) ** 2 - 1) / 64); }
  });
}

test('Float32 ASCII / big endian / reordered inputs canonicalize without value transforms', () => {
  const expected = attributes(fixture({ degree: 3 }));
  for (const format of ['ascii', 'binary_big_endian', 'binary_little_endian']) {
    const source = fixture({ degree: 3, format, names: fieldNames(3).reverse() });
    const out = exportStandardGaussianPly(source);
    assert.equal(out.metadata.mode, 'canonical-format-conversion');
    assert.deepEqual(out.metadata.numericTransforms, []);
    assert.deepEqual(attributes(out.bytes), expected);
    assert.equal(inspectGaussianPly(out.bytes).canonical, true);
  }
});

test('absent optional normals are explicitly filled with zero', () => {
  const source = fixture({ degree: 1, names: fieldNames(1, false) });
  const out = exportStandardGaussianPly(source);
  assert.deepEqual(out.metadata.addedProperties, ['nx', 'ny', 'nz']);
  const row = attributes(out.bytes)[0];
  assert.equal(row.nx, 0); assert.equal(row.ny, 0); assert.equal(row.nz, 0);
  assert.equal(row.rot_0, 2); assert.equal(row.opacity, -2); assert.equal(row.scale_0, -4);
});

test('ordinary points are classified separately and cannot masquerade as Gaussian export', () => {
  const points = fixture({ names: ['x', 'y', 'z', 'red', 'green', 'blue'] });
  assert.equal(inspectGaussianPly(points).kind, 'points');
  assert.equal(inspectGaussianPly(points).bodyValidated, false);
  assert.throws(() => exportStandardGaussianPly(points), { code: 'NOT_GAUSSIAN_PLY' });
});

test('any missing mandatory Gaussian field fails rather than downgrading to points', () => {
  for (const omitted of fieldNames().filter(name => !['nx', 'ny', 'nz'].includes(name))) {
    assert.throws(() => inspectGaussianPly(fixture({ names: fieldNames().filter(name => name !== omitted) })),
      { code: 'INCOMPLETE_GAUSSIAN_SCHEMA' }, omitted);
  }
  assert.throws(() => inspectGaussianPly(fixture({ names: ['x', 'y', 'z', 'opacity'] })), /Incomplete Gaussian/);
});

test('SH counts, gaps, malformed indices and extra parameter fields are rejected', () => {
  const bad = [fieldNames(1).filter(name => name !== 'f_rest_0'),
    fieldNames(1).map(name => name === 'f_rest_0' ? 'f_rest_9' : name),
    fieldNames(1).map(name => name === 'f_rest_0' ? 'f_rest_00' : name),
    [...fieldNames(), 'f_dc_3'], [...fieldNames(), 'rot_4'], [...fieldNames(), 'scale_3']];
  for (const names of bad) assert.throws(() => inspectGaussianPly(fixture({ names })), /Gaussian/);
});

test('non-finite values fail in centers, all SH orders, opacity, scale and rotation', () => {
  for (const field of ['x', 'y', 'z', 'f_dc_0', 'f_rest_0', 'f_rest_44', 'opacity', 'scale_2', 'rot_3']) {
    for (const bad of [NaN, Infinity, -Infinity]) {
      assert.throws(() => inspectGaussianPly(fixture({ degree: 3, overrides: { [field]: bad } })), /must be finite/, field);
    }
  }
  assert.throws(() => inspectGaussianPly(fixture({ format: 'ascii', overrides: { opacity: NaN } })), /must be finite/);
});

test('zero quaternion fails; arbitrary nonzero raw quaternion is preserved', () => {
  assert.throws(() => inspectGaussianPly(fixture({ overrides: { rot_0: 0, rot_1: 0, rot_2: 0, rot_3: 0 } })), /quaternion must be nonzero/);
  const raw = { rot_0: 0, rot_1: -5, rot_2: 2, rot_3: -3 };
  const out = exportStandardGaussianPly(fixture({ overrides: raw }));
  for (const [key, value] of Object.entries(raw)) assert.equal(attributes(out.bytes)[0][key], value);
});

test('truncation, trailing data, duplicates and unsupported precision fail explicitly', () => {
  const original = fixture();
  assert.throws(() => inspectGaussianPly(original.subarray(0, original.length - 1)), /body length/);
  const extra = new Uint8Array(original.length + 1); extra.set(original);
  assert.throws(() => inspectGaussianPly(extra), /body length/);
  assert.throws(() => inspectGaussianPly(fixture({ names: [...fieldNames(), 'x'] })), /duplicate property/);
  assert.throws(() => inspectGaussianPly(fixture({ type: 'double' })), { code: 'UNSUPPORTED_GAUSSIAN_ENCODING' });
  assert.throws(() => inspectGaussianPly(fixture({ names: fieldNames().filter(name => name !== 'nz') })), /normals/);
});

test('unknown attributes cannot be silently discarded during canonical export', () => {
  const source = fixture({ names: [...fieldNames(), 'custom_confidence'] });
  assert.equal(inspectGaussianPly(source).kind, 'gaussian');
  assert.throws(() => exportStandardGaussianPly(source), { code: 'UNSUPPORTED_GAUSSIAN_FIELDS' });
});

test('typed array slices and DataView respect their byte offsets', () => {
  const original = fixture({ degree: 2 });
  const wrapped = new Uint8Array(original.length + 20); wrapped.set(original, 11);
  const view = new DataView(wrapped.buffer, 11, original.length);
  assert.equal(inspectGaussianPly(view).shDegree, 2);
  assert.deepEqual(exportStandardGaussianPly(view).bytes, original);
});
