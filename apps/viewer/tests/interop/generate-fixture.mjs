/** Deterministic Graphdeco/INRIA 3DGS PLY, generated without the Shuolee reader. */
import { writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

export function createFixture() {
  const fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
    ...Array.from({ length: 45 }, (_, i) => `f_rest_${i}`),
    'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'];
  const colors = [[0.95, 0.12, 0.08], [0.10, 0.90, 0.18], [0.12, 0.25, 0.95]];
  const c0 = 0.28209479177387814;
  const rows = [];
  for (let y = 0; y < 3; y++) {
    for (let x = 0; x < 3; x++) {
      const index = y * 3 + x;
      const angle = (index + 1) * Math.PI / 12;
      const probability = 0.50 + index * 0.05;
      rows.push([
        (x - 1) * 0.6, (y - 1) * 0.6, (index % 2) * 0.08, 0, 0, 0,
        ...colors[x].map((channel) => (channel - 0.5) / c0),
        ...Array.from({ length: 45 }, (_, coefficient) =>
          0.03 * Math.sin((index + 1) * (coefficient + 1))),
        Math.log(probability / (1 - probability)),
        Math.log(0.15), Math.log(0.055), Math.log(0.025),
        Math.cos(angle / 2), 0, 0, Math.sin(angle / 2),
      ]);
    }
  }
  const header = Buffer.from(['ply', 'format binary_little_endian 1.0',
    'comment Synthetic interoperability control; no personal data',
    'comment Graphdeco log scales, opacity logits, wxyz quaternions, channel-major SH3',
    `element vertex ${rows.length}`, ...fields.map((field) => `property float ${field}`),
    'end_header', ''].join('\n'), 'ascii');
  const data = Buffer.alloc(rows.length * fields.length * 4);
  rows.forEach((row, i) => row.forEach((value, j) => data.writeFloatLE(value, (i * fields.length + j) * 4)));
  const ply = Buffer.concat([header, data]);
  const manifest = {
    schema: 'shuolee.interop-fixture.v1', format: 'binary_little_endian',
    count: rows.length, propertyCount: fields.length, shDegree: 3,
    sha256: createHash('sha256').update(ply).digest('hex'),
    semantics: { scale: 'natural logarithm', opacity: 'logit', quaternion: 'wxyz',
      sh: 'channel-major f_rest_0..44; Graphdeco real SH' },
    expected: { rows: 3, columns: 3, columnColors: ['red', 'green', 'blue'],
      anisotropic: true, nonidentityRotations: true, nonzeroSh3: true,
      opacityRange: [0.5, 0.9], bounds: { min: [-0.6, -0.6, 0], max: [0.6, 0.6, 0.08] } },
  };
  return { ply, manifest, fields, rows };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const { ply, manifest } = createFixture();
  writeFileSync(new URL('./anisotropic-sh3.ply', import.meta.url), ply);
  writeFileSync(new URL('./anisotropic-sh3.json', import.meta.url), `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(JSON.stringify(manifest));
}
