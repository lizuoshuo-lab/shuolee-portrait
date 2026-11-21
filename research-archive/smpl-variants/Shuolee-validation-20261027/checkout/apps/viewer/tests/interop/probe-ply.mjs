/** Read-only local probe. It never uploads a model and does not claim rendering success. */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { PlyParser } from '@mkkellogg/gaussian-splats-3d';
import { decodePlayCanvasCpu } from './playcanvas-cpu-reader.mjs';

const path = process.argv[2];
if (!path) throw new Error('Usage: node tests/interop/probe-ply.mjs <local-standard-3dgs.ply>');
const bytes = readFileSync(resolve(path));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const mkk = PlyParser.parseToUncompressedSplatBuffer(buffer, 2);
const pc = await decodePlayCanvasCpu(bytes);
const result = {
  file: resolve(path), sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length,
  gaussianSplats3D: { version: '0.4.7', decodedCount: mkk.getSplatCount(),
    decodedShDegree: mkk.getMinSphericalHarmonicsDegree(), rendererShLimit: 2 },
  playcanvas: { version: '2.23.3', decodedCount: pc.numSplats, decodedShDegree: pc.shBands },
  countsAgree: mkk.getSplatCount() === pc.numSplats,
  cpuDecodeVerified: true, guiVerified: false, qualityAccepted: false,
};
console.log(JSON.stringify(result, null, 2));
if (!result.countsAgree || pc.numSplats <= 0) process.exitCode = 1;
