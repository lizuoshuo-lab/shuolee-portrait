import { cameras3dgs, colmapCameraFiles, storedZip } from './camera-formats.js';

export async function sha256(bytes) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), x => x.toString(16).padStart(2, '0')).join('');
}

export function exportPointPly(points) {
  const { positions, colors, count } = points;
  if (!Number.isSafeInteger(count) || count < 1 || positions.length !== count * 3) throw Error('点云数量无效');
  if (colors && colors.length !== count * 3) throw Error('点云颜色数量无效');
  const header = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\ncomment Standard point cloud; original world coordinates\nelement vertex ${count}\nproperty float x\nproperty float y\nproperty float z\n${colors ? 'property uchar red\nproperty uchar green\nproperty uchar blue\n' : ''}end_header\n`);
  const stride = colors ? 15 : 12;
  const bytes = new Uint8Array(header.length + count * stride);
  bytes.set(header);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < count; i++) {
    let at = header.length + i * stride;
    for (let j = 0; j < 3; j++) {
      const v = positions[i * 3 + j];
      if (!Number.isFinite(v)) throw Error('点云包含无效坐标');
      view.setFloat32(at, v, true); at += 4;
    }
    if (colors) for (let j = 0; j < 3; j++) {
      const c = colors[i * 3 + j];
      if (!Number.isFinite(c) || c < 0 || c > 1) throw Error('点云颜色无效');
      view.setUint8(at++, Math.round(c * 255));
    }
  }
  return { bytes, metadata: { format: 'point-cloud-ply', count, coordinates: 'original-world-float32',
    color: colors ? 'sRGB-uint8' : null, skippedSourcePoints: points.skipped || 0 } };
}

export async function exchangePackage({ bytes, cameras = [], metadata = {} }) {
  const modelHash = await sha256(bytes);
  const files = { 'point_cloud.ply': bytes };
  if (cameras.length) {
    Object.assign(files, colmapCameraFiles(cameras, { model: { sha256: modelHash } }));
    files['cameras.json'] = JSON.stringify(cameras3dgs(cameras), null, 2) + '\n';
  }
  files['metadata.json'] = JSON.stringify({ format: 'standard-3dgs-or-point-cloud-ply',
    model: { file: 'point_cloud.ply', sha256: modelHash }, cameras: cameras.length,
    coordinates: 'original-world-no-transform', conversion: metadata,
    qualityAccepted: false }, null, 2) + '\n';
  files['README.txt'] = 'Open point_cloud.ply directly in a compatible Gaussian or point-cloud viewer.\n'
    + 'Gaussian PLY uses Graphdeco 3DGS log scales, logit opacity, Hamilton wxyz and RGB-major SH.\n'
    + 'No application-specific HTML or renderer is required.\n'
    + (cameras.length ? 'COLMAP cameras.txt/images.txt/points3D.txt retain full fx/fy/cx/cy and world-to-camera poses.\ncameras.json is the common 3DGS array; some readers assume a centered principal point.\nNo source images or fabricated sparse tracks are included.\n' : 'No cameras were supplied.\n')
    + 'Coordinates and Gaussian values are not reoriented or rescaled.\nFormat compatibility does not certify reconstruction quality or equivalent rendering.\n';
  return { bytes: storedZip(files), modelHash, files: Object.keys(files) };
}
