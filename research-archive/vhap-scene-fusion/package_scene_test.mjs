import {readFile, writeFile} from 'node:fs/promises';
import {homedir} from 'node:os';
import {basename, extname, join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [exportDirectory, outputZip, viewerOverride, ...extra] = process.argv.slice(2);
if (!exportDirectory || !outputZip || extname(outputZip).toLowerCase() !== '.zip' || extra.length) {
    throw new Error('Usage: node package_scene_test.mjs JOINT_EXPORT_DIR OUTPUT.zip [VIEWER_DIR]');
}

const viewer = resolve(viewerOverride ?? join(homedir(), 'Desktop', 'GS-SLAM-Lab', 'outputs',
    'mini-splatting2-mip-depth-001', 'viewer'));
const viewerModule = name => import(pathToFileURL(join(viewer, name)).href);
const [loader, cameraIO, rendering, plyIO] = await Promise.all([
    viewerModule('scene-package.js'),
    viewerModule('camera-io.js'),
    viewerModule('render-contract.js'),
    viewerModule('point-cloud-loader.js'),
]);
const result = resolve(exportDirectory);
const [ply, cameraText, metricsText] = await Promise.all([
    readFile(join(result, 'point_cloud.ply')),
    readFile(join(result, 'cameras.json'), 'utf8'),
    readFile(join(result, 'metrics.json'), 'utf8'),
]);
const original = JSON.parse(cameraText);
const metrics = JSON.parse(metricsText);
const model = ply.buffer.slice(ply.byteOffset, ply.byteOffset + ply.byteLength);
const plyInfo = plyIO.inspectPly(model);
if (plyInfo.kind !== 'gaussian' || !Array.isArray(metrics.scene_and_head_point_counts)
        || metrics.scene_and_head_point_counts.length !== 2
        || metrics.scene_and_head_point_counts.some(n => !Number.isSafeInteger(n) || n < 1)
        || metrics.scene_and_head_point_counts.reduce((sum, count) => sum + count, 0) !== plyInfo.count) {
    throw new Error('Joint metrics and Gaussian PLY point counts disagree');
}
if (!Number.isSafeInteger(metrics.steps) || metrics.steps < 1 || !Number.isFinite(metrics.readback_rgb_max)
        || metrics.readback_rgb_max < 0 || metrics.readback_rgb_max > 1e-3) {
    throw new Error('Expected the completed joint metrics and its native render readback');
}
if (!Array.isArray(original) || original.length === 0 || original.some(camera =>
    !camera || !Number.isFinite(camera.cx) || !Number.isFinite(camera.cy)
        || camera.pixelConvention && camera.pixelConvention !== 'pixel_corners')) {
    throw new Error('Joint cameras must have explicit principal points in pixel-corner coordinates');
}
const {cameras} = cameraIO.normalizeCameras({pixelConvention: 'pixel_corners', cameras: original});
const reference = metrics.reference;
if (typeof reference !== 'string' || !reference.length) {
    throw new Error('Joint metrics are missing the snapshot reference frame');
}
const referenceFrameId = cameras.some(camera => camera.img_name === reference)
    ? reference : basename(reference, extname(reference));
const initialView = cameras.findIndex(camera => camera.img_name === referenceFrameId);
if (initialView < 0) {
    throw new Error('The joint snapshot reference frame is missing from the camera array');
}

const packaged = await loader.scenePackage(model, 'ply', cameras, {
    label: basename(result),
    steps: metrics.steps,
    referenceFrameId,
    initialView,
    rendererContract: {...rendering.previewRendererContract(),
        trainingRenderer: metrics.training_renderer ?? 'diff-gaussian-rasterization'},
});
const entries = await loader.unpackSceneZip(packaged.bytes);
const files = entries.map(entry => {
    const file = new File([entry.data], basename(entry.name));
    file.sceneRelativePath = entry.name;
    return file;
});
const models = await loader.primaryModelFiles(files);
if (models.length !== 1 || !ply.equals(Buffer.from(await models[0].arrayBuffer()))) {
    throw new Error('The viewer package did not retain the exact source PLY');
}
const paired = await loader.pairedCameras(models[0], files, models);
const cameraFields = ['img_name', 'width', 'height', 'fx', 'fy', 'cx', 'cy', 'position', 'rotation'];
if (!paired.verifiedBinding || paired.referenceFrameId !== referenceFrameId || paired.initialView !== initialView
        || paired.cameras.length !== original.length
        || paired.cameras.some((camera, index) => cameraFields.some(field =>
            JSON.stringify(camera[field]) !== JSON.stringify(original[index][field])))) {
    throw new Error('The viewer package did not retain the source cameras and reference view');
}

await writeFile(outputZip, packaged.bytes, {flag: 'wx'});
console.log(JSON.stringify({
    zip: resolve(outputZip),
    plySha256: packaged.hash,
    points: plyInfo.count,
    cameras: paired.cameras.length,
    referenceFrameId,
    verifiedBinding: paired.verifiedBinding,
    nativePreviewReadbackMax: metrics.readback_rgb_max,
}, null, 2));
