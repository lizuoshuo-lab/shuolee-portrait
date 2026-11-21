import './styles.css';
import { ViewerRenderer } from './render/viewer-renderer.js';
import { inspectGaussianPly, exportStandardGaussianPly } from './io/gaussian-ply.js';
import { parsePointCloud } from './io/point-cloud-loader.js';
import { normalizeCameras } from './io/camera-io.js';
import { assertPreviewCompatible } from './io/render-contract.js';
import { exportPointPly, exchangePackage, sha256 } from './io/exchange-package.js';

const $ = id => document.getElementById(id);
const MAX_BYTES = 512 * 1024 * 1024;
let current = null, cameras = [], busy = false, view;
window.viewerState = { status: 'initializing', version: __APP_VERSION__, commit: __GIT_COMMIT__ };
$('version').textContent = `v${__APP_VERSION__} · ${__GIT_COMMIT__}`;
function status(message, error = false) { $('status').textContent = message; $('status').classList.toggle('error', error); }
function setBusy(value, message = '正在读取模型…') {
  busy = value; $('busy').hidden = !value; $('busy').textContent = message;
  for (const id of ['open', 'open-empty']) $(id).disabled = value;
  for (const id of ['fit', 'mode', 'open-cameras', 'export-ply', 'export-bundle', 'screenshot']) $(id).disabled = value || !current;
}
function download(bytes, name, type = 'application/octet-stream') {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement('a'); a.href = url; a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}
function exported() {
  if (!current) throw Error('先打开模型');
  return current.kind === 'gaussian' ? exportStandardGaussianPly(current.buffer) : exportPointPly(current.points);
}
async function loadCameras(file) {
  if (!current) throw Error('先打开对应模型');
  if (file.size > 20 * 1024 * 1024) throw Error('相机JSON超过20MB');
  const data = JSON.parse((await file.text()).replace(/^\uFEFF/, ''));
  assertPreviewCompatible(data);
  if (data.sha256 && data.sha256 !== current.hash) throw Error('相机元数据与当前模型SHA256不符');
  const parsed = normalizeCameras(data, { expectedSha256: current.hash });
  cameras = parsed.cameras;
  $('views').replaceChildren(new Option('选择已采集视角…', ''), ...cameras.map((c, i) => new Option(c.label || c.img_name, String(i))));
  $('views').disabled = !cameras.length;
  $('camera-info').textContent = `${cameras.length}个相机 · 完整主点保留${parsed.notes.length ? ' · ' + parsed.notes.join('；') : ''}`;
  window.viewerState.cameras = cameras.length;
  status(`已加载${cameras.length}个相机`);
}
async function importFiles(list) {
  if (busy) return;
  const files = Array.from(list), models = files.filter(f => /\.(ply|pcd|xyz|pts)$/i.test(f.name));
  const json = files.filter(f => /\.json$/i.test(f.name));
  if (models.length !== 1) throw Error('每次选择一个模型；可同时选择一个配套相机JSON');
  if (json.length > 1) throw Error('有多个JSON，请只选择与模型配套的相机文件');
  const file = models[0];
  if (file.size > MAX_BYTES) throw Error('文件超过512MB，当前版本不会静默抽样');
  setBusy(true);
  current = null; cameras = [];
  $('views').replaceChildren(new Option('尚未载入相机')); $('views').disabled = true;
  $('camera-info').textContent = '';
  try {
    const buffer = await file.arrayBuffer(), hash = await sha256(buffer);
    const info = /\.ply$/i.test(file.name) ? inspectGaussianPly(buffer) : { kind: 'points' };
    if (info.kind === 'gaussian') {
      const standard = exportStandardGaussianPly(buffer);
      const count = await view.loadGaussian(standard.bytes, file.name);
      if (count !== info.count) throw Error('渲染器读取点数与标准PLY校验不符');
      current = { name: file.name, kind: 'gaussian', count, shDegree: info.shDegree, buffer, hash };
      $('mode').value = 'gaussian'; $('mode').options[0].disabled = false;
      $('model-info').textContent = `${count.toLocaleString()}个高斯 · SH${info.shDegree} · 标准3DGS`;
    } else {
      const points = parsePointCloud(buffer, file.name);
      view.loadPoints(points);
      current = { name: file.name, kind: 'points', count: points.count, points, buffer, hash };
      $('mode').value = 'points'; $('mode').options[0].disabled = true;
      $('model-info').textContent = `${points.count.toLocaleString()}个点${points.skipped ? ' · 跳过' + points.skipped + '个无效源点' : ''}`;
    }
    $('name').textContent = file.name; $('empty').hidden = true; $('view-label').textContent = '自由视角';
    window.viewerState = { ...window.viewerState, status: 'ready', kind: current.kind, count: current.count, hash, cameras: 0 };
    status('模型已载入 · 原始文件保持不变');
    if (json.length) await loadCameras(json[0]);
  } catch (error) {
    if (!current) { view.clear(); $('empty').hidden = false; }
    window.viewerState.status = 'error';
    throw error;
  } finally { setBusy(false); }
}
function run(task) { return async (...args) => { try { await task(...args); } catch (e) { status(e.message || String(e), true); console.error(e); } }; }
try {
  view = new ViewerRenderer($('canvas'), $('viewport'));
  window.viewerState.status = 'empty';
  $('open').onclick = $('open-empty').onclick = () => $('files').click();
  $('files').onchange = run(async () => { try { await importFiles($('files').files); } finally { $('files').value = ''; } });
  $('open-cameras').onclick = () => $('camera-file').click();
  $('camera-file').onchange = run(async () => { try { if ($('camera-file').files[0]) await loadCameras($('camera-file').files[0]); } finally { $('camera-file').value = ''; } });
  $('fit').onclick = () => { view.fit(); $('view-label').textContent = '自由视角'; };
  $('mode').onchange = run(() => view.setMode($('mode').value));
  $('background').onchange = () => view.setBackground($('background').value);
  $('point-size').oninput = () => view.setPointSize(Number($('point-size').value));
  $('views').onchange = () => { if ($('views').value === '') return; const c = cameras[Number($('views').value)]; view.chooseCamera(c); $('view-label').textContent = c.label || c.img_name; };
  $('export-ply').onclick = run(() => { const result = exported(); download(result.bytes, current.name.replace(/\.[^.]+$/, '') + '.ply'); status(`已导出通用PLY · ${result.metadata.mode === 'byte-preserved' ? '与原件字节一致' : '标准格式转换'}`); });
  $('export-bundle').onclick = run(async () => { const result = exported(); const bundle = await exchangePackage({ ...result, cameras }); download(bundle.bytes, current.name.replace(/\.[^.]+$/, '') + '_exchange.zip', 'application/zip'); status(`已导出标准模型${cameras.length ? '与COLMAP相机' : ''}包`); });
  $('screenshot').onclick = () => { const a = document.createElement('a'); a.href = view.screenshot(); a.download = 'view.png'; a.click(); };
  $('viewport').ondragover = e => e.preventDefault();
  $('viewport').ondrop = run(async e => { e.preventDefault(); await importFiles(e.dataTransfer.files); });
} catch (e) { $('empty').hidden = false; status(`图形初始化失败：${e.message}`, true); window.viewerState.status = 'error'; }
