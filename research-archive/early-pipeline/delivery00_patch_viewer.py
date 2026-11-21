from pathlib import Path
import json, shutil, datetime

ROOT=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer')
WORK=Path(__file__).with_name('delivery00-build')
WORK.mkdir(exist_ok=True)
for name in ['index.html','viewer.js']:
    backup=WORK/('original-'+name)
    if not backup.exists(): shutil.copy2(ROOT/name,backup)
html=(ROOT/'index.html').read_text(encoding='utf-8-sig')
if 'id="gaussian-scale"' not in html:
    anchor='<div id="point-options" hidden>'
    assert anchor in html
    html=html.replace(anchor,'''<div id="gaussian-options" hidden><label class="micro" for="gaussian-scale">高斯球尺寸 <output id="gaussian-scale-value">1.00 ×</output></label><input id="gaussian-scale" type="range" min="0.05" max="2" step="0.05" value="1" aria-label="高斯球尺寸"><button id="gaussian-scale-reset">恢复 1×</button><p class="micro">缩小可观察高斯分布；只调整显示尺寸，不改变点的位置或原文件。</p></div>
'''+anchor)
    html=html.replace('<div class="links">','<div class="links"><a href="progress.html" target="_blank">最新实验与当天资料 ↗</a>')
    (ROOT/'index.html').write_text(html,encoding='utf-8')
js=(ROOT/'viewer.js').read_text(encoding='utf-8-sig')
if 'function updateGaussianScale' not in js:
    js=js.replace('SparkRenderer, SplatMesh, SplatFileType','SparkRenderer, SplatMesh, SplatFileType, dyno')
    js=js.replace('let current=null,', 'let gaussianScaleUniform=null;\nlet current=null,',1)
    js=js.replace("version:'shuolee-viewer-20261020'", "version:'shuolee-viewer-20261025',gaussianScale:1")
    js=js.replace('async function displayMode(){','''// Scale each covariance about its own center, never the whole mesh transform.
// Multiplying standard deviations by s gives Sigma_display = s^2 Sigma.
function attachGaussianScale(){
  gaussianScaleUniform=dyno.dynoFloat(Number($('gaussian-scale').value));
  mesh.objectModifier=dyno.dynoBlock({gsplat:dyno.Gsplat},{gsplat:dyno.Gsplat},({gsplat})=>{
    const {scales}=dyno.splitGsplat(gsplat).outputs;
    return {gsplat:dyno.combineGsplat({gsplat,scales:dyno.mul(scales,gaussianScaleUniform)})};
  });
  mesh.updateGenerator();
  updateGaussianScale();
}
function updateGaussianScale(){
  const value=Math.max(.05,Math.min(2,Number($('gaussian-scale').value)||1));
  $('gaussian-scale').value=String(value);
  text('gaussian-scale-value',value.toFixed(2)+' ×');
  if(gaussianScaleUniform){gaussianScaleUniform.value=value;mesh?.updateGenerator();}
  window.viewerState.gaussianScale=value;
}
async function displayMode(){''',1)
    js=js.replace("$('point-options').hidden=mode!=='points';", "$('point-options').hidden=mode!=='points';\n    $('gaussian-options').hidden=mode!=='gaussian'||!mesh;")
    js=js.replace('scene.add(mesh);await mesh.initialized;','scene.add(mesh);await mesh.initialized;attachGaussianScale();',1)
    js=js.replace('function disposeActive(){','function disposeActive(){\n  gaussianScaleUniform=null;',1)
    js=js.replace("$('render-mode').onchange=displayMode;", "$('render-mode').onchange=displayMode;\n  $('gaussian-scale').oninput=updateGaussianScale;\n  $('gaussian-scale-reset').onclick=()=>{$('gaussian-scale').value='1';updateGaussianScale();};")
    (ROOT/'viewer.js').write_text(js,encoding='utf-8')
print(json.dumps({'patched':str(ROOT),'scale':'0.05–2.00; center/rotation/SH/opacity unchanged'},ensure_ascii=False))
