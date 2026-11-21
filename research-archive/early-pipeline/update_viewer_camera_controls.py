from pathlib import Path
import shutil

V=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\viewer')
B=Path(__file__).parent/'viewer-camera-before-20261027'
B.mkdir(exist_ok=True)
for name in ['viewer.js','index.html','viewer.css']:
    if not (B/name).exists():shutil.copy2(V/name,B/name)
s=(V/'viewer.js').read_text(encoding='utf-8-sig')
def sub(old,new):
    global s
    assert s.count(old)==1,(old[:80],s.count(old))
    s=s.replace(old,new)
sub("import { inspectPly, parsePointCloud } from './point-cloud-loader.js';", "import { inspectPly, parsePointCloud } from './point-cloud-loader.js';\nimport { normalizeCameras, cameraBundle } from './camera-io.js';")
sub("let importCounter=0,loadToken=0,downloadUrl=null;", "let importCounter=0,loadToken=0,downloadUrl=null;\nlet cameraWorld=null, cameraSourceModel=null;\nconst moveKeys=new Set();let moveTime=0;")
sub("version:'shuolee-viewer-20261025'", "version:'shuolee-viewer-20261027-camera-wasd'")
sub("'next','refresh-library'].forEach", "'next','refresh-library','open-cameras','download-view'].forEach")
sub("function configureCameras(list){\n  stop();cameras=list||[];", "function configureCameras(list){\n  stop();moveKeys.clear();cameras=list||[];\n  text('camera-status',cameras.length?'已加载 '+cameras.length+' 个相机 · 焦距、主点与位姿完整保留':'尚无相机 · 可加载与当前模型配套的 camera JSON');\n  $('download-cameras').disabled=!cameras.length;\n  window.viewerState.cameraCount=cameras.length;")
sub("configureCameras(record.meta?(record.meta.cameras||defaultCameras):[]);", "cameraWorld=record.importedCameraWorld||metadata?.coordinateSystem||null;\n  cameraSourceModel=record.importedCameraModel||null;\n  configureCameras(record.importedCameras||(record.meta?(record.meta.cameras||defaultCameras):[]));")
sub("function frame(t){\n  try{", "function frame(t){\n  try{\n    const dt=Math.min(Math.max((t-moveTime)/1000,0),.05);moveTime=t;\n    if(moveKeys.size&&!busy&&document.activeElement===renderer.domElement)moveCamera(dt);")
sub("viewport.appendChild(renderer.domElement);", "viewport.appendChild(renderer.domElement);\n  renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','3D场景 · 点击后使用 WASD 移动');\n  renderer.domElement.addEventListener('pointerdown',()=>renderer.domElement.focus({preventScroll:true}));\n  renderer.domElement.addEventListener('keydown',e=>{\n    if(e.code==='Escape'){moveKeys.clear();renderer.domElement.blur();return;}\n    if(e.altKey||e.ctrlKey||e.metaKey||busy)return;\n    if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE'].includes(e.code)){\n      e.preventDefault();\n      if(!moveKeys.has(e.code)){moveKeys.add(e.code);if(e.shiftKey)moveKeys.add('Shift');else moveKeys.delete('Shift');moveCamera(1/60);}\n    }else if(e.code.startsWith('Shift'))moveKeys.add('Shift');\n  });\n  window.addEventListener('keyup',e=>{moveKeys.delete(e.code);if(e.code.startsWith('Shift'))moveKeys.delete('Shift');});\n  renderer.domElement.addEventListener('blur',()=>moveKeys.clear());\n  window.addEventListener('blur',()=>moveKeys.clear());")
sub("$('open-files').onclick=()=>$('file-input').click();", "$('open-files').onclick=()=>$('file-input').click();\n  $('open-cameras').onclick=()=>$('camera-input').click();\n  $('camera-input').onchange=async()=>{const f=$('camera-input').files[0];if(f)await importCameras(f);$('camera-input').value='';};\n  $('download-cameras').onclick=()=>saveCameras(false);\n  $('download-view').onclick=()=>saveCameras(true);")
sub("if(document.hidden)stop();", "if(document.hidden){stop();moveKeys.clear();}")
# Interpolated path previews must retain each source camera's fx/fy and off-center principal point.
# This is a viewing interpolation only; captured views and downloads remain measured records.
sub("camera.fov=2*Math.atan(cameras[a].height/(2*cameras[a].fy))*180/Math.PI;camera.updateProjectionMatrix();\n      selected=Math.round(idx);", "selected=Math.round(idx);size();\n      text('camera-status','路径插值仅用于浏览；下载仍保留原始逐帧相机');\n      ")
extra=r'''
function moveCamera(dt){
  if(!controls||busy||!current)return;
  const x=Number(moveKeys.has('KeyD'))-Number(moveKeys.has('KeyA'));
  const y=Number(moveKeys.has('KeyE'))-Number(moveKeys.has('KeyQ'));
  const z=Number(moveKeys.has('KeyW'))-Number(moveKeys.has('KeyS'));
  if(!(x||y||z))return;
  stop();if(capturedFrame){capturedFrame=false;size();}
  camera.updateMatrixWorld();
  const direction=new THREE.Vector3(x,0,-z).applyQuaternion(camera.quaternion).addScaledVector(camera.up.clone().normalize(),y);
  direction.normalize().multiplyScalar(Math.max(radius*.4,.0001)*Number($('move-speed').value)*(moveKeys.has('Shift')?3:1)*dt);
  camera.position.add(direction);controls.target.add(direction);controls.update();
  text('mode','自由移动 · WASD 前后左右 · Q/E 升降');
  window.viewerState.view='keyboard-free-view';
}
async function importCameras(file){
  const record=current,token=loadToken;
  if(busy||!record){text('camera-status','请先打开模型，再加载对应相机。');return;}
  try{
    if(file.size>20*1024*1024)throw Error('相机 JSON 超过 20 MB');
    let expectedSha256=metadata?.sha256||record.fileSha256;
    if(!expectedSha256&&record.file){
      expectedSha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await record.file.arrayBuffer())),b=>b.toString(16).padStart(2,'0')).join('');
      record.fileSha256=expectedSha256;
    }
    const parsed=normalizeCameras(JSON.parse((await file.text()).replace(/^\uFEFF/,'')),{expectedSha256});
    if(token!==loadToken||record!==current)throw Error('模型已切换，请重新加载相机');
    record.importedCameras=parsed.cameras;record.importedCameraWorld=parsed.world;
    record.importedCameraModel={sha256:parsed.sourceModelHash||expectedSha256||null,name:record.name||record.url};
    cameraWorld=parsed.world;cameraSourceModel=record.importedCameraModel;
    followLatest(false);configureCameras(parsed.cameras);$('camera-section').hidden=false;$('camera-section').open=true;
    choose(0);
    text('camera-status','已加载 '+cameras.length+' 个相机 · '+file.name+'\n'+(parsed.sourceModelHash&&expectedSha256?'与当前模型 SHA256 匹配':'同一世界坐标需由配套导出保证；未自动对齐')+(parsed.notes.length?'\n'+parsed.notes.join('；'):''));
    notice('相机只用于查看；不会修改点云或训练数据。');
  }catch(e){text('camera-status','相机未加载，保留原相机：'+e.message);}
}
function saveCameras(currentOnly){
  if(!current||busy)return;
  try{
    let list=cameras;
    if(currentOnly){
      camera.updateMatrixWorld();
      const e=camera.matrixWorld.elements,p=camera.projectionMatrix.elements;
      const dimensions=capturedFrame&&cameras[selected]?new THREE.Vector2(cameras[selected].width,cameras[selected].height):renderer.getSize(new THREE.Vector2());
      const width=Math.round(dimensions.x),height=Math.round(dimensions.y);
      list=[{id:0,img_name:'view_'+Date.now(),width,height,fx:p[0]*width/2,fy:p[5]*height/2,
        cx:(1-p[8])*width/2,cy:(1+p[9])*height/2,
        rotation:[[e[0],-e[4],-e[8]],[e[1],-e[5],-e[9]],[e[2],-e[6],-e[10]]],position:camera.position.toArray(),
        synthetic:!capturedFrame||playing,label:playing?'浏览插值视角':capturedFrame?cameraLabel(selected):'手动自由视角',split:'viewer_export_not_training'}];
    }
    const bundle=cameraBundle(list,{model:cameraSourceModel||{name:current.name||current.url,sha256:metadata?.sha256||current.fileSha256||null},world:cameraWorld,selected:currentOnly?0:selected});
    const url=URL.createObjectURL(new Blob([JSON.stringify(bundle,null,2)+'\n'],{type:'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=(current.id||'model').replace(/[^a-zA-Z0-9_-]/g,'_')+(currentOnly?'_view.json':'_cameras.json');a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
    text('camera-status','已导出 '+list.length+' 个相机 · '+a.download+' · 保留原坐标与完整 K');
  }catch(e){text('camera-status','相机导出失败：'+e.message);}
}
'''
s=s.replace('function frame(t){',extra+'\nfunction frame(t){')
(V/'viewer.js').write_text(s,encoding='utf-8',newline='\n')
h=(V/'index.html').read_text(encoding='utf-8-sig')
assert 'id="open-cameras"' not in h
h=h.replace('左键旋转 · 右键平移 · 滚轮缩放 · 双击聚焦点','左键旋转 · 右键平移 · 滚轮缩放<br>点击场景后：WASD 移动 · Q/E 升降 · Shift 加速 · Esc 退出键盘')
h=h.replace('<details id="camera-section">','''<div class="camera-tools"><div class="sectionlabel">CAMERA · 相机文件</div>
<div class="import-actions"><button id="open-cameras">加载相机 JSON</button><button id="download-cameras" disabled>下载全部相机</button></div>
<button id="download-view">下载当前视角</button><input id="camera-input" type="file" accept=".json" hidden>
<p id="camera-status" class="micro" role="status">加载与当前模型配套的 camera JSON</p>
<p class="micro">支持 3DGS cameras.json 与本站导出相机；保留完整焦距、主点和位姿。相机须与点云处于同一坐标系。</p></div>
<details id="camera-section">''')
h=h.replace('<div class="display-grid"><label>显示精度','<label class="sectionlabel" for="move-speed">WASD 移动速度</label><select id="move-speed"><option value="0.1">精细 · 0.1×</option><option value="1" selected>标准 · 1×</option><option value="3">快速 · 3×</option></select><div class="display-grid"><label>显示精度')
h=h.replace('src="viewer.js"','src="viewer.js?v=20261027-camera-wasd"')
(V/'index.html').write_text(h,encoding='utf-8',newline='\n')
c=(V/'viewer.css').read_text(encoding='utf-8-sig')
c+='\n.camera-tools{border-top:1px solid #34403f;padding:16px 0 8px}.camera-tools>button{width:100%}#camera-status{white-space:pre-line;color:#c1e9d2;overflow-wrap:anywhere}canvas:focus-visible{outline:2px solid #9de2bd;outline-offset:-3px}#hint{max-width:calc(100% - 145px);line-height:1.65}\n'
(V/'viewer.css').write_text(c,encoding='utf-8',newline='\n')
print('Updated camera import/export and viewport-scoped keyboard navigation. Backup:',B)
