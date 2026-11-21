import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { SparkRenderer, SplatMesh, SplatFileType, dyno } from '@sparkjsdev/spark';
import { inspectPly, parsePointCloud } from './point-cloud-loader.js';
import { normalizeCameras, cameraBundle } from './camera-io.js?v=20261027-5';
import { cameras3dgs, colmapCameraFiles, storedZip } from './camera-formats.js?v=20261027-5';
import { scenePackage, unpackSceneZip, localFilePath, primaryModelFiles, pairedCameras } from './scene-package.js?v=20261030-reference-pose';

const $ = id => document.getElementById(id);
const viewport = $('viewport');
const BUILTINS = ["demo2-vhap-video-first", "demo2-vhap-profile-v1", "cross-person-124-sofa-front24-clock18k240-sh0-s018000", "cross-person-124-sofa-front24-sh0-232-sh0-s018000", "video21-exploratory179-sh0-s001000", "msv2-video15-parent1-child4-swinir4x-20261106", "msv2-video15-head-swinir4x-20261105", "msv2-video15-original-20261105", "msv2-video15-interpolation2x-20261105", "msv2-video15-swinir4x-20261105", "msv2-demo-fullframe59-swinir4x-20261105", "msv2-demo-fullframe59-interpolation2x-20261105", "msv2-demo-fullframe59-original-20261105", "msv2-demo-fullframe-interpolation2x-20261104", "msv2-demo-fullframe-original-20261104", "cross-person-126-football-half12-paired18k126-sh0-s018000", "cross-person-124-winterhat-half8-paired18k124-sh0-s018000", "cross-person-124-sofa-half8-paired18k124-sh0-s018000", "cross-person-124-indoor-circle16-paired18k124-sh0-s018000", "cross-person-124-indoor-paired-background18k124-sh0-s018000", "cross-person-124-indoor-paired-background124-sh0-s001000", "cross-person-124-indoor-zoom-clock18k124-sh0-s018000", "cross-person-124-winterhat-clock18k-cache124-sh0-s018000", "cross-person-124-indoor-selected-sh0-pipeline124-sh0-s018000", "cross-person-124-grass-selected-sh0-pipeline124-sh0-s018000", "cross-person-124-winterhat-selected-sh0-pipeline124-sh0-s018000", "cross-person-124-sofa-selected-sh0-pipeline124-sh0-s018000", "remy-20261103-112806", "local-surface-123-demo-body-smplx-offset-s120000-sh0", "local-surface-123-demo-body-smplx-offset-s120000", "local-surface-123-demo-body-smplx-offset-s060000", "local-surface-123-demo-body-smplx-offset-s030000", "local-surface-123-demo-body-smplx-offset-s010000", "local-surface-123-demo-head-hat-s18000", "local-surface-123-grass-official-smplifyx-native1520-shared-vertex-offset-adc300-60-s01000", "local-surface-123-grass-official-smplifyx-native1520-shared-vertex-offset-lr1e-5-s01000", "local-surface-123-grass-official-smplifyx-native1520-shared-vertex-offset-s18000", "local-surface-123-grass-official-smplifyx-native1520-shared-vertex-offset-s01000", "local-surface-123-grass-official-smplifyx-native1520-s01000-pose51", "local-surface-123-grass-official-smplifyx-native1520-s01000", "local-surface-123-grass-official-smplifyx-s18000", "local-surface-123-whiteT-s18000", "local-surface-123-winterhat-s18000", "new-capture-119-sofa-clothed", "new-capture-119-portrait-clothed", "new-capture-118-portrait-stopped1000", "new-capture-118-sofa", "new-capture-118-fixedk", "new-capture-118-probe", "whole-body-avatar114-smplx", "whole-body-avatar114-ga", "whole-body-native-align", "whole-body-shared-world", "whole-body-early-trace", "whole-body-static-ncc-off", "whole-body-static-ncc-control", "whole-body-static-human-init", "whole-body-static-surface", "whole-body-static-correspondence", "whole-body-static-fullscene", "full-scene-localization-074", "whole-body-static-registered-camera", "whole-body-static-mask-foreground", "whole-body-static-boundary", "whole-body-static-no-multiview", "whole-body-static-no-scale", "whole-body-static-pgsr", "whole-body-static-fresh", "whole-body-clothing-motion", "whole-body-clothing-fresh", "whole-body-gvhmr-current", "whole-body-best", "whole-body-smplx-aligned", "whole-body-smplx-fresh", "whole-body-density", "whole-body-dynamic", "whole-body-clothing-candidate", "original", "native-motion-mip-depthmask", "native-mip-depth003", "native-mip-resolution-full", "remy-shared", "whole-body-control", "whole-body-current", "whole-body-matched-control", "whole-body-surface-candidate", "cross-person-124-sofa-automatic-sh0-251-sh0-s018000", "cross-person-124-grass-automatic-sh0-252-sh0-s018000", "cross-person-124-winterhat-automatic-sh0-253-sh0-s018000", "cross-person-124-indoor-automatic-sh0-254-sh0-s018000", "cross-person-124-sofa-frozen-rgb-cache256-sh0-s018000", "cross-person-124-video16-cold-cache10k259-sh0-s010000", "cross-person-124-sofa-placement-cache10k261-sh0-s010000", "cross-person-124-sofa-cold-placement269-sh0-s010000", "cross-person-124-sofa-placement-bound-006", "cross-person-124-sofa-alpha-area-006", "cross-person-124-grass-coverage-alpha-005", "cross-person-124-winterhat-coverage-alpha-001", "cross-person-124-indoor-source-visibility-008", "cross-person-124-winterhat-shared-checkpoint-opacity-001", "cross-person-124-sofa-sixworker-opacity-cold-006", "cross-person-124-grass-sixworker-ffprobe-cold-005", "cross-person-124-grass-shared-checkpoint-opacity-005", "cross-person-124-video16-smpl-unified79b906f", "cross-person-124-sofa-smpl-unified79b906f", "cross-person-124-winterhat-smpl-unified79b906f", "cross-person-124-grass-smpl-unified79b906f", "cross-person-124-indoor-smpl-unified79b906f", "cross-person-124-sofa-face-photometric006"];
const EXTENSIONS = new Set(['ply','splat','spz','ksplat','sog','pcd','xyz','pts']);
const MAX_BYTES = 1024 * 1024 * 1024;
const library = [];
const collator = new Intl.Collator('zh-CN', {numeric:true,sensitivity:'base'});
let renderer,scene,camera,controls,spark,mesh,pointCloud,route,axisHelper;
let gaussianScaleUniform=null;
let current=null, metadata=null, cameras=[], defaultCameras=[], bounds=null;
let focus=new THREE.Vector3(), radius=1, selected=24, capturedFrame=false;
let playing=false,clockStart=0,playOffset=0,frames=0,lastTime=0,busy=false;
let importCounter=0,loadToken=0,downloadUrl=null;
let cameraWorld=null, cameraSourceModel=null;
const moveKeys=new Set();let moveTime=0;
let followingLatest=false,latestRevision=null,checkingLatest=false;
window.viewerState={status:'initializing',version:'shuolee-viewer-20261103-model-path',gaussianScale:1};
const extension = name => name.split('.').pop().toLowerCase();
const text = (id,value) => $(id).textContent=String(value);
const finite3 = v => Number.isFinite(v.x)&&Number.isFinite(v.y)&&Number.isFinite(v.z);

function notice(message=''){text('notice',message);}
function followLatest(enabled){
  followingLatest=enabled;$('follow-latest').checked=enabled;
  window.viewerState.followingLatest=enabled;
  text('latest-status',enabled?'每 15 秒检查新结果 · 最新一轮与质量验收分别标记':'历史对照模式 · 可随时恢复跟随');
  const q=new URLSearchParams(location.search);
  if(enabled)q.set('model','latest');else if(current&&!current.file)q.set('model',current.id);else q.delete('model');
  history.replaceState(null,'',location.pathname+(q.size?'?'+q.toString():''));
}
function showLatestDiagnostic(d){
  if(!d||!$('latest-diagnostic'))return;
  $('latest-diagnostic').hidden=false;
  text('diagnostic-title','最新诊断 · '+d.round+' · '+d.title);
  text('diagnostic-summary',d.summary);
  const url=new URL(d.reportUrl||'progress.html',location.href);
  $('diagnostic-report').href=url.origin===location.origin?url.href:'progress.html';
}
async function refreshLatest(){
  if(busy||checkingLatest)return false;
  checkingLatest=true;
  try{
    const response=await fetch('latest-result.json',{cache:'no-store'});
    if(!response.ok)throw Error('尚未发布经过导出核验的新结果');
    const latest=await response.json();
    showLatestDiagnostic(latest.latestDiagnostic);
    if(!followingLatest)return true;
    if(latest.revision===latestRevision)return true;
    if(!/^[a-z0-9-]+$/i.test(latest.modelId)||latest.compatibilityPassed!==true)throw Error('新结果尚未完成导出核验');
    const mr=await fetch('model-'+latest.modelId+'.json',{cache:'no-store'});
    if(!mr.ok)throw Error('最新模型信息未就绪');
    const meta=await mr.json();
    if(meta.sha256!==latest.sha256||meta.gaussianCount!==latest.points)throw Error('最新模型与发布记录不一致');
    const namedPoseCamera=meta.representation==='named_reference_static_snapshot'&&meta.referenceFrameIsPoseSnapshot===true&&meta.singlePlyIsNotDynamic===true&&meta.cameras?.length===1&&meta.cameras[0].img_name===meta.referenceFrameId;
    if(!namedPoseCamera&&meta.cameras?.length!==meta.realFramesUsed)throw Error('本轮相机清单不完整');
    const record={id:latest.modelId,label:meta.label,url:meta.model,name:meta.model,meta,group:'预设训练结果'};
    const index=library.findIndex(r=>r.id===record.id);
    if(index<0)library.unshift(record);else library[index]=record;
    refreshSelect();
    const priorView=selected,wasCaptured=capturedFrame;
    if(current?.id!==record.id||metadata?.sha256!==meta.sha256){
      await loadRecord(record,!!current&&$('keep-view').checked);
      if(window.viewerState.status!=='ready')return false;
      if(wasCaptured)choose(Math.min(priorView,meta.cameras.length-1));
    }else{
      current=record;metadata=meta;text('model-title',meta.label);
      text('current-model-status',meta.status+' · 文件校验通过');
      $('current-report').href=meta.report;
    }
    latestRevision=latest.revision;
    text('latest-status','自动跟随 · 第 '+latest.round+' 轮 · '+(meta.qualityAccepted?'质量已通过验收':'质量尚未通过验收'));
    window.viewerState.latestRound=latest.round;window.viewerState.followingLatest=true;
    return true;
  }catch(error){text('latest-status','保留当前模型 · '+error.message);return false;}
  finally{checkingLatest=false;}
}
function setBusy(value,message='正在载入模型…',detail=''){
  busy=value;
  $('loading').style.display=value?'flex':'none';
  $('loading').classList.remove('error');
  $('dismiss-error').hidden=true;
  text('loading-text',message);text('loading-detail',detail);
  ['open-files','open-folder','model-select','previous','next','refresh-library','open-cameras','download-view'].forEach(id=>$(id).disabled=value);
  $('download-scene').disabled=value||!current||!cameras.length;
}
function fail(error){
  console.error(error);setBusy(false);
  $('loading').style.display='flex';$('loading').classList.add('error');
  text('loading-text','文件未能打开');
  text('loading-detail',error.message?.includes('binary_compressed')?'PCD 的 binary_compressed 编码暂不支持。请导出为普通 binary、ASCII 或 PLY 后打开。':error.message||String(error));
  $('dismiss-error').hidden=false;
  window.viewerState={...window.viewerState,status:'error',error:String(error),httpStatus:error.httpStatus||null,requestedUrl:error.requestedUrl||null};
}
function size(){
  if(!renderer)return;
  const w=Math.max(1,viewport.clientWidth),h=Math.max(1,viewport.clientHeight),c=cameras[selected];
  let rw=w,rh=h;
  if(capturedFrame&&c){
    const k=Math.min(w/c.width,h/c.height);
    rw=Math.max(1,Math.round(c.width*k));rh=Math.max(1,Math.round(c.height*k));
    camera.aspect=(c.width/c.height)*(c.fy/c.fx);
  }else camera.aspect=w/h;
  renderer.setSize(rw,rh);
  Object.assign(renderer.domElement.style,{position:'absolute',left:Math.round((w-rw)/2)+'px',top:Math.round((h-rh)/2)+'px'});
  viewport.dataset.projection=capturedFrame?'captured-image':'free-view';
  updateFrameProjection();
}
function updateFrameProjection(){
  const c=capturedFrame?cameras[selected]:null;
  if(c){
    camera.aspect=(c.width/c.height)*(c.fy/c.fx);
    camera.fov=2*Math.atan(c.height/(2*c.fy))*180/Math.PI;
  }
  camera.updateProjectionMatrix();
  if(c){
    // COLMAP coordinates measure pixel centers from the upper-left image edge.
    const cx=Number.isFinite(c.cx)?c.cx:c.width/2;
    const cy=Number.isFinite(c.cy)?c.cy:c.height/2;
    camera.projectionMatrix.elements[8]=1-2*cx/c.width;
    camera.projectionMatrix.elements[9]=2*cy/c.height-1;
    camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  }
}
function pose(c){
  const r=c.rotation;
  return {p:new THREE.Vector3(...c.position),
    q:new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(r[0][0],-r[0][1],-r[0][2],0,r[1][0],-r[1][1],-r[1][2],0,r[2][0],-r[2][1],-r[2][2],0,0,0,0,1)),
    up:new THREE.Vector3(-r[0][1],-r[1][1],-r[2][1]),
    forward:new THREE.Vector3(r[0][2],r[1][2],r[2][2])};
}
function stop(){
  playing=false;text('play',metadata?.cameraSource==='canonical'?'沿诊断路径浏览':'沿拍摄路径浏览');$('play').classList.remove('active');
  if(controls)controls.enabled=true;
}
function newControls(target){
  controls?.dispose();
  controls=new OrbitControls(camera,renderer.domElement);
  controls.target.copy(target);controls.enableDamping=true;controls.dampingFactor=.12;
  controls.minDistance=Math.max(radius*1e-5,1e-5);controls.maxDistance=Math.max(radius*100,200);
  controls.screenSpacePanning=true;
  controls.addEventListener('start',()=>{stop();capturedFrame=false;size();text('mode','自由浏览');});
  controls.update();
}
function cameraLabel(i){
  const c=cameras[i];if(c?.label)return c.label;
  if(!c)return '无相机记录';
  const role=c.split?.startsWith('imported')?'导入相机':c.synthetic?'手动 / 合成视角':c.split?.startsWith('training')?'训练视角':current?.id==='remy-shared'?'来源记录视角':'记录视角';
  return (c.img_name||String(i))+' · '+role;
}
function choose(i){
  if(!cameras[i]||busy)return;
  stop();selected=i;capturedFrame=true;size();
  const c=cameras[i],p=pose(c);
  camera.position.copy(p.p);camera.up.copy(p.up);camera.quaternion.copy(p.q);
  updateFrameProjection();
  newControls(p.p.clone().addScaledVector(p.forward,Math.max(.5,p.p.distanceTo(focus))));
  $('views').value=String(i);$('timeline').value=String(i);text('viewlabel',cameraLabel(i));
  text('mode',cameraLabel(i)+(c.synthetic?' · 诊断视角':c.projectionModel==='rectified_pinhole'?' · 去畸变画幅':' · 原照片画幅'));window.viewerState.view=c.img_name;
}
function setUpAxis(){
  const axis=$('up-axis').value;
  const choices={y:[0,1,0],z:[0,0,1],'-y':[0,-1,0],'-z':[0,0,-1]};
  if(choices[axis])camera.up.fromArray(choices[axis]);
  else if(cameras.length)camera.up.copy(pose(cameras[Math.min(24,cameras.length-1)]).up);
  else camera.up.set(0,1,0);
}
function fitAll(resetOrientation=false){
  if(!bounds)return;
  stop();capturedFrame=false;size();setUpAxis();
  const center=bounds.getCenter(new THREE.Vector3());
  radius=Math.max(bounds.getSize(new THREE.Vector3()).length()/2,1e-4);
  const vertical=THREE.MathUtils.degToRad(55),horizontal=2*Math.atan(Math.tan(vertical/2)*camera.aspect);
  const distance=radius/Math.sin(Math.min(vertical,horizontal)/2)*1.12;
  // A newly imported file can use an unrelated world coordinate system.
  // Do not carry a previous model's low/reversed viewing direction into it.
  let dir=resetOrientation?new THREE.Vector3(.35,.15,1):camera.position.clone().sub(controls?.target||center);
  if(!finite3(dir)||dir.length()<1e-6)dir.set(.7,.35,1);
  dir.normalize();camera.position.copy(center).addScaledVector(dir,distance);
  camera.fov=55;camera.near=Math.max(distance/100000,1e-5);camera.far=Math.max(distance+radius*100,100);
  camera.updateProjectionMatrix();camera.lookAt(center);newControls(center);
  text('mode','整体模型 · 包含全部有效点');window.viewerState.view='overview';
}
function builtinOverview(){
  if(!cameras.length){fitAll();return;}
  stop();capturedFrame=false;selected=Math.min(24,cameras.length-1);size();
  const p=pose(cameras[selected]);camera.up.copy(p.up);
  camera.position.copy(focus).add(p.p.clone().sub(focus).multiplyScalar(1.6));
  camera.fov=64;camera.near=.02;camera.far=Math.max(1000,radius*20);camera.updateProjectionMatrix();
  camera.lookAt(focus);newControls(focus);text('mode','整体场景 · 自由浏览');
  $('views').value=String(selected);$('timeline').value=String(selected);text('viewlabel',cameraLabel(selected));
}
function configureCameras(list){
  stop();moveKeys.clear();cameras=list||[];
  text('camera-status',cameras.length?'已加载 '+cameras.length+' 个相机 · 焦距、主点与位姿完整保留':'尚无相机 · 可加载与当前模型配套的 camera JSON');
  $('download-cameras').disabled=!cameras.length;
  $('download-scene').disabled=busy||!current||!cameras.length;
  window.viewerState.cameraCount=cameras.length;
  if(route){scene.remove(route);route.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});route=null;}
  $('views').replaceChildren();
  cameras.forEach((c,i)=>{const o=document.createElement('option');o.value=String(i);o.textContent=cameraLabel(i);$('views').append(o);});
  for(const id of ['views','timeline','home','play','path'])$(id).disabled=!cameras.length;
  $('play').disabled=cameras.length<2;
  $('camera-section').hidden=!cameras.length;$('timeline').max=Math.max(0,cameras.length-1);
  if(cameras.length){
    const geometry=new THREE.BufferGeometry().setFromPoints(cameras.map(c=>new THREE.Vector3(...c.position)));
    route=new THREE.Group();
    route.add(new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0xa5ffce,transparent:true,opacity:.8,depthTest:false})));
    route.add(new THREE.Points(geometry,new THREE.PointsMaterial({color:0xffcc88,size:5,sizeAttenuation:false,depthTest:false})));
    route.visible=false;scene.add(route);
  }
  $('path').classList.remove('active');text('path','显示相机路径');
}
function disposeActive(){
  gaussianScaleUniform=null;
  if(mesh){scene.remove(mesh);mesh.dispose();mesh=null;}
  if(pointCloud){scene.remove(pointCloud);pointCloud.geometry.dispose();pointCloud.material.dispose();pointCloud=null;}
  if(axisHelper){scene.remove(axisHelper);axisHelper.dispose();axisHelper=null;}
  if(downloadUrl){URL.revokeObjectURL(downloadUrl);downloadUrl=null;}
  bounds=null;
}
function linearColors(colors){
  const out=new Float32Array(colors.length);
  for(let i=0;i<out.length;i++){
    const c=Math.max(0,Math.min(1,colors[i]));
    out[i]=c<=.04045?c/12.92:Math.pow((c+.055)/1.055,2.4);
  }
  return out;
}
function makePoints(positions,colors,colorEncoding='srgb'){
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));
  if(colors)geometry.setAttribute('color',new THREE.BufferAttribute(colorEncoding==='srgb'?linearColors(colors):colors,3));
  geometry.computeBoundingBox();geometry.computeBoundingSphere();
  const material=new THREE.PointsMaterial({size:Number($('point-size').value),sizeAttenuation:false,
    vertexColors:!!colors,color:colors?0xffffff:0xb8e8d0,transparent:false,toneMapped:false});
  material.onBeforeCompile=shader=>{
    shader.fragmentShader=shader.fragmentShader.replace('#include <clipping_planes_fragment>',
      '#include <clipping_planes_fragment>\nif (length(gl_PointCoord - vec2(0.5)) > 0.5) discard;');
  };
  const object=new THREE.Points(geometry,material);object.frustumCulled=false;scene.add(object);return object;
}
async function ensureGaussianPoints(){
  if(pointCloud||!mesh)return;
  const n=mesh.splats.getNumSplats();
  const positions=new Float32Array(n*3),colors=new Float32Array(n*3);
  let count=0;
  mesh.forEachSplat((_index,p,_scale,_quat,_opacity,c)=>{
    if(!finite3(p))return;
    positions.set([p.x,p.y,p.z],count*3);colors.set([c.r,c.g,c.b],count*3);count++;
  });
  if(!count)throw Error('没有可显示的有效高斯中心。');
  pointCloud=makePoints(positions.subarray(0,count*3),colors.subarray(0,count*3),'srgb');
  if(count!==n)notice('点模式跳过 '+(n-count)+' 个非有限坐标，高斯原文件未改变。');
  window.viewerState.pointCount=count;
}
// Scale each covariance about its own center, never the whole mesh transform.
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
function updateShLimit(){
  if(!mesh)return;
  const cap=Math.max(0,Math.min(3,Number($('sh-limit').value)||0));
  const source=mesh.splats.getNumSh(),active=Math.min(source,cap);
  // Spark rebuilds its fetch shader using maxSh; the source coefficients stay intact.
  mesh.maxSh=cap;mesh.updateGenerator();
  text('format-detail','文件 SH'+source+' · 显示 SH'+active+' · 针孔预览');
  text('sh-status',active===0?'显示 SH0：关闭随视角变化的球谐颜色。':'显示 SH'+active+'：保留到第'+active+'阶方向颜色。');
  text('sh-preview-label',active===0?'当前：SH0 · 稳定颜色':'当前：SH'+active+' · 方向颜色');
  $('sh-preview-flat').setAttribute('aria-pressed',String(cap===0));
  $('sh-preview-original').setAttribute('aria-pressed',String(cap===3));
  text('sh-preview-original',metadata?.colourPostprocess?.sourceModelId?'对比原 SH3 模型':'原始颜色对比');
  $('sh-preview-original').disabled=source===0&&!metadata?.colourPostprocess?.sourceModelId;
  window.viewerState.shDegree=source;
  window.viewerState.displayShDegree=active;
  window.viewerState.shDisplayLimit=cap;
}
function selectShLimit(value){
  $('sh-limit').value=String(value);
  updateShLimit();
  const query=new URLSearchParams(location.search);query.set('sh',$('sh-limit').value);
  history.replaceState(null,'',location.pathname+'?'+query.toString()+location.hash);
}
async function displayMode(){
  if(busy||!current)return;
  try{
    const mode=$('render-mode').value;
    if(mode==='points'&&mesh)await ensureGaussianPoints();
    if(mesh)mesh.visible=mode==='gaussian';
    spark.visible=mode==='gaussian';
    if(pointCloud)pointCloud.visible=mode==='points';
    $('point-options').hidden=mode!=='points';
    $('gaussian-options').hidden=mode!=='gaussian'||!mesh;
    $('sh-preview-controls').hidden=mode!=='gaussian'||!mesh;
    $('point-options').querySelector('p').textContent=metadata?.assetType==='sfm'?'SfM稀疏点用于检查场景和相机关系，不是完整表面或训练后的高斯。':mesh?'高斯点模式显示中心位置和基础颜色，不代表真实表面。':'普通点云按文件中的位置与颜色显示。';
    window.viewerState.renderMode=mode;
    text('header-status',mode==='points'?'点云模式':'高斯渲染');
  }catch(e){fail(e);}
}
function refreshSelect(){
  const value=current?.id||$('model-select').value;
  $('model-select').replaceChildren();
  for(const group of ['预设训练结果','场景定位诊断','推理候选','外部参考','本地模型库','导入文件']){
    const records=library.filter(r=>r.group===group).sort((a,b)=>collator.compare(a.label,b.label));
    if(!records.length)continue;
    const optgroup=document.createElement('optgroup');optgroup.label=group;
    for(const r of records){const o=document.createElement('option');o.value=r.id;o.textContent=r.label;optgroup.append(o);}
    $('model-select').append(optgroup);
  }
  if(library.some(r=>r.id===value))$('model-select').value=value;
  window.viewerState.libraryCount=library.length;
}
async function refreshLibrary(){
  let response;
  try{response=await fetch('api/models',{cache:'no-store'});if(!response.ok)return;}
  catch{return;}
  const payload=await response.json();
  for(let i=library.length-1;i>=0;i--)if(library[i].group==='本地模型库')library.splice(i,1);
  for(const f of payload.models||[]){
    if(library.some(r=>r.url===f.url))continue;
    library.push({id:'library:'+f.url,label:f.name,url:f.url,name:f.name,bytes:f.bytes,group:'本地模型库'});
  }
  refreshSelect();
  if(payload.truncated)notice('模型库超过扫描上限；可通过“导入迭代文件夹”选择其他模型。');
}
async function modelReadError(record,status){
  const requested=new URL(record.url,location.href);
  let message='无法读取模型文件（HTTP '+status+'）。\n文件：'+record.url;
  if(status===404){
    message+='\n此模型条目存在，但当前查看器未找到对应文件。';
    if(requested.origin===location.origin){
      try{
        const response=await fetch('viewer-info',{cache:'no-store',signal:AbortSignal.timeout(2000)});
        if(response.ok){
          const info=await response.json();
          if(typeof info.root==='string')message+='\n当前查看器目录：'+info.root;
        }
      }catch{}
    }
    message+='\n请将对应文件在 WPS 下载到本机，或点“返回查看器”后用“打开文件”导入点云＋相机 ZIP（不用解压）。';
    if(record.id==='remy-20261103-112806')message+='\nRemy 配套包：工作/查看器/exports/Remy_261103_112806_点云与相机.zip';
  }
  const error=new Error(message);error.httpStatus=status;error.requestedUrl=requested.href;
  return error;
}
async function readBytes(record){
  if(record.file){
    if(record.file.size>MAX_BYTES)throw Error('单文件超过 1 GiB，请先分块或导出较小的查看版本。');
    return record.file.arrayBuffer();
  }
  const response=await fetch(record.url,{cache:'no-store'});
  if(!response.ok)throw await modelReadError(record,response.status);
  const total=Number(response.headers.get('content-length'))||0;
  if(total>MAX_BYTES)throw Error('单文件超过 1 GiB，请先分块查看。');
  const reader=response.body?.getReader();
  if(!reader)return response.arrayBuffer();
  const chunks=[];let loaded=0;
  while(true){
    const result=await reader.read();if(result.done)break;
    loaded+=result.value.byteLength;
    if(loaded>MAX_BYTES){await reader.cancel();throw Error('模型超过 1 GiB 的当前查看上限。');}
    chunks.push(result.value);text('loading-detail',total?Math.round(loaded/total*100)+'% · '+record.label:(loaded/1048576).toFixed(1)+' MiB');
  }
  const joined=new Uint8Array(loaded);let offset=0;
  for(const chunk of chunks){joined.set(chunk,offset);offset+=chunk.length;}
  return joined.buffer;
}
async function loadRecord(record,keep=false){
  if(busy||!record)return;
  const token=++loadToken;
  const remembered=keep&&current&&controls?{position:camera.position.clone(),up:camera.up.clone(),target:controls.target.clone(),
    fov:camera.fov,near:camera.near,far:camera.far,selected,capturedFrame}:null;
  stop();setBusy(true,'正在载入 '+record.label);notice();disposeActive();
  current=record;metadata=record.meta||null;$('model-select').value=record.id;
  $('camera-section').querySelector('summary').textContent=metadata?.cameraSource==='canonical'?'诊断相机视角（非原始拍摄）':metadata?.assetType==='sfm'?'重建相机（去畸变投影）':'原始拍摄视角';
  $('timeline').setAttribute('aria-label',metadata?.cameraSource==='canonical'?'诊断视角序号':'拍摄视角序号');
  $('current-report').href=metadata?.report||'report.html';
  text('model-title',record.label);
  text('model-count','—');text('model-format','—');text('format-detail','正在识别文件');
  text('current-model-status',metadata?.status||'外部 / 迭代模型 · 未验收');
  $('download').hidden=true;$('metadata').hidden=!record.meta;
  if(record.meta)$('metadata').href='model-'+record.id+'.json';
  cameraWorld=record.importedCameraWorld||metadata?.coordinateSystem||null;
  cameraSourceModel=record.importedCameraModel||null;
  configureCameras(record.importedCameras||(record.meta?(record.meta.cameras||defaultCameras):[]));
  try{
    const bytes=await readBytes(record);
    if(metadata?.sha256){
      const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
      if(digest!==metadata.sha256)throw Error('当前全身模型文件校验失败，请重新传输。');
      window.viewerState.verifiedSha256=digest;
      text('current-model-status',metadata.status+' · 文件校验通过');
    }
    if(token!==loadToken)return;
    const ext=extension(record.name||record.url),head=new TextDecoder().decode(bytes.slice(0,65536));
    let kind=['pcd','xyz','pts'].includes(ext)?'points':'gaussian';
    if(ext==='ply'){
      // Compressed PlayCanvas PLY uses packed fields instead of ordinary vertex x/y/z.
      if(!head.includes('packed_position'))kind=inspectPly(bytes).kind;
    }
    if(kind==='gaussian'){
      const fileTypes={ply:SplatFileType.PLY,splat:SplatFileType.SPLAT,spz:SplatFileType.SPZ,
        ksplat:SplatFileType.KSPLAT,sog:SplatFileType.PCSOGSZIP};
      const fileType=fileTypes[ext];if(!fileType)throw Error('不支持的高斯格式：'+ext);
      const displayFilter=metadata?.filter||record.importedFilter;
      spark.preBlurAmount=displayFilter?.preBlurAmount??.3;
      spark.blurAmount=displayFilter?.blurAmount??0;
      mesh=new SplatMesh({fileBytes:new Uint8Array(bytes),fileName:record.name||record.url,fileType,
        extSplats:true,lod:false,enableLod:false,nonLod:true,editable:false,raycastable:false});
      scene.add(mesh);await mesh.initialized;
      mesh.maxSh=Number($('sh-limit').value);attachGaussianScale();
      const count=mesh.splats.getNumSplats(),sh=mesh.splats.getNumSh();
      if(!count)throw Error('文件没有有效高斯。');
      if(metadata?.gaussianCount&&count!==metadata.gaussianCount)throw Error('高斯数量与预设标记不符。');
      bounds=mesh.getBoundingBox(true);
      if(!finite3(bounds.min)||!finite3(bounds.max))throw Error('高斯包含无效坐标范围。');
      text('model-count',count.toLocaleString());text('count-label','高斯数量');
      text('model-format',ext.toUpperCase());text('format-detail','球谐阶数 '+sh+' · 针孔预览');
      $('render-mode').options[0].disabled=false;$('render-mode').value='gaussian';
      window.viewerState={...window.viewerState,gaussians:count,pointCount:count,shDegree:sh,kind:'gaussian',format:ext};
      updateShLimit();
    }else{
      const points=parsePointCloud(bytes,record.name||record.url);
      pointCloud=makePoints(points.positions,points.colors,points.colorEncoding);
      bounds=pointCloud.geometry.boundingBox.clone();
      text('model-count',points.count.toLocaleString());text('count-label','有效点数量');
      text('model-format',ext.toUpperCase());text('format-detail',points.colors?'含原始颜色':'无颜色属性');
      $('render-mode').options[0].disabled=true;$('render-mode').value='points';
      if(points.skipped)notice('跳过 '+points.skipped+' 个非有限坐标；未采样或裁减其余点。');
      window.viewerState={...window.viewerState,gaussians:0,pointCount:points.count,sourceCount:points.sourceCount,
        skipped:points.skipped,kind:'points',format:points.format,shDegree:null};
    }
    focus=metadata?.focus?new THREE.Vector3(...metadata.focus):bounds.getCenter(new THREE.Vector3());
    radius=Math.max(bounds.getSize(new THREE.Vector3()).length()/2,1e-4);
    axisHelper=new THREE.AxesHelper(Math.max(radius*.35,.01));axisHelper.position.copy(focus);
    axisHelper.visible=$('axes').value==='on';scene.add(axisHelper);
    if(remembered){
      camera.position.copy(remembered.position);camera.up.copy(remembered.up);camera.fov=remembered.fov;
      camera.near=Math.min(remembered.near,Math.max(radius/100000,1e-5));camera.far=Math.max(remembered.far,radius*100);
      selected=Math.min(remembered.selected,Math.max(0,cameras.length-1));
      capturedFrame=remembered.capturedFrame&&!!cameras.length;
      newControls(remembered.target);size();text('mode','保留观察位置 · 可点击适应整体');
    }else if(record.meta)builtinOverview();else fitAll(true);
    if(record.file){
      downloadUrl=URL.createObjectURL(record.file);$('download').href=downloadUrl;
      $('download').download=record.file.name;
    }else{$('download').href=record.url+'?download=1';$('download').download=(record.name||record.url).split('/').pop();}
    $('download').hidden=false;
    const query=new URLSearchParams(location.search);
    if(Number.isInteger(metadata?.initialView)&&!remembered){setBusy(false);choose(metadata.initialView);}
    if(query.has('view')&&record.meta&&!remembered){
      selected=Number(query.get('view'));setBusy(false);choose(selected);
    }
    if(query.get('framing')==='fit'&&!remembered){setBusy(false);fitAll();}
    setBusy(false);await displayMode();
    if(record.importedCameras&&!remembered){
      choose(record.importedInitialView??0);$('camera-section').open=true;
      text('camera-status','自动加载 '+cameras.length+' 个配套相机 · '+(record.cameraBindingVerified?'模型与相机 SHA256 匹配':'未声明绑定哈希，请确认同轮坐标')+(record.importedReferenceFrameId?'\n点云固定姿态：'+record.importedReferenceFrameId:'')+(record.cameraImportNotes?.length?'\n'+record.cameraImportNotes.join('；'):''));
    }
    window.viewerState={...window.viewerState,status:'ready',model:record.url||record.name,variant:record.id,
      filter:{preBlurAmount:spark.preBlurAmount,blurAmount:spark.blurAmount},libraryCount:library.length,
      source:'local',renderingScope:'pinhole-preview-only',nativeRenderingEquivalent:false,modelBounds:{min:bounds.min.toArray(),max:bounds.max.toArray()},error:null};
    const modelQuery=new URLSearchParams(location.search);
    if(followingLatest)modelQuery.set('model','latest');else if(!record.file)modelQuery.set('model',record.id);else modelQuery.delete('model');
    history.replaceState(null,'',location.pathname+(modelQuery.size?'?'+modelQuery.toString():''));
  }catch(e){disposeActive();setBusy(false);fail(e);}
}
async function importFiles(files){
  if(busy){notice('当前文件仍在载入，请稍后再导入。');return;}
  setBusy(true,'正在核验点云与配套相机…');
  try{
    const all=[];
    for(const file of files){
      if(extension(file.name)==='zip'){
        if(file.size>MAX_BYTES)throw Error('配套包超过1GiB');
        for(const entry of await unpackSceneZip(await file.arrayBuffer())){
          const f=new File([entry.data],entry.name.split('/').pop(),{lastModified:file.lastModified});
          f.sceneRelativePath=file.name+'/'+entry.name;all.push(f);
        }
      }else all.push(file);
    }
    const supported=await primaryModelFiles(all);
    if(!supported.length)throw Error('请选择点云＋相机配套 ZIP、文件夹，或模型文件和对应 JSON');
    supported.sort((a,b)=>collator.compare(localFilePath(a),localFilePath(b)));
    const records=[];
    for(const file of supported){
      const label=localFilePath(file),pair=await pairedCameras(file,all,supported);
      const r={id:'import:'+ ++importCounter,label,name:label,file,group:'导入文件'};
      if(pair)Object.assign(r,{label:pair.label?pair.label+' · 配套导入':label,importedSteps:pair.steps,importedCameras:pair.cameras,importedCameraWorld:pair.world,
        importedReferenceFrameId:pair.referenceFrameId,importedInitialView:pair.initialView,importedFilter:pair.filter,
        importedCameraModel:{sha256:pair.hash,name:label},fileSha256:pair.hash,
        cameraBindingVerified:pair.verifiedBinding,cameraImportNotes:pair.notes});
      records.push(r);
    }
    followLatest(false);library.push(...records);refreshSelect();setBusy(false);
    await loadRecord(records[0],false);
    if(window.viewerState.status==='ready')notice('已加入 '+records.length+' 个模型，'+records.filter(r=>r.importedCameras).length+' 个已自动配对相机。'+(!records[0].importedCameras?'仅选PLY无法自动访问旁边文件；请同时选相机JSON或导入整个文件夹。':''));
  }catch(e){setBusy(false);notice('导入未完成，保留原模型：'+e.message);}
}
function switchRelative(delta){
  const options=[...$('model-select').options],index=options.findIndex(o=>o.value===current?.id);
  const next=options[Math.max(0,Math.min(options.length-1,index+delta))];
  if(next&&next.value!==current?.id)loadRecord(library.find(r=>r.id===next.value),$('keep-view').checked);
}

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
    const context={model:cameraSourceModel||{name:current.name||current.url,sha256:metadata?.sha256||current.fileSha256||null},world:cameraWorld,selected:currentOnly?0:selected};
    const format=$('camera-format').value,stem=(current.id||'model').replace(/[^a-zA-Z0-9_-]/g,'_')+(currentOnly?'_view':'_cameras');
    const zip=format==='colmap',payload=zip?storedZip(colmapCameraFiles(list,context)):JSON.stringify(format==='3dgs'?cameras3dgs(list):cameraBundle(list,context),null,2)+'\n';
    const url=URL.createObjectURL(new Blob([payload],{type:zip?'application/zip':'application/json'}));
    const a=document.createElement('a');a.href=url;a.download=stem+(zip?'_colmap.zip':format==='native'?'_provenance.json':'.json');a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
    text('camera-status','已导出 '+list.length+' 个相机 · '+a.download+(format==='3dgs'?'\n通用3DGS数组；若目标工具忽略cx/cy，请使用COLMAP格式。':'\n保留原坐标与完整 K；不代表标定已通过。'));
  }catch(e){text('camera-status','相机导出失败：'+e.message);}
}
async function saveScene(){
  if(!current||busy||!cameras.length)return;
  const record=current,context={model:cameraSourceModel||{name:record.name||record.url,sha256:metadata?.sha256||record.fileSha256||null},label:metadata?.label||record.label,world:cameraWorld,selected,steps:metadata?.steps??record.importedSteps,
    filter:{preBlurAmount:spark.preBlurAmount,blurAmount:spark.blurAmount},
    referenceFrameId:metadata?.referenceFrameId??record.importedReferenceFrameId,
    referenceFrameIsPoseSnapshot:metadata?.referenceFrameIsPoseSnapshot??Boolean(record.importedReferenceFrameId)};
  setBusy(true,'正在打包点云＋相机…','保留原始点云，生成 SIBR 目录与 Spark 相机适配页');
  try{
    const result=await scenePackage(await readBytes(record),extension(record.name||record.url),cameras,context);
    const url=URL.createObjectURL(new Blob([result.bytes],{type:'application/zip'}));
    const a=document.createElement('a');a.href=url;a.download=(record.id||'model').replace(/[^a-zA-Z0-9_-]/g,'_')+'_scene.zip';a.click();setTimeout(()=>URL.revokeObjectURL(url),60000);
    notice('已打包原始点云与 '+result.cameraCount+' 个相机。解压可供 SIBR / Spark 使用，也可将本 ZIP 直接导回查看器。');
  }catch(e){notice('打包失败：'+e.message);}finally{setBusy(false);}
}

function frame(t){
  try{
    const dt=Math.min(Math.max((t-moveTime)/1000,0),.05);moveTime=t;
    if(moveKeys.size&&!busy&&document.activeElement===renderer.domElement)moveCamera(dt);
    if(playing&&cameras.length&&!busy){
      const n=cameras.length-1,cycle=(playOffset+(t-clockStart)/350)%(2*n),idx=cycle>n?2*n-cycle:cycle;
      const a=Math.floor(idx),b=Math.min(n,a+1),f=idx-a,p=pose(cameras[a]),q=pose(cameras[b]);
      camera.position.copy(p.p).lerp(q.p,f);camera.quaternion.copy(p.q).slerp(q.q,f);
      camera.up.copy(p.up).lerp(q.up,f).normalize();
      selected=Math.round(idx);size();
      text('camera-status','路径插值仅用于浏览；下载仍保留原始逐帧相机');
      $('timeline').value=String(selected);$('views').value=String(selected);
      text('viewlabel',cameraLabel(selected));text('mode',(metadata?.cameraSource==='canonical'?'诊断路径 · ':'拍摄路径 · ')+cameras[selected].img_name);
      // Keep the orbit pivot consistent when playback is stopped.
      const forward=new THREE.Vector3();camera.getWorldDirection(forward);
      controls.target.copy(camera.position).addScaledVector(forward,Math.max(.5,camera.position.distanceTo(focus)));
    }else controls?.update();
    renderer.render(scene,camera);frames++;
    if(t-lastTime>1000){text('fps',Math.round(frames*1000/(t-lastTime))+' FPS');frames=0;lastTime=t;}
    window.viewerState.framesRendered=(window.viewerState.framesRendered||0)+1;
  }catch(e){renderer.setAnimationLoop(null);fail(e);}
}
async function init(){
  const requestedSh=new URLSearchParams(location.search).get('sh');
  $('sh-limit').value=/^[0-3]$/.test(requestedSh||'')?requestedSh:'0';
  try{
    const response=await fetch('viewer-info',{cache:'no-store'});
    if(response.ok){const host=await response.json();if(typeof host.root==='string'&&host.identity?.startsWith('shuolee-viewer:')){text('app-source','当前文件夹：'+host.root);$('app-source').hidden=false;}}
  }catch{}

  renderer=new THREE.WebGLRenderer({antialias:false,alpha:false,preserveDrawingBuffer:true,powerPreference:'high-performance'});
  renderer.setPixelRatio(1);renderer.setClearColor(0x080b0c,1);
  renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.NoToneMapping;
  viewport.appendChild(renderer.domElement);
  renderer.domElement.tabIndex=0;renderer.domElement.setAttribute('aria-label','3D场景 · 点击后使用 WASD 移动');
  renderer.domElement.addEventListener('pointerdown',()=>renderer.domElement.focus({preventScroll:true}));
  renderer.domElement.addEventListener('keydown',e=>{
    if(e.code==='Escape'){moveKeys.clear();renderer.domElement.blur();return;}
    if(e.altKey||e.ctrlKey||e.metaKey||busy)return;
    if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE'].includes(e.code)){
      e.preventDefault();
      if(!moveKeys.has(e.code)){moveKeys.add(e.code);if(e.shiftKey)moveKeys.add('Shift');else moveKeys.delete('Shift');moveCamera(1/60);}
    }else if(e.code.startsWith('Shift'))moveKeys.add('Shift');
  });
  window.addEventListener('keyup',e=>{moveKeys.delete(e.code);if(e.code.startsWith('Shift'))moveKeys.delete('Shift');});
  renderer.domElement.addEventListener('blur',()=>moveKeys.clear());
  window.addEventListener('blur',()=>moveKeys.clear());
  scene=new THREE.Scene();camera=new THREE.PerspectiveCamera(64,1,.02,1000);camera.position.set(3,2,4);
  newControls(new THREE.Vector3());size();
  spark=new SparkRenderer({renderer,enableLod:false,accumExtSplats:true,preBlurAmount:.3,blurAmount:0,maxStdDev:Math.sqrt(8),minAlpha:1/255});
  scene.add(spark);renderer.setAnimationLoop(frame);
  $('dismiss-error').onclick=()=>{$('loading').style.display='none';notice('可重新选择其他文件；原文件没有改变。');};
  $('open-files').onclick=()=>$('file-input').click();
  $('open-cameras').onclick=()=>$('camera-input').click();
  $('camera-input').onchange=async()=>{const f=$('camera-input').files[0];if(f)await importCameras(f);$('camera-input').value='';};
  $('download-cameras').onclick=()=>saveCameras(false);
  $('download-view').onclick=()=>saveCameras(true);
  $('download-scene').onclick=saveScene;
  $('open-folder').onclick=()=>$('folder-input').click();
  for(const id of ['file-input','folder-input'])$(id).onchange=async()=>{await importFiles($(id).files);$(id).value='';};
  $('model-select').onchange=()=>{followLatest(false);loadRecord(library.find(r=>r.id===$('model-select').value),$('keep-view').checked);};
  $('previous').onclick=()=>{followLatest(false);switchRelative(-1);};$('next').onclick=()=>{followLatest(false);switchRelative(1);};
  $('refresh-library').onclick=async()=>{await refreshLibrary();if(followingLatest)await refreshLatest();else notice('模型库已刷新 · 当前为历史对照模式');};
  $('follow-latest').onchange=()=>{followLatest($('follow-latest').checked);if(followingLatest){latestRevision=null;refreshLatest();}};
  $('render-mode').onchange=displayMode;
  $('gaussian-scale').oninput=updateGaussianScale;
  $('gaussian-scale-reset').onclick=()=>{$('gaussian-scale').value='1';updateGaussianScale();};
  $('sh-limit').onchange=()=>selectShLimit($('sh-limit').value);
  $('sh-preview-flat').onclick=()=>selectShLimit(0);
  $('sh-preview-original').onclick=async()=>{
    const source=library.find(r=>r.id===metadata?.colourPostprocess?.sourceModelId);
    selectShLimit(3);
    if(source){followLatest(false);await loadRecord(source,true);}
  };
  $('point-size').oninput=()=>{text('point-size-value',$('point-size').value+' px');if(pointCloud)pointCloud.material.size=Number($('point-size').value);};
  $('overview').onclick=()=>fitAll();$('home').onclick=()=>choose(Math.min(24,cameras.length-1));
  $('views').onchange=()=>choose(Number($('views').value));$('timeline').oninput=()=>choose(Number($('timeline').value));
  $('play').onclick=()=>{if(playing){stop();return;}if(cameras.length<2)return;playing=true;capturedFrame=true;size();controls.enabled=false;clockStart=performance.now();playOffset=selected;text('play','暂停路径浏览');$('play').classList.add('active');};
  $('path').onclick=()=>{if(!route)return;route.visible=!route.visible;$('path').classList.toggle('active',route.visible);text('path',route.visible?'隐藏相机路径':'显示相机路径');};
  $('quality').onchange=()=>{renderer.setPixelRatio(Number($('quality').value));size();};
  $('background').onchange=()=>renderer.setClearColor($('background').value,1);
  $('up-axis').onchange=()=>{stop();capturedFrame=false;setUpAxis();const target=controls.target.clone();newControls(target);size();};
  $('axes').onchange=()=>{if(axisHelper)axisHelper.visible=$('axes').value==='on';};
  $('fullscreen').onclick=()=>{const p=document.fullscreenElement?document.exitFullscreen():viewport.requestFullscreen();p?.catch(e=>notice(e.message));};
  $('shot').onclick=()=>{renderer.render(scene,camera);const a=document.createElement('a');a.download='shuolee_'+(current?.label||'view').replace(/[\\/:*?"<>|]/g,'_')+'_'+Date.now()+'.png';a.href=renderer.domElement.toDataURL('image/png');a.click();};
  let dragDepth=0;
  viewport.addEventListener('dragenter',e=>{e.preventDefault();dragDepth++;$('drop-hint').hidden=false;});
  viewport.addEventListener('dragover',e=>{e.preventDefault();e.dataTransfer.dropEffect='copy';});
  viewport.addEventListener('dragleave',e=>{e.preventDefault();if(--dragDepth<=0)$('drop-hint').hidden=true;});
  viewport.addEventListener('drop',e=>{e.preventDefault();dragDepth=0;$('drop-hint').hidden=true;importFiles(e.dataTransfer.files);});
  renderer.domElement.addEventListener('dblclick',e=>{
    if(!pointCloud?.visible||busy)return;
    const rect=renderer.domElement.getBoundingClientRect();
    const ray=new THREE.Raycaster();
    ray.params.Points.threshold=Math.max(radius*.008,1e-5);
    ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1),camera);
    const hit=ray.intersectObject(pointCloud)[0];if(hit){stop();capturedFrame=false;newControls(hit.point);size();text('mode','点云 · 已聚焦所选点');}
  });
  new ResizeObserver(size).observe(viewport);window.addEventListener('resize',size);
  document.addEventListener('visibilitychange',()=>{if(document.hidden){stop();moveKeys.clear();}});
  renderer.domElement.addEventListener('webglcontextlost',e=>{e.preventDefault();stop();fail(Error('显卡上下文丢失。请刷新页面后重新打开模型，或降低显示精度。'));});
  const metas=await Promise.all(BUILTINS.map(async id=>{
    try{const r=await fetch('model-'+id+'.json',{cache:'no-store'});return r.ok?{id,meta:await r.json()}:null;}catch{return null;}
  }));
  try{const r=await fetch('cameras.json');if(r.ok)defaultCameras=await r.json();}catch{}
  for(const item of metas.filter(Boolean)){
    library.push({id:item.id,label:item.meta.label,url:item.meta.model,name:item.meta.model,meta:item.meta,group:item.id==='remy-shared'?'外部参考':item.meta.assetType==='sfm'?'场景定位诊断':item.meta.cameraSource==='canonical'?'推理候选':'预设训练结果'});
  }
  await refreshLibrary();refreshSelect();
  const requested=new URLSearchParams(location.search).get('model');
  followLatest(!requested||requested==='latest');
  const initial=library.find(r=>r.id===requested||r.url===requested)||library[0];
  setBusy(false);
  if(followingLatest&&await refreshLatest()){}
  else if(initial)await loadRecord(initial,false);
  else{configureCameras([]);text('mode','打开文件或拖入模型开始查看');text('loading-text','打开模型开始查看');window.viewerState.status='empty';}
  setInterval(()=>refreshLatest(),15000);
}
init().catch(fail);
