// Small companion page, using Spark's public SDK without changing model coordinates.
import {assertPinholeCamera,assertPreviewCompatible} from './render-contract.js?v=20261027-5';
export function sparkCameraPage(){return String.raw`<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Spark · 点云与配套相机</title>
<style>body{margin:0;color:#edf5f4;background:#101716;font:15px system-ui}header{padding:16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}h1{font-size:20px;margin:0}button,select{padding:9px;background:#25453e;color:white;border:1px solid #52776c;border-radius:5px}#stage{height:calc(100vh - 170px);min-height:250px;position:relative;background:#080b0c}#state{padding:8px 16px;white-space:pre-wrap}p{margin:8px 16px;color:#bed2cc}canvas:focus{outline:none}</style>
<header><h1>Spark · 点云与配套相机</h1><button id="open">打开点云和相机</button><button id="folder">导入整个文件夹</button><select id="views" disabled aria-label="相机"></select><button id="reset" disabled>回到所选相机</button></header>
<input id="files" type="file" multiple accept=".ply,.spz,.splat,.ksplat,.sog,.json" hidden><input id="directory" type="file" webkitdirectory multiple hidden>
<div id="stage"></div><div id="state" role="status">选择配套点云、cameras.json、scene.json；也可导入解压后的整个文件夹。</div>
<p>左键旋转 · 右键平移 · 滚轮缩放 · 点击场景后 WASD 移动、Q/E 升降。点云在本机读取；联网只加载 Spark SDK。此页为相机适配页，非 Spark 官网原版。</p>
<script type="importmap">{"imports":{"three":"https://sparkjs.dev/examples/js/vendor/three/build/three.module.js","three/addons/":"https://sparkjs.dev/examples/js/vendor/three/examples/jsm/","@sparkjsdev/spark":"https://sparkjs.dev/dist/spark.module.js"}}</script>
<script type="module">
${assertPinholeCamera.toString()}
${assertPreviewCompatible.toString()}
import * as THREE from 'three';
import {SparkRenderer,SplatMesh} from '@sparkjsdev/spark';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const $=id=>document.getElementById(id),stage=$('stage'),scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(55,1,.001,10000);
const renderer=new THREE.WebGLRenderer({antialias:false});renderer.setPixelRatio(1);renderer.setClearColor(0x080b0c);renderer.domElement.tabIndex=0;stage.append(renderer.domElement);
scene.add(new SparkRenderer({renderer,enableLod:false,preBlurAmount:.3,blurAmount:0}));
let mesh=null,cams=[],controls=null,exact=false,radius=1,active=0,busy=false;const keys=new Set();
const digest=async b=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',b)),x=>x.toString(16).padStart(2,'0')).join('');
function resize(){
 const c=cams[active],w=stage.clientWidth,h=stage.clientHeight;let rw=w,rh=h;
 if(exact&&c){const s=Math.min(w/c.width,h/c.height);rw=Math.max(1,Math.round(c.width*s));rh=Math.max(1,Math.round(c.height*s));camera.aspect=c.width/c.height*c.fy/c.fx;camera.fov=2*Math.atan(c.height/(2*c.fy))*180/Math.PI;}else camera.aspect=w/h;
 renderer.setSize(rw,rh);Object.assign(renderer.domElement.style,{position:'absolute',left:Math.round((w-rw)/2)+'px',top:Math.round((h-rh)/2)+'px'});
 camera.updateProjectionMatrix();if(exact&&c){camera.projectionMatrix.elements[8]=1-2*c.cx/c.width;camera.projectionMatrix.elements[9]=2*c.cy/c.height-1;camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();}
}
function choose(){
 const c=cams[active];if(!c)return;const r=c.rotation;
 controls?.dispose();camera.position.fromArray(c.position);camera.up.set(-r[0][1],-r[1][1],-r[2][1]);
 camera.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(r[0][0],-r[0][1],-r[0][2],0,r[1][0],-r[1][1],-r[1][2],0,r[2][0],-r[2][1],-r[2][2],0,0,0,0,1));
 controls=new OrbitControls(camera,renderer.domElement);controls.target.copy(camera.position).addScaledVector(new THREE.Vector3(r[0][2],r[1][2],r[2][2]),radius);controls.update();
 controls.addEventListener('start',()=>{exact=false;resize();});exact=true;resize();
}
function validCamera(c,i){
 assertPreviewCompatible(c);
 if(!Number.isInteger(c.width)||!Number.isInteger(c.height)||Math.min(c.width,c.height)<=0||!['fx','fy'].every(k=>Number.isFinite(c[k])&&c[k]>0)||!Array.isArray(c.position)||c.position.length!==3||!c.position.every(Number.isFinite)||!Array.isArray(c.rotation)||c.rotation.length!==3||!c.rotation.every(r=>Array.isArray(r)&&r.length===3&&r.every(Number.isFinite)))throw Error('相机 '+i+' 字段无效');
 const r=c.rotation,det=r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]);
 if(Math.abs(det-1)>1e-4||r.some((a,j)=>r.some((b,k)=>Math.abs(a.reduce((s,x,l)=>s+x*b[l],0)-(j===k?1:0))>1e-4)))throw Error('相机旋转不是 SO(3)');
 if(c.pixelConvention&&!['pixel_corners','integer_centers'].includes(c.pixelConvention))throw Error('未知像素约定');
 const shift=c.pixelConvention==='integer_centers'?.5:0,cx=c.cx==null?c.width/2:c.cx+shift,cy=c.cy==null?c.height/2:c.cy+shift;
 if(!Number.isFinite(cx)||!Number.isFinite(cy))throw Error('主点无效');return {...c,cx,cy};
}
async function load(list){
 if(busy)return;busy=true;keys.clear();$('state').textContent='正在读取与校验…';
 try{
  const files=Array.from(list),mf=files.find(f=>f.name==='scene.json'),manifest=mf?JSON.parse(await mf.text()):null;
  if(manifest&&manifest.schema!=='shuolee-scene-v1')throw Error('scene.json 版本不支持');
  assertPreviewCompatible(manifest);
  const candidates=files.filter(f=>/\.(ply|spz|splat|ksplat|sog)$/i.test(f.name)&&f.name!=='input.ply');
  const model=manifest?candidates.find(f=>(f.webkitRelativePath||f.name).endsWith('/'+manifest.model.file)||f.name===manifest.model.file.split('/').pop()):candidates.length===1?candidates[0]:null;
  const cf=files.find(f=>f.name===(manifest?.cameras?.file||'cameras.json'));
  if(!model||!cf)throw Error('请选择一个模型及其 cameras.json；多个人物请分别导入');
  const bytes=await model.arrayBuffer(),cb=await cf.arrayBuffer();
  if(manifest&&(await digest(bytes)!==manifest.model.sha256||await digest(cb)!==manifest.cameras.sha256))throw Error('模型或相机 SHA256 不匹配');
  const rows=JSON.parse(new TextDecoder().decode(cb));if(!Array.isArray(rows)||!rows.length||rows.length>100000)throw Error('需要标准3DGS相机数组');const next=rows.map(validCamera);
  if(manifest&&manifest.cameras.count!==next.length)throw Error('相机数量不一致');
  const incoming=new SplatMesh({fileBytes:new Uint8Array(bytes),fileName:model.name,lod:false,enableLod:false});await incoming.initialized;
  const box=incoming.getBoundingBox(true);radius=Math.max(box.getSize(new THREE.Vector3()).length()/2,.01);
  if(mesh){scene.remove(mesh);mesh.dispose();}mesh=incoming;scene.add(mesh);cams=next;active=0;
  $('views').replaceChildren(...cams.map((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.img_name||String(i);return o;}));$('views').disabled=false;$('reset').disabled=false;
  choose();$('state').textContent='已加载 '+mesh.splats.getNumSplats().toLocaleString()+' 个高斯 · '+cams.length+' 个相机 · '+(manifest?'模型与相机 SHA256 通过':'未提供清单，配套身份未验证')+'\n完整 fx/fy/cx/cy；针孔预览，未验证与训练原生渲染等价。';
 }catch(e){$('state').textContent='加载失败：'+e.message;}finally{busy=false;}
}
$('open').onclick=()=>$('files').click();$('folder').onclick=()=>$('directory').click();
for(const id of ['files','directory'])$(id).onchange=async()=>{await load($(id).files);$(id).value='';};
$('views').onchange=()=>{active=Number($('views').value);choose();};$('reset').onclick=choose;
renderer.domElement.addEventListener('pointerdown',()=>renderer.domElement.focus());
renderer.domElement.addEventListener('keydown',e=>{if(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE'].includes(e.code)){keys.add(e.code);e.preventDefault();}});
window.addEventListener('keyup',e=>keys.delete(e.code));window.addEventListener('blur',()=>keys.clear());renderer.domElement.addEventListener('blur',()=>keys.clear());window.addEventListener('resize',resize);
let previous=0;renderer.setAnimationLoop(t=>{const dt=Math.min((t-previous)/1000,.05);previous=t;if(controls&&keys.size){exact=false;resize();const d=new THREE.Vector3(Number(keys.has('KeyD'))-Number(keys.has('KeyA')),0,Number(keys.has('KeyS'))-Number(keys.has('KeyW'))).applyQuaternion(camera.quaternion).addScaledVector(camera.up,Number(keys.has('KeyE'))-Number(keys.has('KeyQ')));d.normalize().multiplyScalar(radius*.4*dt);camera.position.add(d);controls.target.add(d);}controls?.update();renderer.render(scene,camera);});resize();
</script></html>`;}
