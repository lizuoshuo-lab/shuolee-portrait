import fs from 'node:fs';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import vm from 'node:vm';
import {webcrypto} from 'node:crypto';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
const V='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer';
const load=n=>import(pathToFileURL(V+'/'+n));
const {normalizeCameras}=await load('camera-io.js');
const {assertPinholeCamera,assertPreviewCompatible,previewRendererContract}=await load('render-contract.js');
const {cameras3dgs,colmapCameraFiles}=await load('camera-formats.js');
const {scenePackage,unpackSceneZip,primaryModelFiles,pairedCameras}=await load('scene-package.js');
const {sparkCameraPage}=await load('spark-camera-template.js');
const c={img_name:'asymmetric',width:127,height:99,fx:81,fy:78,cx:61.125,cy:53.75,position:[0,0,0],rotation:[[1,0,0],[0,1,0],[0,0,1]],pixelConvention:'pixel_corners'};
const cases=[{projectionModel:'fisheye'},{camera_model:'OPENCV_FISHEYE'},{cameraModel:'FISHEYE_8'},{projection:{model:'ftheta'}},{projection:'fisheye'},{radial_coeffs:[.01,0,0,0]},{D:[0,NaN]},{k1:.01},{rendererContract:{...previewRendererContract(),gaussianEvaluation:'ray-3d'}}];
let checks=0;
for(const x of cases){assert.throws(()=>normalizeCameras({cameras:[{...c,...x}]}));assert.throws(()=>normalizeCameras({cameras:[c],...x}));checks+=2;}
assert.throws(()=>normalizeCameras([{...c,model:'OPENCV_FISHEYE'}]));checks++;
assert.equal(normalizeCameras({model:'models/example.ply',cameras:[c]}).cameras.length,1);checks++;
const standard=cameras3dgs([c]);assert.equal(standard[0].projectionModel,'pinhole');assert.equal(standard[0].cx,c.cx);assert.equal(standard[0].cy,c.cy);checks++;
assert.match(colmapCameraFiles([c])['cameras.txt'],/PINHOLE 127 99 81 78 61.125 53.75/);checks++;
const props=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];
const header=new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement vertex 1\n'+props.map(k=>'property float '+k+'\n').join('')+'end_header\n');
const bytes=new Uint8Array(header.length+props.length*4);bytes.set(header);const d=new DataView(bytes.buffer);d.setFloat32(header.length+2*4,2,true);d.setFloat32(header.length+13*4,1,true);
const packed=await scenePackage(bytes.buffer,'ply',[c]);
const entries=await unpackSceneZip(packed.bytes);const manifest=JSON.parse(new TextDecoder().decode(entries.find(e=>e.name==='scene.json').data));
assert.deepEqual(manifest.rendererContract,previewRendererContract());assert.equal(manifest.compatibility.nativeRenderingEquivalent,false);checks++;
const toFile=e=>({name:e.name.split('/').pop(),sceneRelativePath:e.name,size:e.data.length,arrayBuffer:async()=>e.data.buffer.slice(e.data.byteOffset,e.data.byteOffset+e.data.length),text:async()=>new TextDecoder().decode(e.data)});
const files=entries.map(toFile),models=await primaryModelFiles(files),pair=await pairedCameras(models[0],files,models);
assert.equal(pair.verifiedBinding,true);assert.equal(pair.cameras[0].cx,c.cx);checks++;
for(const contract of [{...previewRendererContract(),cameraProjection:'fisheye'},{...previewRendererContract(),gaussianEvaluation:'ray-3d'}]){
  await assert.rejects(scenePackage(bytes.buffer,'ply',[c],{rendererContract:contract}));
  const bad={...manifest,rendererContract:contract};const replacement={name:'scene.json',data:new TextEncoder().encode(JSON.stringify(bad))};
  const badFiles=entries.map(e=>toFile(e.name==='scene.json'?replacement:e));
  await assert.rejects(primaryModelFiles(badFiles));await assert.rejects(pairedCameras(models[0],badFiles,models));checks+=3;
}
const page=sparkCameraPage(),validation=page.slice(page.indexOf('function validCamera'),page.indexOf('async function load'));
const context=vm.createContext({});vm.runInContext(assertPinholeCamera.toString()+'\n'+assertPreviewCompatible.toString()+'\n'+validation,context);
assert.equal(context.validCamera(c,0).cx,c.cx);checks++;
for(const x of cases){assert.throws(()=>context.validCamera({...c,...x},0));checks++;}
for(const id of ['new-capture-119-sofa-clothed','new-capture-119-portrait-clothed']){
  const data=JSON.parse(fs.readFileSync(V+'/model-'+id+'.json'));
  const rows=normalizeCameras(data).cameras,back=normalizeCameras(cameras3dgs(rows)).cameras;
  for(let i=0;i<rows.length;i++)for(const k of ['width','height','fx','fy','cx','cy','position','rotation'])assert.deepEqual(back[i][k],rows[i][k]);checks++;
}
const result={checks,allPassed:true,realModelsCameraCounts:[89,112],nativeRendererImplemented:false,gpuOrBrowserUiTested:false,trainingStarted:false};
fs.writeFileSync(new URL('./render-contract122-tests.json',import.meta.url),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
