import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
import {webcrypto} from 'node:crypto';
import assert from 'node:assert/strict';
if(!globalThis.crypto)globalThis.crypto=webcrypto;
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001';
const out=root+'/shuolee/current/robust-portrait-122/covariance-sh-ply';
const {scenePackage,unpackSceneZip}=await import(pathToFileURL(root+'/viewer/scene-package.js'));
const {storedZip}=await import(pathToFileURL(root+'/viewer/camera-formats.js'));
const raw=fs.readFileSync(out+'/point_cloud.ply');
const recorded=JSON.parse(fs.readFileSync(out+'/cameras.json','utf8'));
// The synthetic native record contains redundant w2c and c2w for diagnosis.
// The portable reader correctly rejects ambiguous dual pose fields. Verify the
// algebraic equivalence before serializing only standard rotation/position.
let maxPoseDifference=0;
const cameras=recorded.map(c=>{
  for(let i=0;i<3;i++){
    for(let j=0;j<3;j++)maxPoseDifference=Math.max(maxPoseDifference,Math.abs(c.w2c[i][j]-c.rotation[j][i]));
    const t=-c.position.reduce((sum,x,j)=>sum+c.rotation[j][i]*x,0);
    maxPoseDifference=Math.max(maxPoseDifference,Math.abs(c.w2c[i][3]-t));
  }
  const {w2c,...portable}=c;return portable;
});
assert.ok(maxPoseDifference<1e-6);
fs.writeFileSync(out+'/portable-cameras.json',JSON.stringify(cameras,null,2)+'\n',{flag:'wx'});
fs.writeFileSync(out+'/portable-camera-adapter-check.json',JSON.stringify({initial_rejection:'Both w2c and rotation/position in synthetic diagnostic record',verifiedMaximumPoseDifference:maxPoseDifference,fix:'Emit standard rotation/position only after comparing both transforms; keep original evidence',native_images_rerun:false},null,2)+'\n',{flag:'wx'});
const packed=await scenePackage(raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength),'ply',cameras);
const entries=await unpackSceneZip(packed.bytes);
const manifestEntry=entries.find(x=>x.name==='scene.json');
const manifest=JSON.parse(new TextDecoder().decode(manifestEntry.data));
manifest.rendererContract.gaussianEvaluation='ray-3d';
manifest.syntheticNegativeControl=true;
const files=Object.fromEntries(entries.map(x=>[x.name,x.name==='scene.json'?JSON.stringify(manifest):x.data]));
fs.writeFileSync(out+'/expected-reject-native-ray3d.zip',storedZip(files),{flag:'wx'});
console.log(JSON.stringify({scope:'Synthetic native-contract rejection via existing OpenFile UI; no model promotion',cameraCount:cameras.length,points:6,path:out+'/expected-reject-native-ray3d.zip'}));
