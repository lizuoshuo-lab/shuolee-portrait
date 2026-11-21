import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const V='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer';
const H='C:/Users/Administrator/Documents/New project/shuolee-slide-update';
const {normalizeCameras}=await import(pathToFileURL(V+'/camera-io.js'));
const {cameras3dgs,colmapCameraFiles,storedZip}=await import(pathToFileURL(V+'/camera-formats.js'));
const dir=path.join(H,'camera-formats121-test');fs.mkdirSync(dir,{recursive:true});
const result={tests:[],actualThirdPartyUiTested:false};
for(const id of ['new-capture-119-sofa-clothed','new-capture-119-portrait-clothed']){
  const meta=JSON.parse(fs.readFileSync(V+'/model-'+id+'.json'));
  const rows=normalizeCameras(meta).cameras;
  const standard=cameras3dgs(rows);assert.ok(Array.isArray(standard));
  const back=normalizeCameras(standard).cameras;
  for(let i=0;i<rows.length;i++)for(const k of ['fx','fy','cx','cy','width','height','rotation','position'])assert.deepEqual(back[i][k],rows[i][k]);
  fs.writeFileSync(path.join(dir,id+'.json'),JSON.stringify(standard));
  const files=colmapCameraFiles(rows,{model:{sha256:meta.sha256}});
  const out=path.join(dir,id);fs.mkdirSync(out,{recursive:true});
  for(const [n,text] of Object.entries(files))fs.writeFileSync(path.join(out,n),text);
  fs.writeFileSync(path.join(dir,id+'.zip'),storedZip(files));
  result.tests.push({model:id,cameras:rows.length,standardArrayRoundtrip:true,colmapZipWritten:true});
}
// Exercise 180-degree branches and path sanitization without fake observations.
const base={id:0,img_name:'test name',width:640,height:480,fx:501,fy:499,cx:322.2,cy:240.7,position:[.3,-.2,2]};
const variants=[[[1,0,0],[0,1,0],[0,0,1]],[[1,0,0],[0,-1,0],[0,0,-1]],[[-1,0,0],[0,1,0],[0,0,-1]],[[-1,0,0],[0,-1,0],[0,0,1]]].map((rotation,i)=>({...base,img_name:'view / '+i,rotation}));
fs.writeFileSync(path.join(dir,'branches.json'),JSON.stringify(variants));
const branchdir=path.join(dir,'branches');fs.mkdirSync(branchdir,{recursive:true});for(const[n,t]of Object.entries(colmapCameraFiles(variants)))fs.writeFileSync(path.join(branchdir,n),t);
fs.writeFileSync(path.join(H,'camera-formats121-tests.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
