import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const V='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/viewer';
const {normalizeCameras,cameraBundle}=await import(pathToFileURL(V+'/camera-io.js'));
const THREE=await import(pathToFileURL(V+'/vendor/three.module.js'));
const result={schema:'camera-io-regression-v1',tests:[],realWorldAccuracyClaimed:false};
function test(name,fn){fn();result.tests.push({name,passed:true});}
const source={frame:'asymmetric',wh:[1600,899],K:[[701.2,0,834.25],[0,659.7,423.125],[0,0,1]],w2c:[[0,0,1,.23],[0,1,0,-.1],[-1,0,0,2],[0,0,0,1]]};
let normalized;
test('w2c inverse and half pixel exactly once',()=>{
  normalized=normalizeCameras({records:[source],pixelConvention:'integer_centers'}).cameras[0];
  assert.deepEqual(normalized.rotation,[[0,0,-1],[0,1,0],[1,0,0]]);
  assert.deepEqual(normalized.position,[2,.1,-.23]);assert.equal(normalized.cx,834.75);
  assert.deepEqual(normalizeCameras(cameraBundle([normalized])).cameras,[normalized]);
});
test('full K to actual Three.js projection with off-center and unequal focal lengths',()=>{
  const c=normalized,r=c.rotation,p=new THREE.PerspectiveCamera(2*Math.atan(c.height/(2*c.fy))*180/Math.PI,(c.width/c.height)*(c.fy/c.fx),.02,1000);
  p.position.fromArray(c.position);p.quaternion.setFromRotationMatrix(new THREE.Matrix4().set(r[0][0],-r[0][1],-r[0][2],0,r[1][0],-r[1][1],-r[1][2],0,r[2][0],-r[2][1],-r[2][2],0,0,0,0,1));
  p.updateMatrixWorld();p.updateProjectionMatrix();p.projectionMatrix.elements[8]=1-2*c.cx/c.width;p.projectionMatrix.elements[9]=2*c.cy/c.height-1;
  let error=0;
  for(const point of [[.1,.2,2],[.3,-.2,4],[-.4,.1,3]]){
    const q=source.w2c.slice(0,3).map(row=>row[0]*point[0]+row[1]*point[1]+row[2]*point[2]+row[3]);
    const expected=[c.fx*q[0]/q[2]+c.cx,c.fy*q[1]/q[2]+c.cy];
    const ndc=new THREE.Vector3(...point).project(p),pixel=[(ndc.x+1)*c.width/2,(1-ndc.y)*c.height/2];
    error=Math.max(error,...pixel.map((v,i)=>Math.abs(v-expected[i])));
  }
  assert.ok(error<1e-9);result.maxProjectionErrorPx=error;
});
test('malformed, reflected, duplicated, skewed and model-mismatch inputs rejected',()=>{
  const mutations=[c=>c.fx=-1,c=>c.rotation[0][0]=2,c=>c.rotation[1][1]=-1,c=>c.position[0]=null,c=>c.width=0,c=>c.w2c=source.w2c];
  for(const f of mutations){const c=structuredClone(normalized);f(c);assert.throws(()=>normalizeCameras([c]));}
  assert.throws(()=>normalizeCameras([normalized,normalized]));
  const skew=structuredClone(source);skew.K[0][1]=2;assert.throws(()=>normalizeCameras({records:[skew]}));
  assert.throws(()=>normalizeCameras({cameras:[normalized],model:{sha256:'a'}},{expectedSha256:'b'}));
});
test('real 119 camera sidecars agree with model metadata and roundtrip',()=>{
  result.realModels=[];
  for(const id of ['new-capture-119-sofa-clothed','new-capture-119-portrait-clothed']){
    const meta=JSON.parse(fs.readFileSync(path.join(V,'model-'+id+'.json')));
    const side=JSON.parse(fs.readFileSync(path.join(V,'models',id,'cameras.json')));
    const parsed=normalizeCameras(side,{expectedSha256:meta.sha256}).cameras;
    const compare=normalizeCameras(meta).cameras;
    let error=0;
    for(let i=0;i<parsed.length;i++){
      for(const k of ['fx','fy','cx','cy','width','height'])assert.equal(parsed[i][k],compare[i][k]);
      for(let j=0;j<3;j++){error=Math.max(error,Math.abs(parsed[i].position[j]-compare[i].position[j]));for(let k=0;k<3;k++)error=Math.max(error,Math.abs(parsed[i].rotation[j][k]-compare[i].rotation[j][k]));}
    }
    assert.ok(error<1e-9);assert.deepEqual(normalizeCameras(cameraBundle(parsed)).cameras,parsed);
    result.realModels.push({id,cameras:parsed.length,maxPoseDifference:error});
  }
});
test('raw fisheye and explicit residual distortion cannot silently become pinhole',()=>{
  for(const field of [{projectionModel:'fisheye'},{camera_model:'FISHEYE_8'},{D:[[.1],[0],[0],[0]]},{distortion:[.1,0,0,0]},{distortion_coefficients:[null,0]}]){
    assert.throws(()=>normalizeCameras({cameras:[{...normalized,...field}]}));
    assert.throws(()=>normalizeCameras({cameras:[normalized],...field}));
  }
  assert.equal(normalizeCameras({cameras:[{...normalized,D:[0,0,0,0]}]}).cameras.length,1);
});
fs.writeFileSync(new URL('./viewer-camera-tests.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
