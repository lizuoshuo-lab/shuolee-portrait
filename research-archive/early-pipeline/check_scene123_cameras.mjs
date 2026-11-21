import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const args=process.argv.slice(2),explicitReceipt=args[0]==='--receipt',subject=explicitReceipt?null:args[0];
if(explicitReceipt?(args.length!==2||!args[1]):args.length!==1)throw Error('Use <subject> or --receipt <actual-staged-delivery.json>');
const t='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001';
const rec=t+'/shuolee/current/local-clothed-surface-123';
const receiptPath=explicitReceipt?path.resolve(args[1]):`${rec}/${subject}-staged-delivery.json`;
const receipt=JSON.parse(await fs.readFile(receiptPath,'utf8'));
const reportPath=suffix=>explicitReceipt?path.join(path.dirname(receiptPath),`package-camera-parity${suffix}.json`):`${rec}/${subject}-package-camera-parity${suffix}.json`;
const rows=JSON.parse(await fs.readFile(receipt.camera_file.replace('cameras.json','camera-records.json'),'utf8')).records;
const {unpackSceneZip}=await import(pathToFileURL(t+'/viewer/scene-package.js'));
const {normalizeCameras}=await import(pathToFileURL(t+'/viewer/camera-io.js'));
const zip=await fs.readFile(receipt.zip);const entries=await unpackSceneZip(zip.buffer.slice(zip.byteOffset,zip.byteOffset+zip.byteLength));
const json=n=>JSON.parse(new TextDecoder().decode(entries.find(e=>e.name===n).data));
const cameras=normalizeCameras(json('cameras.json')).cameras;
const provenance=normalizeCameras(json('camera_provenance.json'),{expectedSha256:receipt.source_sha256}).cameras;
if(cameras.length!==rows.length)throw Error('camera count mismatch');
if(provenance.length!==rows.length)throw Error('provenance count mismatch');
let maxK=0,maxR=0,maxT=0,maxP=0;
for(let n=0;n<rows.length;n++){
  const a=rows[n],b=cameras[n],p=provenance[n],E=a.w2c;
  if(a.frame!==b.img_name||a.frame!==p.img_name||a.source_frame_index!==p.raw_source_frame||a.wh[0]!==b.width||a.wh[1]!==b.height){
    const failure={passed:false,index:n,reason:'frame or dimensions mismatch',source:{frame:a.frame,source_frame_index:a.source_frame_index,wh:a.wh},parsed:b};
    await fs.writeFile(reportPath('-initial-failure'),JSON.stringify(failure,null,2)+'\n');throw Error(JSON.stringify(failure));
  }
  for(const key of ['width','height','fx','fy','cx','cy','rotation','position'])if(JSON.stringify(b[key])!==JSON.stringify(p[key]))throw Error(`Standard and provenance differ: ${n} ${key}`);
  for(const [x,y] of [[a.K[0][0],b.fx],[a.K[1][1],b.fy],[a.K[0][2],b.cx],[a.K[1][2],b.cy]])maxK=Math.max(maxK,Math.abs(x-y));
  for(let i=0;i<3;i++){
    for(let j=0;j<3;j++)maxR=Math.max(maxR,Math.abs(E[i][j]-b.rotation[j][i]));
    const ti=-b.rotation.reduce((sum,row,j)=>sum+row[i]*b.position[j],0);maxT=Math.max(maxT,Math.abs(E[i][3]-ti));
  }
  // Synthetic points expressed in each source camera, only a coordinate transfer check.
  for(const p of [[0,0,3],[.2,.3,2],[-.4,.1,4]]){
    const world=E[0].slice(0,3).map((_,j)=>E.slice(0,3).reduce((sum,row,i)=>sum+row[j]*(p[i]-E[i][3]),0));
    const old=E.slice(0,3).map(row=>row.slice(0,3).reduce((s,v,j)=>s+v*world[j],row[3]));
    const fresh=b.rotation[0].map((_,i)=>b.rotation.reduce((s,row,j)=>s+row[i]*(world[j]-b.position[j]),0));
    const du=b.fx*(old[0]/old[2]-fresh[0]/fresh[2]),dv=b.fy*(old[1]/old[2]-fresh[1]/fresh[2]);maxP=Math.max(maxP,Math.hypot(du,dv));
  }
}
if(maxK>1e-9||maxR>1e-9||maxT>2e-6||maxP>.01)throw Error(JSON.stringify({maxK,maxR,maxT,maxP}));
const result={passed:true,cameras:cameras.length,frame_identity_preserved:true,full_K_preserved:true,max_K_difference:maxK,max_rotation_difference:maxR,max_translation_difference:maxT,max_synthetic_projection_difference_px:maxP,physical_calibration_accepted:false,initial_check_correction:'Standard 3DGS array preserves img_name; source video indices and labels intentionally live in bound camera_provenance.json. Corrected audit reads both, verifies their shared K and poses, and checks original source identity in provenance. Package and training cameras were not changed.'};
await fs.writeFile(reportPath(''),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
