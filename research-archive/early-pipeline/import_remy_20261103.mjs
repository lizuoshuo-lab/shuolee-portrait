import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001';
const source='C:/Users/Administrator/Documents/New project/shuolee-slide-update/remy-import-20261103';
const viewer=base+'/viewer',sync='D:/00/工作/查看器';
const id='remy-20261103-112806',label='Remy · 沉浸空间261103_112806 · 外部参考';
const m=await import(pathToFileURL(viewer+'/scene-package.js'));
const io=await import(pathToFileURL(viewer+'/camera-io.js'));
const f=await import(pathToFileURL(viewer+'/camera-formats.js'));
const bytes=await fs.readFile(source+'/3DGS.sog');
const hash=await m.sha256(bytes);
if(hash!=='5c854fd08fc765cb16167bc302be98c912e2278b6f49f845cac846ec76352f8b')throw Error('Source SOG changed');
const raw=JSON.parse(await fs.readFile(source+'/cameras.json','utf8'));
const cameras=io.normalizeCameras(raw).cameras;
if(cameras.length!==174)throw Error('Unexpected source camera roster');
const contract={schema:'shuolee-renderer-contract-v1',cameraProjection:'pinhole',gaussianEvaluation:'screen-space-2d',previewBackend:'spark',scope:'preview-only',trainingRenderer:'external-source-not-independently-verified',nativeImageEquivalence:false};
const context={label,initialView:0,model:{sha256:hash},rendererContract:contract,
  provenance:'Source output cameras.json, not raw acquisition transforms.json; source omits principal points, image-center assumption retained.'};
const pack=await m.scenePackage(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'sog',cameras,context);
const entries=await m.unpackSceneZip(pack.bytes);
const objects=entries.map(x=>{const q=new File([x.data],path.basename(x.name));q.sceneRelativePath=x.name;return q;});
const models=await m.primaryModelFiles(objects),paired=await m.pairedCameras(models[0],objects,models);
if(models.length!==1||paired.cameras.length!==174||!paired.verifiedBinding)throw Error('Package binding failed');
const sourceEntry=entries.find(e=>e.name==='model.sog');
if(await m.sha256(sourceEntry.data)!==hash)throw Error('Model must remain byte-identical');
await fs.writeFile(source+'/scene.zip',pack.bytes,{flag:'wx'});
const meta={label,assetType:'external',model:'models/'+id+'/model.sog',sha256:hash,gaussianCount:789005,
  initialView:0,cameras:f.cameras3dgs(cameras),cameraSource:'Remy output cameras.json; no raw transforms mixed',
  rendererContract:contract,qualityAccepted:false,coordinateSystem:'source model and output camera coordinates unchanged',
  status:'外部参考 · 原始SOG与174相机配套；源相机未给主点，按图像中心读取；未作重建质量验收。'};
for(const root of [viewer,sync]){
  const dir=root+'/models/'+id;
  await fs.mkdir(dir,{recursive:true});
  await fs.writeFile(dir+'/model.sog',bytes,{flag:'wx'});
  await fs.writeFile(dir+'/scene.zip',pack.bytes,{flag:'wx'});
  await fs.copyFile(source+'/cameras.json',dir+'/source-cameras.json');
  await fs.writeFile(root+'/model-'+id+'.json',JSON.stringify(meta,null,2)+'\n',{flag:'wx'});
  const script=root+'/viewer.js',old=await fs.readFile(script,'utf8');
  if(!old.includes('"'+id+'"')){
    if(!old.includes('const BUILTINS = ['))throw Error('Unexpected registry format');
    await fs.writeFile(script,old.replace('const BUILTINS = [','const BUILTINS = ["'+id+'", '));
  }
}
await fs.writeFile(source+'/import-validation.json',JSON.stringify({updated:new Date().toISOString(),id,label,
  formatPassed:true,points:789005,cameras:174,sourceModelSha256:hash,packageSha256:await m.sha256(pack.bytes),
  modelBytesPreserved:true,pairedCamerasHashVerified:true,principalPoint:'source omits cx/cy; image-center assumption',
  rawAcquisitionTransformsUsed:false,guiVerified:false,company00Verified:false,qualityAccepted:false},null,2),{flag:'wx'});
console.log('REMY_REGISTERED',id,789005,174,hash);
