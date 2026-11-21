import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001';
const out=root+'/shuolee/current/local-clothed-surface-123/demo-vhap-initialization/body-gaussian-upstream-offset/delivery/step-120000/sh0-visible';
const m=await import(pathToFileURL(root+'/viewer/scene-package.js'));
const f=await import(pathToFileURL(root+'/viewer/camera-formats.js'));
const read=async name=>JSON.parse(await fs.readFile(out+'/'+name,'utf8'));
const native=await read('native-result.json'),fields=await read('field-validation.json');
if(!native.passed||native.views.length!==2||native.views.some(x=>x.max_abs!==0))throw Error('Native readback not accepted');
const bytes=await fs.readFile(out+'/point_cloud.ply');
if(await m.sha256(bytes)!==fields.derivative_sha256)throw Error('SH0 file changed');
const cameras=await read('cameras.json'),context=await read('context.json');
const packaged=await m.scenePackage(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'ply',cameras,context);
const files=Object.fromEntries((await m.unpackSceneZip(packaged.bytes)).map(x=>[x.name,x.data]));
files['README.txt']=new TextDecoder().decode(files['README.txt'])+'\nSH0稳定颜色版：来自120000步同一个模型，只移除45个高阶球谐字段。位置、尺度、旋转、透明度、基础颜色与相机逐字段不变；不是一次新训练。侧面彩斑改善不代表手臂、背面几何已修好。原SH3文件另行保留。\n';
files['colour-conversion.json']=JSON.stringify(fields,null,2)+'\n';
const zip=f.storedZip(files),unpacked=await m.unpackSceneZip(zip);
const fileObjects=unpacked.map(x=>{const file=new File([x.data],path.basename(x.name));file.sceneRelativePath=x.name;return file;});
const models=await m.primaryModelFiles(fileObjects),paired=await m.pairedCameras(models[0],fileObjects,models);
if(models.length!==1||!paired.verifiedBinding||paired.referenceFrameId!=='frame_000033'||paired.cameras.length!==1)throw Error('SH0 camera binding failed');
if(!new TextDecoder().decode(files.cfg_args).includes('sh_degree=0'))throw Error('SIBR SH0 config not preserved');
await fs.writeFile(out+'/scene.zip',zip,{flag:'wx'});
await fs.writeFile(out+'/package-result.json',JSON.stringify({updated:new Date().toISOString(),passed:true,
  sha256:await m.sha256(zip),bytes:zip.length,cameras:paired.cameras.length,verifiedBinding:paired.verifiedBinding,
  referenceFrameId:paired.referenceFrameId,shDegree:0,entries:unpacked.map(x=>x.name),sibrGuiVerified:false},null,2)+'\n',{flag:'wx'});
console.log('SH0_PACKAGE_AND_CAMERA_BINDING_VERIFIED',zip.length);
