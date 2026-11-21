import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const subject=process.argv[2],step=Number(process.argv[3]||18000);
if(!['grass','winterhat','sofa-selected-sh0-pipeline124','winterhat-selected-sh0-pipeline124','grass-selected-sh0-pipeline124','indoor-selected-sh0-pipeline124','winterhat-clock18k-cache124','indoor-zoom-clock18k124','indoor-paired-background124','indoor-paired-background18k124','indoor-circle16-paired18k124','sofa-half8-paired18k124','winterhat-half8-paired18k124','football-half12-paired18k126','sofa-front24-sh0-232'].includes(subject)||!Number.isInteger(step))throw Error('Explicit registered subject and step required');
const t='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001';
const dir=t+'/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124/'+subject+'/delivery/step-'+String(step).padStart(6,'0');
const m=await import(pathToFileURL(t+'/viewer/scene-package.js'));
const f=await import(pathToFileURL(t+'/viewer/camera-formats.js'));
const read=async n=>JSON.parse(await fs.readFile(dir+'/'+n,'utf8'));
const v=await read('validation.json'),c=await read('cameras.json'),context=await read('context.json'),fields=await read('sh0-field-check.json');
const plan=JSON.parse(await fs.readFile(path.resolve(dir,'../../plan.json'),'utf8'));
if(!Number.isInteger(plan.training_frames)||plan.training_frames<1)throw Error('Explicit training roster required');
const bytes=await fs.readFile(dir+'/point_cloud.ply');
const diagnosedPreview=subject==='indoor-circle16-paired18k124'&&v.diagnostic_preview_only===true&&
  v.native_source_roundoff_diagnosis?.passed===true&&v.native_rotated_roundoff_diagnosis?.passed===true&&
  v.quality_accepted===false;
if(!v.format_passed||(!v.native_rotated_render_parity.passed&&!diagnosedPreview)||!fields.passed||await m.sha256(bytes)!==v.sha256)throw Error('Native/fields validation required');
if(subject==='sofa-front24-sh0-232'){
  const receipt=path.resolve(dir,`../../gaussian-run/step-${String(step).padStart(6,'0')}/export-source-receipt232.json`);
  const source=JSON.parse(await fs.readFile(receipt,'utf8'));
  if(!source.terminal_verified||source.step!==step||source.source_ply_sha256!==v.source_sha256||!v.native_source_ply_parity.passed||
     await m.sha256(await fs.readFile(dir+'/cameras.json'))!==v.serialized_camera_sha256)throw Error('Terminal source and camera verification required');
  const K=source.reference_corner_K;
  const [cam]=c, [width,height]=source.reference_size;
  const expected=[K[0][0],K[1][1],K[0][2],K[1][2]];
  if(c.length!==1||cam.img_name!==source.reference_frame||cam.width!==width||cam.height!==height||
     cam.pixelConvention!=='pixel_corners'||[cam.fx,cam.fy,cam.cx,cam.cy].some((x,i)=>!Number.isFinite(x)||Math.abs(x-expected[i])>1e-8))throw Error('Reference camera does not match native source');
}
const p=await m.scenePackage(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'ply',c,context);
const files=Object.fromEntries((await m.unpackSceneZip(p.bytes)).map(x=>[x.name,x.data]));
files['README.txt']=new TextDecoder().decode(files['README.txt'])+`\n跨人物SH0试验：${subject}，${step}步。标准PLY固定${v.reference_frame}姿态，仅配同姿态相机；其他拟合帧含不同身体姿态，不能直接当作该静态点云的相机。${plan.training_frames}图RGB训练；质量尚未验收。训练checkpoint、绑定和逐帧mesh保留在工程，PLY不含可驱动骨架。\n`;
if(plan.supervision_scope)files['README.txt']+=`\n监督范围：${plan.supervision_scope}\n被遮挡、未拍到的腿脚没有被可靠恢复；保守有效像素规则也会忽略部分真实背景。\n`;
if(['indoor-zoom-clock18k124','indoor-paired-background124','indoor-paired-background18k124','indoor-circle16-paired18k124','sofa-half8-paired18k124','winterhat-half8-paired18k124','football-half12-paired18k126'].includes(subject)){
  const root=path.resolve(dir,'../..');
  const maps=JSON.parse(await fs.readFile(root+'/pixel-mapping.json','utf8')).records;
  const rows=JSON.parse(await fs.readFile(root+'/data/training-inputs.json','utf8')).records;
  const mapping=maps.find(x=>x.frame===v.reference_frame),row=rows.find(x=>x.frame===v.reference_frame);
  if(!mapping||!row||c.length!==1||c[0].width!==row.wh[0]||c[0].height!==row.wh[1])throw Error('Derived camera size mismatch');
  const values=[c[0].fx,c[0].fy,c[0].cx,c[0].cy],expected=[row.K[0][0],row.K[1][1],row.K[0][2],row.K[1][2]];
  if(c[0].pixelConvention!=='pixel_corners'||values.some((x,i)=>Math.abs(x-expected[i])>1e-8))throw Error('Derived full K mismatch');
  files['camera_input_mapping.json']=JSON.stringify({referenceFrameId:v.reference_frame,
    sourceSize:mapping.source.wh,sourceKIntegerCenters:mapping.source.K_integer_centers,
    cropXYXY:mapping.crop_xyxy,scale:mapping.scale,integerCenterTransform:mapping.integer_center_transform,
    exportedSize:row.wh,exportedKPixelCorners:row.K,
    note:'Provenance only. cameras.json describes the enlarged crop, not the original whole video frame. No second undistortion or physical calibration acceptance.'},null,2);
  files['README.txt']+='\n相机对应人物裁片放大2倍后的画布，不是原视频整图。完整主点已同步变换；camera_input_mapping.json记录原图到裁片的像素变换，仅供追溯。不要把这份相机直接覆盖原视频整图K。\n';
}
files['validation.json']=JSON.stringify(v,null,2);
if(diagnosedPreview){
  for(const n of ['source-roundoff-diagnosis.json','rigid-roundoff-diagnosis.json'])files[n]=JSON.stringify(await read(n),null,2);
  files['README.txt']+='\n本包仅作诊断预览。原检查点→标准PLY以及+Y旋转回读的严格最大图像误差检查仍为失败：5195040像素中1像素最大RGB差0.006273。已独立核对中心、颜色、透明度、完整协方差和相机，并定位到float32尺度/四元数读回触发原光栅器alpha=1/255截断。未修改模型、光栅器或原验收阈值；不是逐像素完全一致或重建质量通过。详情见validation及roundoff诊断。\n';
}
const zip=f.storedZip(files),entries=await m.unpackSceneZip(zip);
const objects=entries.map(x=>{const q=new File([x.data],path.basename(x.name));q.sceneRelativePath=x.name;return q;});
const models=await m.primaryModelFiles(objects),paired=await m.pairedCameras(models[0],objects,models);
if(models.length!==1||paired.cameras.length!==1||!paired.verifiedBinding||paired.referenceFrameId!==v.reference_frame)throw Error('Camera pairing failed');
await fs.writeFile(dir+'/scene.zip',zip,{flag:'wx'});
await fs.writeFile(dir+'/package-result.json',JSON.stringify({updated:new Date().toISOString(),passed:true,points:v.points,bytes:zip.length,sha256:await m.sha256(zip),cameraCount:paired.cameras.length,verifiedBinding:paired.verifiedBinding,referenceFrameId:paired.referenceFrameId,shDegree:0,sibrGuiVerified:false},null,2),{flag:'wx'});
console.log('CROSS_SUBJECT_PACKAGE_VERIFIED',subject,zip.length);
