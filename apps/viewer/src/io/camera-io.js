// Camera I/O uses OpenCV camera axes and the SAME world as the loaded PLY.
// Pixel coordinates in the viewer are measured from the top-left image edge.
import {assertPinholeCamera,assertPreviewCompatible} from './render-contract.js';
export const CAMERA_SCHEMA = 'shuolee-cameras-v1';
const finite = x => typeof x === 'number' && Number.isFinite(x);
const vector = (a,n,name) => {
  if(!Array.isArray(a)||a.length!==n||!a.every(finite))throw Error(name+' 必须是有限数值数组');
  return a.slice();
};
function matrix(a,n,name){
  if(!Array.isArray(a)||a.length!==n)throw Error(name+' 矩阵维度不符');
  return a.map(r=>vector(r,n,name));
}
export function validateRotation(a){
  const r=matrix(a,3,'rotation');
  let e=0;
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)e=Math.max(e,Math.abs(r[i].reduce((s,v,k)=>s+v*r[j][k],0)-(i===j?1:0)));
  const det=r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])-r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])+r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]);
  if(e>1e-4||Math.abs(det-1)>1e-4)throw Error('rotation 含缩放、反射或非正交变换');
  return r;
}
function rigid(a,name){
  const m=matrix(a,4,name);
  if(m[3].some((x,i)=>Math.abs(x-(i===3?1:0))>1e-9))throw Error(name+' 最后一行必须为 [0,0,0,1]');
  validateRotation(m.slice(0,3).map(r=>r.slice(0,3)));return m;
}
function pinholeOnly(c){
  assertPinholeCamera(c);
  assertPreviewCompatible(c);
}
export function normalizeCameras(data,{expectedSha256=null}={}){
  if(!data||typeof data!=='object')throw Error('相机 JSON 无效');
  pinholeOnly(data);
  const rows=Array.isArray(data)?data:(data.cameras||data.records);
  if(!Array.isArray(rows)||!rows.length||rows.length>10000)throw Error('需要 1–10000 个 cameras / records');
  const declaredHash=data.model?.sha256||data.world_to_viewer?.sha256;
  if(declaredHash&&expectedSha256&&declaredHash.toLowerCase()!==expectedSha256.toLowerCase())throw Error('相机绑定的模型 SHA256 与当前 PLY 不符');
  if(data.schema&&data.schema!==CAMERA_SCHEMA)throw Error('不支持的相机 schema：'+data.schema);
  if(data.coordinateConvention&&data.coordinateConvention!=='opencv-camera-to-world')throw Error('相机坐标约定不支持；请转换为 OpenCV 相机轴');
  const notes=[],ids=new Set();
  const pixel=data.pixelConvention||(String(data.convention||'').includes('COLMAP pixel corners')?'pixel_corners':null);
  if(pixel&&!['pixel_corners','integer_centers'].includes(pixel))throw Error('未知像素中心约定');
  if(!pixel&&rows.some(c=>!c.pixelConvention))notes.push('未声明像素约定的 3DGS 相机：按图像左上边缘像素坐标读取');
  const cameras=rows.map((c,i)=>{
    if(!c||typeof c!=='object')throw Error('相机 '+i+' 无效');
    pinholeOnly(c);
    if(c.projectionModel==='pinhole_export_assumption')notes.push('针孔导出假设相机：物理标定仍未验证');
    const width=c.width??c.wh?.[0],height=c.height??c.wh?.[1];
    if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||width<=0||height<=0||Math.max(width,height)>100000)throw Error('相机 '+i+' 图像尺寸无效');
    const k=c.K?matrix(c.K,3,'K'):null;
    if(k&&(Math.abs(k[0][1])+Math.abs(k[1][0])+Math.abs(k[2][0])+Math.abs(k[2][1])+Math.abs(k[2][2]-1)>1e-9))throw Error('目前仅支持零 skew 的去畸变针孔 K');
    const fx=k?k[0][0]:c.fx,fy=k?k[1][1]:c.fy;
    if(!finite(fx)||!finite(fy)||Math.min(fx,fy)<=0)throw Error('相机 '+i+' 缺少有效 fx/fy');
    const pc=c.pixelConvention||pixel||'pixel_corners';
    if(!['pixel_corners','integer_centers'].includes(pc))throw Error('相机 '+i+' 未知像素中心约定');
    const shift=pc==='integer_centers'?.5:0;
    const rawcx=k?k[0][2]:c.cx,rawcy=k?k[1][2]:c.cy;
    if(rawcx==null||rawcy==null)notes.push('相机 '+i+' 缺少主点，旧格式采用图像中心');
    const cx=rawcx==null?width/2:rawcx+shift,cy=rawcy==null?height/2:rawcy+shift;
    if(!finite(cx)||!finite(cy))throw Error('相机 '+i+' 主点无效');
    if([!!c.w2c,!!c.c2w,!!c.rotation].filter(Boolean).length!==1)throw Error('每个相机只能提供 w2c、c2w 或 rotation/position 之一');
    let rotation,position;
    if(c.w2c){
      const e=rigid(c.w2c,'w2c'),r=e.slice(0,3).map(x=>x.slice(0,3)),t=e.slice(0,3).map(x=>x[3]);
      rotation=r[0].map((_,j)=>r.map(x=>x[j]));position=rotation.map(row=>-row.reduce((s,x,j)=>s+x*t[j],0));
    }else if(c.c2w){const m=rigid(c.c2w,'c2w');rotation=m.slice(0,3).map(x=>x.slice(0,3));position=m.slice(0,3).map(x=>x[3]);}
    else{rotation=validateRotation(c.rotation);position=vector(c.position,3,'position');}
    const name=String(c.img_name??c.frame??c.id??i);
    if(ids.has(name))throw Error('重复相机/帧身份：'+name);ids.add(name);
    return {id:c.id??i,img_name:name,width,height,fx,fy,cx,cy,rotation,position,
      ...(c.label?{label:String(c.label)}:{}),raw_source_frame:c.raw_source_frame??c.source_frame_index??null,
      time_seconds:c.time_seconds??null,projectionModel:c.projectionModel||'pinhole',
      synthetic:c.synthetic===true,split:c.split||'imported_not_quality_validated'};
  });
  return {cameras,notes:[...new Set(notes)].slice(0,4),world:data.world||data.world_to_viewer?.coordinate_space||'当前模型原坐标（不自动旋转、缩放或配准）',sourceModelHash:declaredHash||null};
}
export function cameraBundle(cameras,{model=null,world=null,selected=0}={}){
  const normalized=normalizeCameras({schema:CAMERA_SCHEMA,pixelConvention:'pixel_corners',cameras}).cameras;
  return {schema:CAMERA_SCHEMA,coordinateConvention:'opencv-camera-to-world',pixelConvention:'pixel_corners',
    world:world||'Same world coordinates as the loaded model; no implicit transform',model,selected,
    cameras:normalized,qualityValidated:false};
}
