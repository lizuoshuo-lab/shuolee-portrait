// Camera exports; no implicit model/world rotation or intrinsic recentering.
import { normalizeCameras, cameraBundle } from './camera-io.js';
const rowsOf = cameras => normalizeCameras({pixelConvention:'pixel_corners',cameras}).cameras;
export function cameras3dgs(cameras){
  return rowsOf(cameras).map((c,i)=>({id:i,img_name:c.img_name,width:c.width,height:c.height,
    position:c.position,rotation:c.rotation,fy:c.fy,fx:c.fx,cx:c.cx,cy:c.cy,
    pixelConvention:'pixel_corners',projectionModel:c.projectionModel,...(c.synthetic?{synthetic:true}:{})}));
}
function quaternion(r){
  const t=r[0][0]+r[1][1]+r[2][2];let q;
  if(t>0){const s=Math.sqrt(t+1)*2;q=[s/4,(r[2][1]-r[1][2])/s,(r[0][2]-r[2][0])/s,(r[1][0]-r[0][1])/s];}
  else if(r[0][0]>r[1][1]&&r[0][0]>r[2][2]){const s=Math.sqrt(1+r[0][0]-r[1][1]-r[2][2])*2;q=[(r[2][1]-r[1][2])/s,s/4,(r[0][1]+r[1][0])/s,(r[0][2]+r[2][0])/s];}
  else if(r[1][1]>r[2][2]){const s=Math.sqrt(1+r[1][1]-r[0][0]-r[2][2])*2;q=[(r[0][2]-r[2][0])/s,(r[0][1]+r[1][0])/s,s/4,(r[1][2]+r[2][1])/s];}
  else{const s=Math.sqrt(1+r[2][2]-r[0][0]-r[1][1])*2;q=[(r[1][0]-r[0][1])/s,(r[0][2]+r[2][0])/s,(r[1][2]+r[2][1])/s,s/4];}
  const n=Math.hypot(...q),sign=q[0]<0?-1:1;return q.map(v=>v/n*sign);
}
export function colmapCameraFiles(cameras,context={}){
  const rows=rowsOf(cameras),intrinsics=new Map(),cs=[],ims=[],names=new Set(),mapping=[];
  for(let i=0;i<rows.length;i++){
    const c=rows[i],key=JSON.stringify([c.width,c.height,c.fx,c.fy,c.cx,c.cy]);
    if(!intrinsics.has(key)){
      const id=intrinsics.size+1;intrinsics.set(key,id);
      cs.push([id,'PINHOLE',c.width,c.height,c.fx,c.fy,c.cx,c.cy].join(' '));
    }
    const r=c.rotation[0].map((_,j)=>c.rotation.map(row=>row[j]));
    const t=r.map(row=>-row.reduce((s,v,j)=>s+v*c.position[j],0));
    let name=c.img_name.replace(/[\s\\/\x00-\x1f]+/g,'_');
    if(!/\.[A-Za-z0-9]{2,5}$/.test(name))name+='.png';
    if(names.has(name))name=String(i+1)+'_'+name;names.add(name);
    ims.push([i+1,...quaternion(r),...t,intrinsics.get(key),name].join(' ')+'\n');
    mapping.push({image_id:i+1,name,source_img_name:c.img_name,synthetic:c.synthetic});
  }
  const meta={...cameraBundle(rows,context),colmapImageMapping:mapping};
  return {
    'cameras.txt':'# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n# PINHOLE: fx fy cx cy; top-left image-edge pixels\n'+cs.join('\n')+'\n',
    'images.txt':'# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# Each pose line is followed by an empty POINTS2D line. World-to-camera, Hamilton wxyz.\n'+ims.join('\n')+'\n',
    'points3D.txt':'# Camera-only export: no fabricated 3D points or observation tracks.\n',
    'camera_provenance.json':JSON.stringify(meta,null,2)+'\n',
    'README.txt':'COLMAP camera-only text model. Import cameras.txt/images.txt/points3D.txt together.\nPINHOLE parameters retain fx/fy/cx/cy. Pose is world-to-camera; world coordinates are unchanged.\nImages and 3D scene points are not included. Renamed image paths are listed in camera_provenance.json.\nThis is format conversion, not camera estimation or calibration validation. Synthetic views are marked in provenance.\n3DGS cameras.json has no universal principal-point support; use this PINHOLE model in tools that honor cx/cy.\n'
  };
}
// A small standards-compliant ZIP with stored entries; no runtime download.
const crcTable=Uint32Array.from({length:256},(_,i)=>{let c=i;for(let k=0;k<8;k++)c=(c>>>1)^((c&1)?0xedb88320:0);return c>>>0;});
export function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=(c>>>8)^crcTable[(c^b)&255];return (c^0xffffffff)>>>0;}
function fields(size,write){const a=new Uint8Array(size);write(new DataView(a.buffer));return a;}
export function storedZip(files){
  const parts=[],central=[],enc=new TextEncoder();let offset=0;
  for(const [name,value] of Object.entries(files)){
    if(!/^[A-Za-z0-9_./-]+$/.test(name)||name.split('/').some(x=>!x||x==='.'||x==='..'))throw Error('ZIP entry name must be a safe relative path');
    const n=enc.encode(name),data=value instanceof Uint8Array?value:value instanceof ArrayBuffer?new Uint8Array(value):enc.encode(value),crc=crc32(data),start=offset;
    const h=fields(30,v=>{v.setUint32(0,0x04034b50,true);v.setUint16(4,20,true);v.setUint16(6,0x800,true);v.setUint16(12,33,true);v.setUint32(14,crc,true);v.setUint32(18,data.length,true);v.setUint32(22,data.length,true);v.setUint16(26,n.length,true);});
    parts.push(h,n,data);offset+=h.length+n.length+data.length;
    const ch=fields(46,v=>{v.setUint32(0,0x02014b50,true);v.setUint16(4,20,true);v.setUint16(6,20,true);v.setUint16(8,0x800,true);v.setUint16(14,33,true);v.setUint32(16,crc,true);v.setUint32(20,data.length,true);v.setUint32(24,data.length,true);v.setUint16(28,n.length,true);v.setUint32(42,start,true);});central.push(ch,n);
  }
  const csize=central.reduce((s,a)=>s+a.length,0),count=Object.keys(files).length;
  const end=fields(22,v=>{v.setUint32(0,0x06054b50,true);v.setUint16(8,count,true);v.setUint16(10,count,true);v.setUint32(12,csize,true);v.setUint32(16,offset,true);});
  const out=new Uint8Array(offset+csize+22);let p=0;for(const part of [...parts,...central,end]){out.set(part,p);p+=part.length;}return out;
}
