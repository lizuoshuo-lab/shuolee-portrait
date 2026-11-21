// A portable model + standard cameras sidecar package. Model bytes stay intact.
import {normalizeCameras} from './camera-io.js?v=20261027-5';
import {assertPreviewCompatible,previewRendererContract} from './render-contract.js?v=20261027-5';
import {cameras3dgs,colmapCameraFiles,storedZip,crc32} from './camera-formats.js?v=20261027-5';
import {inspectPly} from './point-cloud-loader.js';
import {sparkCameraPage} from './spark-camera-template.js?v=20261027-5';
const enc=new TextEncoder(),dec=new TextDecoder();
function previewFilter(value){
  if(value==null)return undefined;
  const filter={preBlurAmount:value.preBlurAmount??.3,blurAmount:value.blurAmount??0};
  if(Object.values(filter).some(x=>typeof x!=='number'||!Number.isFinite(x)||x<0))throw Error('预览滤波参数必须为非负有限数');
  return filter;
}
export const sha256=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
function referencePose(context,cameras){
  if(context.referenceFrameId==null){
    if(context.referenceFrameIsPoseSnapshot)throw Error('静态姿态点云缺少参考帧');
    return {};
  }
  const ids=cameras.map(c=>c.img_name),reference=context.referenceFrameId,index=ids.indexOf(reference);
  if(typeof reference!=='string'||index<0||ids.filter(id=>id===reference).length!==1)throw Error('点云参考帧与配套相机不唯一对应');
  if(context.initialView!=null&&context.initialView!==index)throw Error('默认相机与点云参考帧不一致');
  return {referenceFrameId:reference,referenceFrameIsPoseSnapshot:true,initialView:index};
}
export async function scenePackage(bytes,extension,cameras,context={}){
  assertPreviewCompatible(context);
  if(!/^(ply|splat|spz|ksplat|sog|pcd|xyz|pts)$/.test(extension))throw Error('不支持的模型后缀');
  if(!cameras.length)throw Error('这份模型没有配套相机，请先加载相机文件');
  const standardPly=extension==='ply'&&!dec.decode(new Uint8Array(bytes).subarray(0,65536)).includes('packed_position')&&inspectPly(bytes).kind==='gaussian';
  const step=Number.isSafeInteger(context.steps)&&context.steps>=0?context.steps:0;
  const modelFile=standardPly?'point_cloud/iteration_'+step+'/point_cloud.ply':'model.'+extension,hash=await sha256(bytes);
  if(context.model?.sha256&&context.model.sha256!==hash)throw Error('点云哈希与当前相机绑定不符');
  context={...context,model:{...context.model,name:modelFile,sha256:hash}};
  const standardCameras=cameras3dgs(cameras),pose=referencePose(context,standardCameras);
  const standard=JSON.stringify(standardCameras,null,2)+'\n';
  const files=colmapCameraFiles(cameras,{...context,selected:pose.initialView??context.selected??0});
  files[modelFile]=new Uint8Array(bytes);files['cameras.json']=standard;
  const auxiliaryModels=[];
  if(standardPly){
    const header=dec.decode(new Uint8Array(bytes).subarray(0,65536)).split('end_header')[0];
    const rest=(header.match(/property float f_rest_\d+/g)||[]).length;
    const degree=Math.sqrt(rest/3+1)-1;
    if(!Number.isInteger(degree)||degree<0||degree>3)throw Error('SIBR 包需要标准 0–3 阶 SH 字段，当前字段不完整');
    files['cfg_args']="Namespace(sh_degree="+degree+", source_path='.', model_path='.', images='images', resolution=-1, white_background=False, data_device='cuda', eval=False)\n";
    const info=inspectPly(bytes),props=header.match(/^property .+$/gm)||[];
    const expected=['x','y','z','nx','ny','nz','f_dc_0','f_dc_1','f_dc_2',...Array.from({length:rest},(_,i)=>'f_rest_'+i),'opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];
    if(info.format!=='binary_little_endian'||JSON.stringify(info.properties)!==JSON.stringify(expected)||props.some(p=>!/^property float /.test(p)))throw Error('官方 SIBR 需要标准 binary_little_endian / float 字段顺序；可先仅下载原格式用于 Spark');
    const marker=dec.decode(new Uint8Array(bytes).subarray(0,65536)).match(/end_header\r?\n/),offset=marker.index+marker[0].length,stride=info.properties.length*4;
    if(offset+info.count*stride!==bytes.byteLength)throw Error('标准 PLY 长度不匹配');
    const h=enc.encode('ply\nformat binary_little_endian 1.0\ncomment Gaussian centers for SIBR preview; not SfM initialization\nelement vertex '+info.count+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n');
    const proxy=new Uint8Array(h.length+info.count*15);proxy.set(h);const pv=new DataView(proxy.buffer),source=new DataView(bytes);
    for(let i=0;i<info.count;i++){const at=h.length+i*15;for(let k=0;k<3;k++){const x=source.getFloat32(offset+i*stride+k*4,true),rgb=.5+.28209479177387814*source.getFloat32(offset+i*stride+(6+k)*4,true);if(!Number.isFinite(x)||!Number.isFinite(rgb))throw Error('点云含非有限中心或颜色');pv.setFloat32(at+4*k,x,true);proxy[at+12+k]=Math.round(Math.min(1,Math.max(0,rgb))*255);}}
    files['input.ply']=proxy;auxiliaryModels.push('input.ply');
  }
  files['scene.json']=JSON.stringify({schema:'shuolee-scene-v1',label:context.label||null,steps:context.steps??null,...pose,filter:previewFilter(context.filter),model:{file:modelFile,sha256:hash},
    cameras:{file:'cameras.json',sha256:await sha256(enc.encode(standard)),provenance:'camera_provenance.json',count:cameras.length},
    auxiliaryModels,rendererContract:context.rendererContract||previewRendererContract(),compatibility:{sibrModelLayout:standardPly,sibrUiVerified:false,sparkSdk:true,sparkHostedDemoAutoCameras:false,nativeRenderingEquivalent:false},
    world:'model-and-cameras-share-original-world-no-transform',qualityValidated:false},null,2)+'\n';
  files['spark-camera.html']=sparkCameraPage();
  files['README.txt']='点云与相机配套包 / Model and camera bundle\n\n1. 当前 Shuolee 查看器：直接打开本 ZIP，或解压后导入整个文件夹。自动读取相机并检查模型、相机 SHA256。仅选 PLY 时浏览器没有权限自动读取旁边文件，请同时选配套 JSON，或选整个文件夹。\n\n2. 3DGS 官方 SIBR：'+(standardPly?'在包目录运行 SIBR_gaussianViewer_app -m . -s .\n模型为 '+modelFile+'，根目录含 cfg_args、cameras.json 和 input.ply。input.ply 仅为高斯中心显示代理，不是 SfM 初始化点。未附实拍图像，不使用 --load_images。':'此文件不是标准 3DGS PLY；官方 SIBR 不支持直接加载此格式。')+'\n官方旧版 SIBR 相机 JSON 读取器使用 fx/fy 与中心主点，不读取额外 cx/cy。本包仍完整保存真实主点，不能把其默认画面作为严格像素投影验收。\n\n3. Spark 官网 https://sparkjs.dev/viewer/ ：选择 '+modelFile+' 可加载高斯；当前官网示例只读取一个点云，没有 camera JSON 导入，而且默认绕 X 轴翻转180度。本包不通过改动点云去抵消第三方显示变换。\n需要完整相机：解压后打开 spark-camera.html，再点“打开点云和相机”，同时选择上述点云、cameras.json、scene.json。该页使用 Spark 官方 SDK、完整 fx/fy/cx/cy 和 OpenCV→Three.js 相机转换，不翻转模型；首次联网只加载 SDK，点云在本机读取。也可用当前 Shuolee 查看器离线读取本 ZIP。\n\n4. COLMAP：根目录 cameras.txt/images.txt/points3D.txt 为 PINHOLE 相机模型（完整 fx/fy/cx/cy、w2c Hamilton wxyz）。无虚构稀疏点或特征轨迹。\n\n相机和点云使用相同世界坐标。未旋转、缩放或重新优化点云；高斯球显示尺寸不写回文件。格式校验不证明重影已修复。\n';
  files['README.txt']+='\n渲染范围：本包为针孔 / 屏幕空间高斯预览。标准PLY字段兼容不证明原生求值一致；鱼眼投影、3DGUT三维响应或自定义畸变场须使用对应原生渲染器。没有渲染来源的旧PLY仅供预览，不能用于原生像素精度验收。\n';
  if(pose.referenceFrameId)files['README.txt']+='\n静态点云姿态固定于 '+pose.referenceFrameId+'，默认使用其对应相机。其他实拍帧的人体姿态可能不同，不能将其原图作为此静态姿态的逐像素真值。\n';
  return {bytes:storedZip(files),hash,cameraCount:cameras.length};
}
export async function unpackSceneZip(buffer,{maxBytes=1024**3}={}){
  const a=new Uint8Array(buffer),v=new DataView(a.buffer,a.byteOffset,a.byteLength);
  if(a.length<22||a.length>maxBytes)throw Error('ZIP大小无效或超过1GiB');
  let e=-1;for(let i=a.length-22;i>=Math.max(0,a.length-65557);i--)if(v.getUint32(i,true)===0x06054b50){e=i;break;}
  if(e<0)throw Error('ZIP目录缺失');
  if(v.getUint16(e+4,true)||v.getUint16(e+6,true))throw Error('不支持分卷ZIP');
  const n=v.getUint16(e+10,true),cd=v.getUint32(e+16,true);let at=cd,total=0;const out=[];const names=new Set();
  if(n<1||n>128||v.getUint16(e+8,true)!==n||cd>=e)throw Error('ZIP目录无效或文件数过多');
  for(let i=0;i<n;i++){
    if(at+46>e||v.getUint32(at,true)!==0x02014b50)throw Error('ZIP目录损坏');
    const flags=v.getUint16(at+8,true),method=v.getUint16(at+10,true),crc=v.getUint32(at+16,true),packed=v.getUint32(at+20,true),size=v.getUint32(at+24,true),nl=v.getUint16(at+28,true),extra=v.getUint16(at+30,true),comment=v.getUint16(at+32,true),off=v.getUint32(at+42,true);
    if(at+46+nl+extra+comment>e)throw Error('ZIP目录越界');
    const name=dec.decode(a.subarray(at+46,at+46+nl));at+=46+nl+extra+comment;
    if(flags&1)throw Error('不支持加密ZIP');
    if(!name||name.includes('\\')||name.startsWith('/')||name.includes(':')||name.split('/').some(x=>x==='..'||x==='.')||/[\x00-\x1f]/.test(name))throw Error('ZIP含非法路径');
    if(names.has(name))throw Error('ZIP含重复文件名');names.add(name);
    if(name.endsWith('/'))continue;
    total+=size;if(total>maxBytes||size===0xffffffff||off+30>cd||![0,8].includes(method))throw Error('ZIP过大或压缩方式不支持');
    if(v.getUint32(off,true)!==0x04034b50)throw Error('ZIP文件头损坏');
    const localNameLength=v.getUint16(off+26,true),localExtraLength=v.getUint16(off+28,true),start=off+30+localNameLength+localExtraLength;
    if(start+packed>cd||dec.decode(a.subarray(off+30,off+30+localNameLength))!==name||v.getUint16(off+8,true)!==method)throw Error('ZIP文件记录不一致');
    let data=a.slice(start,start+packed);
    if(method===8){
      if(typeof DecompressionStream==='undefined')throw Error('浏览器无法解压；请先解压ZIP再导入文件夹');
      const reader=new Blob([data]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader();const chunks=[];let length=0;
      while(true){const r=await reader.read();if(r.done)break;length+=r.value.length;if(length>size||length>maxBytes){await reader.cancel();throw Error('ZIP展开大小不符');}chunks.push(r.value);}
      data=new Uint8Array(length);let p=0;for(const c of chunks){data.set(c,p);p+=c.length;}
    }
    if(data.length!==size||crc32(data)!==crc)throw Error('ZIP文件校验失败：'+name);
    out.push({name,data});
  }
  if(at!==cd+v.getUint32(e+12,true))throw Error('ZIP目录长度不符');
  return out;
}
export const localFilePath=f=>(f.sceneRelativePath||f.webkitRelativePath||f.name).replaceAll('\\','/');
const basename=p=>p.slice(p.lastIndexOf('/')+1),folder=p=>p.slice(0,p.lastIndexOf('/')+1);
async function jsonFile(file){if(file.size>20*1024**2)throw Error('相机/清单超过20MB');return JSON.parse((await file.text()).replace(/^\uFEFF/,''));}
export async function primaryModelFiles(files){
  const supported=files.filter(f=>/\.(ply|splat|spz|ksplat|sog|pcd|xyz|pts)$/i.test(f.name));
  const auxiliary=new Set();
  for(const f of files.filter(f=>basename(localFilePath(f))==='scene.json')){
    const m=await jsonFile(f);if(m.schema!=='shuolee-scene-v1')throw Error('配套scene.json版本不支持');
    assertPreviewCompatible(m);
    const dir=folder(localFilePath(f));
    if(typeof m.model?.file!=='string'||!supported.some(x=>localFilePath(x)===dir+m.model.file))throw Error('scene.json 指定的点云缺失或文件名不匹配');
    for(const n of m.auxiliaryModels||[]){if(n===m.model.file)throw Error('主模型不能标记为代理');auxiliary.add(dir+n);}
  }
  return supported.filter(f=>!auxiliary.has(localFilePath(f)));
}
export async function pairedCameras(model,files,modelFiles){
  const path=localFilePath(model),stem=basename(path).replace(/\.[^.]+$/,'');let dir=folder(path),manifest=null;
  for(const f of files.filter(f=>basename(localFilePath(f))==='scene.json')){
    const m=await jsonFile(f),base=folder(localFilePath(f));
    if(m.schema==='shuolee-scene-v1'&&base+m.model?.file===path){assertPreviewCompatible(m);manifest=m;dir=base;break;}
  }
  const siblings=files.filter(f=>folder(localFilePath(f))===dir);
  const named=n=>siblings.find(f=>basename(localFilePath(f))===n);
  let camFile=null,provenance=null;
  if(manifest){
    camFile=named(manifest.cameras?.file);provenance=named(manifest.cameras?.provenance);
    if(!camFile||!provenance)throw Error('配套相机文件缺失，拒绝静默加载不完整包');
  }else{
    camFile=named(stem+'.cameras.json');
    if(!camFile&&modelFiles.filter(f=>folder(localFilePath(f))===dir).length===1)camFile=named('camera_provenance.json')||named('cameras.json');
    if(!camFile)return null;
  }
  const hash=await sha256(await model.arrayBuffer());
  if(manifest&&manifest.model.sha256!==hash)throw Error('配套点云SHA256不匹配');
  if(manifest&&manifest.cameras.sha256!==await sha256(await camFile.arrayBuffer()))throw Error('配套相机SHA256不匹配');
  const parsed=normalizeCameras(await jsonFile(camFile),{expectedSha256:hash});
  if(provenance){
    const p=normalizeCameras(await jsonFile(provenance),{expectedSha256:hash});
    if(p.sourceModelHash!==hash||p.cameras.length!==parsed.cameras.length||manifest.cameras.count!==p.cameras.length)throw Error('配套相机身份不一致');
    for(let i=0;i<p.cameras.length;i++){
      const a=p.cameras[i],b=parsed.cameras[i];
      if(a.img_name!==b.img_name||a.width!==b.width||a.height!==b.height||['fx','fy','cx','cy'].some(k=>Math.abs(a[k]-b[k])>1e-8)||a.position.some((x,k)=>Math.abs(x-b.position[k])>1e-8)||a.rotation.flat().some((x,k)=>Math.abs(x-b.rotation.flat()[k])>1e-8))throw Error('标准相机和来源相机不一致');
    }
    return {...p,hash,verifiedBinding:true,fileName:camFile.name,label:manifest.label,steps:manifest.steps,filter:previewFilter(manifest.filter),...referencePose(manifest,p.cameras)};
  }
  return {...parsed,hash,verifiedBinding:parsed.sourceModelHash===hash,fileName:camFile.name};
}
