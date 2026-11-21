import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const T='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001';
const R=T+'/shuolee/current/robust-portrait-122';
const source=R+'/native-cvo-control/exact-native-package';
const {scenePackage}=await import(pathToFileURL(T+'/viewer/scene-package.js'));
const {previewRendererContract}=await import(pathToFileURL(T+'/viewer/render-contract.js'));
const native=JSON.parse(await fs.readFile(source+'/cameras.json','utf8'));
const cameras=native.map(c=>({...c,img_name:c.image_name,pixelConvention:'pixel_corners',synthetic:true,projectionModel:'pinhole'}));
const b=await fs.readFile(source+'/point_cloud.ply');
const packed=await scenePackage(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),'ply',cameras,{
 label:'T9 synthetic exact-native companion preview',steps:0,
 rendererContract:{...previewRendererContract(),trainingRenderer:'gsplat1.5.3 UT geometry bounds and ray3d',nativeImageEquivalence:'known-not-equivalent'},
 world:'Original synthetic world; model and recorded cameras unchanged; not calibrated human data'});
const out=R+'/native-ui-v3';await fs.mkdir(out,{recursive:false});await fs.mkdir(out+'/packages');
await fs.writeFile(out+'/packages/T9_standard_preview.zip',packed.bytes);
await fs.writeFile(out+'/preview-generation.json',JSON.stringify({model_sha256:packed.hash,cameras:packed.cameraCount,synthetic:true,native_equivalent:false,source_module:T+'/viewer/scene-package.js'},null,2)+'\n');
console.log(JSON.stringify({bytes:packed.bytes.length,cameras:packed.cameraCount}));
