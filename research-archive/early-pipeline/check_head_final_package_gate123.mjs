import fs from 'node:fs/promises';
const file='C:/Users/Administrator/Documents/New project/shuolee-slide-update/package_demo_head_gaussians123.mjs';
const source=await fs.readFile(file,'utf8');
const start=source.indexOf('if(validation.format_passed');const end=source.indexOf('const p=await m.scenePackage',start);
if(start<0||end<0)throw Error('Release gate not found');
const AsyncFunction=Object.getPrototypeOf(async function(){}).constructor;
const check=new AsyncFunction('validation','m','bytes','cameraBytes',source.slice(start,end));
const ply=Buffer.from('ply'),cam=Buffer.from('camera');const m={sha256:async value=>value===ply?'PLY_SHA':'CAMERA_SHA'};
const valid={format_passed:true,native_rotated_render_parity:{passed:true},native_source_ply_parity:{passed:true},sha256:'PLY_SHA',native_replay_uses_serialized_camera:true,serialized_camera_sha256:'CAMERA_SHA'};
const cases=[];
for(const [name,change,accept] of [['valid_complete',{},true],['camera_not_replayed',{native_replay_uses_serialized_camera:false},false],['camera_changed_after_replay',{serialized_camera_sha256:'CHANGED'},false],['rotated_replay_pending',{native_rotated_render_parity:'pending'},false],['PLY_changed_after_replay',{sha256:'CHANGED'},false]]){
 let accepted=true;try{await check({...valid,...change},m,ply,cam);}catch{accepted=false;}
 if(accepted!==accept)throw Error('Wrong gate outcome '+name);cases.push({name,accepted,passed:true});
}
await fs.writeFile('C:/Users/Administrator/Documents/New project/shuolee-slide-update/head-final-package-camera-gate-check123.json',JSON.stringify({updated:new Date().toISOString(),tests:cases,passed:cases.length,new_package_written:false,GPU_started:false},null,2));
console.log('ACTUAL_PACKAGE_CAMERA_GATE_CASES_PASSED',cases.length);