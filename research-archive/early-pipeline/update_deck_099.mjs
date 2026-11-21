import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/shared-point-ray-consistency-099',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='ac1eb51815d93466090f3d6542bba50e3c4d7ed42ca6d56ceff0d1007b6a3190';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),prior=await read(build+'/human098-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('静态人物重建：多视角共享三维特征','静态人物重建：共享特征与相机联合优化');
replace('同一鼻尖或嘴角在全部可见相机中','每个同名五官特征只有一个三维点，与整图特征共同约束相机');
replace('同相机照片、原方案、共享点约束','留出五官投影误差中位数：5.28 降至 3.88 像素，正面部分偏差缩小。');
replace('共享点使落点散布降低','部分侧视角与人物区域仍退步，求解尚未收敛，暂不替换训练相机。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Unexpected image layout');p.delete(images.map(e=>e.id));
p.resolve(slide.id).images.add({blob:await fs.readFile(run+'/held-face-000062.jpg'),contentType:'image/jpeg',
 position:{left:66,top:178,width:1150,height:380},fit:'contain',alt:'第62帧真实照片、整图纹理相机优化和加入共享五官三维点的留出投影对照'});
p.resolve(slide.id).speakerNotes.text=d.presentation_notes.join('\n\n');
const candidatePath=build+'/human099-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_共享特征联合优化.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/human099-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human099-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human099-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
