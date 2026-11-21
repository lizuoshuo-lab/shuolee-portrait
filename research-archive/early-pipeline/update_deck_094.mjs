import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/whole-body-alignment-audit-094',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='d3349ca3f9712ebb41f591cd692dcb9c25082b08559a63b8b9893e45273602c2';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),r=await read(run+'/result.json'),prior=await read(build+'/human093-deck-receipt.json');
if(d.actual_review_images.length!==5||r.optimizer_updates!==0)throw Error('Whole-body audit review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('静态人物重建：重影的形成与约束','静态人物重建：全身投影与边界一致性');
replace('175帧整图相机与人体初始化保持一致','175帧全身检查：肩袖、躯干与腿脚均存在对应误差和轮廓外溢');
replace('同一实拍相机：左为照片','当前模型的同相机全身轮廓：左第70帧，右第100帧。青色为原mask，红色为模型轮廓。');
replace('关闭NCC减弱局部重影','躯干、腿脚的表面投影仍不一致。继续追踪其与白边、重影的共同成因。');
const images=slide.elements.filter(e=>e.kind==='image');
if(images.length!==1)throw Error('Unexpected image layout');
p.delete(images.map(e=>e.id));
for(const [n,raw] of [70,100].entries())p.resolve(slide.id).images.add({blob:await fs.readFile(run+'/no_ncc-body-'+String(raw).padStart(6,'0')+'.jpg'),contentType:'image/jpeg',position:{left:66+n*574,top:178,width:550,height:380},fit:'contain',alt:'第'+raw+'帧当前高斯与实拍全身轮廓的完整对照'});
p.resolve(slide.id).speakerNotes.text=d.presentation_notes.join('\n\n');
const candidatePath=build+'/human094-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_全身一致性.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/human094-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human094-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human094-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false,quality_accepted:r.quality_accepted}));
