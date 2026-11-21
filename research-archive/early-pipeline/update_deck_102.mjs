import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-shared-support-fresh-102',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='4fb0d95a90192b6d12c9453fca6f8f8a5379ee5a63399a99d79c656aacbb3387';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),prior=await read(build+'/human101-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('静态人物重建：共享特征与相机联合优化','静态人物重建：共享特征的投影与高斯覆盖');
replace('50个同名五官对应50个唯一三维点','同名五官共用唯一三维点；高斯约束还需要正确的投影与覆盖');
replace('留出五官投影误差中位数','全相机初检：22张侧视图的43个观测没有高斯贡献，第一步前停止。');
replace('联合求解已收敛','先修正观测、相机与初始化的对应，再开展从头训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Unexpected image layout');p.delete(images.map(e=>e.id));
p.resolve(slide.id).images.add({blob:await fs.readFile(run+'/initial-support-audit/shared-feature-support-review.jpg'),contentType:'image/jpeg',
 position:{left:66,top:178,width:1150,height:380},fit:'contain',alt:'第1和124帧原始像素、未覆盖观测和共享三维点投影；102没有高斯优化更新'});
p.resolve(slide.id).speakerNotes.text=d.presentation_notes.join('\n\n')+'\n\n来源：shuolee/current/person-static-shared-support-fresh-102的预先协议、原生Ceres日志与实际投影。100位置近似的淘汰依据另见shared-point-position-preflight-100。当前没有新的高斯模型。';
const candidatePath=build+'/human102-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_共享特征覆盖核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/human102-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human102-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human102-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
