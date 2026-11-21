import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/portrait-alignment-open-source-116/pixsfm/ba-vgg-candidate';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/human114-deck-receipt.json'),expected=prior.final_sha256,d=await read(run+'/decision.json');
if(await hash(source)!==expected)throw Error('Canonical report changed');
if(!d.visual_review_completed||d.state!=='reviewed_rejected')throw Error('Actual P1 review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','layout','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source slide changed');
await fs.writeFile(build+'/human116-p1-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('两条人像路线：重影尚未消除',d.slide_title);
replace('同相机对照：左实拍，中GaussianAvatar，右SMPL-X适配',d.slide_summary);
replace('SMPL-X实拍纹理较清楚；鼻口、镜框和身体边界仍有重影。',d.slide_observation);
replace('175视角、20万初始点、18,000步；两组均未通过，不追加训练。',d.slide_conclusion);
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one evidence image');
const img=p.resolve(images[0].id);
img.replace({blob:new Uint8Array(await fs.readFile(run+'/slide52-evidence.jpg')),contentType:'image/jpeg',fit:'contain',alt:'三个原始留出观测：绿圈是真实图像材料特征，蓝圈为074投影，橙圈为PixSfM候选投影。三幅均为相同不变形裁剪和缩放。'});
img.frame={left:65,top:177,width:1150,height:384};
p.resolve(slide.id).speakerNotes.text=d.slide_notes.join('\n\n');
const candidatePath=build+'/human116-p1-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_116相机核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human116-p1-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human116-p1-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human116-p1-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
