import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/reports/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'), hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/human116-p3-deck-receipt.json'), expected=prior.canonical_sha256;
if(await hash(source)!==expected)throw Error('Canonical source changed');
const d=await read(build+'/slide117-content.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','layout','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
await fs.writeFile(build+'/human117-layout-inspection.json',JSON.stringify({layout:await p.inspect({kind:'layout'}),masters:p.masters.items.map(x=>({id:x.id})),slide52:snap[51]},null,2));
const slide=snap[51], before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source last slide render changed');
await fs.writeFile(build+'/human117-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('HSfM 联合对齐',d.title);
replace('13视角本地适配',d.summary);
replace('下半脸中位差',d.observation);
replace('候选拒绝，保留074',d.conclusion);
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one image');
const img=p.resolve(images[0].id);
img.replace({blob:new Uint8Array(await fs.readFile(build+'/slide117-evidence.jpg')),contentType:'image/jpeg',fit:'contain',alt:'原38、50、63帧，鼻部语义标签周围原始像素等比例裁剪；用于材料身份核查，未宣称三维对齐。'});
img.frame={left:65,top:177,width:1150,height:384};
p.resolve(slide.id).speakerNotes.text=d.notes.join('\n\n');
const candidatePath=build+'/human117-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_117诊断校订.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/human117-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath},null,2));
execFileSync(python,[build+'/preserve_deck_117.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/human117-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/human117-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human117-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected render change on slide'+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human117-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
