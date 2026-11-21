import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-head-sequence-fresh-048';
const contact=base+'/current/smplx-observable-landmark-audit-044';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='9f4402c574b250789bf3e74472248b0ff04f6bf4d95fb1e13561fa6c3d472727';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='head_temporal_local_gain_geometry_not_accepted')throw Error('Review required');
const previous=JSON.parse(await fs.readFile(build+'/face047r2-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const beforePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(beforePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{const e=row.elements.find(x=>x.text===before);if(!e)throw Error('Missing text '+before);p.resolve(e.id).text.replace(before,after);};
replace('面部表面约束与头颈对齐','头颈连续约束改善诊断帧的局部对齐');
replace('以鼻嘴表面点约束头部，联合检查相邻帧的姿态一致性。','在鼻嘴约束上，抑制世界坐标中最终头颈位姿的跳变。');
replace('每组：原图、身体约束、鼻嘴约束。左44帧拟合，右45帧诊断。','每组：原图、鼻嘴约束、增加连续约束。左33帧，右45帧。');
replace('有监督帧的鼻嘴投影改善。\n留出帧仍有偏差，头颈及整体几何未通过。','诊断帧鼻嘴更贴合，相邻头部位姿更平滑。\n整体站姿、膝部与头颈轮廓仍未通过。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two images');
const positions=[{"left": 66, "top": 240.68144499178982, "width": 565, "height": 248.63711001642037}, {"left": 665, "top": 231.3651226158038, "width": 549, "height": 267.2697547683924}];
for(const [i,file] of ['head-000033.jpg','head-000045.jpg'].entries()){
  const im=p.resolve(images[i].id);
  im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',alt:'面部诊断帧的原图、鼻嘴约束和最终世界头颈连续约束'});
  im.crop={left:0,top:0,right:0,bottom:0};im.position=positions[i];
}
slide.speakerNotes.text=await fs.readFile(build+'/head048-notes.txt','utf8');
await fs.writeFile(build+'/head048-edited-layout.json',JSON.stringify((await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']})).records,null,2));
const draft=build+'/head048-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_头颈连续_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
  pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
  verifyArtifactToolImport:true,receiptPath:build+'/head048-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/head048-final-renders';await fs.mkdir(renderdir,{recursive:true});const checks=[];
for(let i=0;i<55;i++){
  const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
  await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
  const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
  if(i<54&&!unchanged)throw Error('Unintended change '+(i+1));checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/head048-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),
  changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,unchanged_first54:true,review_required:renderdir+'/55.png'}));
