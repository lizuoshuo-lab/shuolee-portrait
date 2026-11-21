import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='1c3844d0b12c4c836b60f0a19b54573c8aa9e337f804fcf546a43fb585d38b01';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical changed');
const con=JSON.parse(await fs.readFile(base+'/current/smplx-foot-semantics-fresh-050/phase-conclusion.json','utf8'));
if(con.state!=='foot_point_conflict_confirmed_removal_not_accepted')throw Error('Review incomplete');
const previous=JSON.parse(await fs.readFile(build+'/head048-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','notes.text']});
const slide=p.resolve(snap.records[0].id);
const before=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(before)!==previous.renders[54].render_sha256)throw Error('Source appearance changed');
const old=slide.speakerNotes.text;
if(old.includes('049诊断和050从头对照'))throw Error('Notes already applied');
const appendix=await fs.readFile(build+'/foot050-notes-append.txt','utf8');
slide.speakerNotes.text=old+appendix;
const draft=build+'/foot050-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_足部对应说明.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/foot050-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
if(!q.slides.items[54].speakerNotes.text.includes(appendix.trim()))throw Error('Notes missing');
const checks=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 const h=digest(bytes);if(h!==previous.renders[i].render_sha256)throw Error('Unintended visual change '+(i+1));
 checks.push({slide:i+1,render_sha256:h,unchanged:true});
 if(i===54)await fs.writeFile(build+'/foot050-final55.png',bytes);
}
await fs.writeFile(build+'/foot050-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[],changed_notes_slides:[55],all_slides_rendered:55,renders:checks,
 notes_preserved:true,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,visual_parity_all55:true,changed_notes:[55]}));
