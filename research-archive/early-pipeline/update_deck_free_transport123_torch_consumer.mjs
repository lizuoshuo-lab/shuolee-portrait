import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const work='C:/Users/Administrator/Documents/New project', b=work+'/shuolee-slide-update', tag='free-transport123-torch-consumer';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/free-clothing-transport';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'), filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/free-transport123-gpu-deck-receipt.json');
const content=await read(b+'/free-transport123-torch-consumer-content-final.json');
const tests=await read(content.final_test_evidence);
if(content.state!=='final_evidence_ready')throw Error('Final test evidence required');
const sourceHash=await filehash(source);
if(sourceHash!=='a10b4f67907a585a7aa68bf9bc68881ed553b457d08aa8327dd6022918c2c6da'||prior.final_sha256!==sourceHash||!prior.canonical_replaced)throw Error('Source changed');
const values=[content.table_headers,...content.table_rows];
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');
const last=snap[51],slide=p.resolve(last.id);
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(b+'/'+tag+'-before-layout.json',JSON.stringify(await p.inspect({kind:'layout'}),null,2));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('11次前向均完成，1次反向完成。',content.body);
replace('未加载实拍人物，未做拟合、高斯训练或PLY导出。',content.footer);
const tables=last.elements.filter(e=>e.kind==='table');
if(tables.length!==1)throw Error('Expected preserved native table');
slide.speakerNotes.text=[...content.notes_required,
 '最终测试证据：'+JSON.stringify(tests),
 '来源：'+root+'/scale-identifiability-review.json，SHA256 '+await filehash(root+'/scale-identifiability-review.json')+'；'+content.final_test_evidence+'，SHA256 '+await filehash(content.final_test_evidence)
].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123原生与GPU消费者分项验证_审阅.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:sourceHash,candidatePath,finalPath,source_result:content.final_test_evidence,source_result_sha256:await filehash(content.final_test_evidence),table_values:values},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_root_policy123_reviewed.py','utf8')).replaceAll('root-policy123-reviewed',tag);
await fs.writeFile(b+'/preserve_deck_free_transport123_torch_consumer.py',preserve);
execFileSync(py,[b+'/preserve_deck_free_transport123_torch_consumer.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','52'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[52],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑','Microsoft YaHei'],referencePath:source,referenceSha256:sourceHash},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
await fs.mkdir(b+'/'+tag+'-renders',{recursive:true});
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 await fs.writeFile(b+'/'+tag+'-renders/slide-'+String(i+1).padStart(2,'0')+'.png',bytes);
 if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected changed slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
if(await filehash(source)!==sourceHash)throw Error('Canonical changed during authoring');
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:sourceHash,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,first51_render_hashes_identical:true,canonical_replaced:false,native_table_slide:52,real_photo_on_slide52:false,actual_changed_slide_review_pending:true,source_result:content.final_test_evidence,source_result_sha256:await filehash(content.final_test_evidence)},null,2));
console.log(JSON.stringify({finalPath,first51_render_hashes_identical:true,canonical_replaced:false}));
