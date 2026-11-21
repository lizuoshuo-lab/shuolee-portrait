import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const readable=process.argv.includes('--readable'),tag=readable?'human112-readable':'human112';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/surface-native-alignment-112',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='752d8dd6e48c406e53f7d66520c1cd55ec740a09700f51928fd52284d5a94862';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),prior=await read(build+'/human111-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','layout','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/'+tag+'-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('训练期对齐：约束可见表面','训练期对齐：先确定可靠表面');
replace('同一特征共享三维位置','原生交点与几何梯度已验证，初始方向必须来自实际表面');
replace('正确表面对照保持几何','球形初值的图像不随方向变化，旧平面对齐损失却明显改变。');
replace('继续统一交点深度','先建立可信法向再约束共享点，实际098的重影尚未修复。');
const images=slide.elements.filter(e=>e.kind==='image');
if(images.length!==1)throw Error('Expected one evidence image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(run+'/initial-normal-review.png')),contentType:'image/png',fit:'contain',alt:'112零步球形高斯的原生RGB与法向输出；固定协方差，三组方向的RGB完全相同，旧对齐损失改变；非新训练效果'});
image.frame=readable?{left:160,top:177,width:960,height:384}:frame;
p.resolve(slide.id).speakerNotes.text=d.presentation_notes.join('\n\n');
const candidatePath=build+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_训练期表面对齐'+(readable?'_修订':'')+'.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/'+tag+'-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/'+tag+'-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false,readability_revision:readable},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
