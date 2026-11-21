import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/shared-camera-points-allview-104',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='03afa430de64d0f2ee4e57a2b66355b01ad7b080b4144ec797c987f0229f5f55';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),prior=await read(build+'/human102-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render does not match reviewed version');
await fs.writeFile(build+'/human104-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('静态人物重建：共享特征的投影与高斯覆盖','静态人物重建：跨视角共享三维特征');
replace('同名五官共用唯一三维点','175个相机联合约束50个共享面部点，每个特征只有一组三维坐标');
replace('全相机初检：22张侧视图','下半脸投影误差中位数从3.39降至2.01像素。躯干保护未通过。');
replace('先修正观测、相机与初始化','下一步核对真实可见性与观测误差，再检验全身一致性。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Unexpected image layout');p.delete(images.map(e=>e.id));
p.resolve(slide.id).images.add({blob:await fs.readFile(run+'/all-view-shared-projection-review.jpg'),contentType:'image/jpeg',
 position:{left:66,top:178,width:1150,height:380},fit:'contain',alt:'第1、70、124帧真实照片中原相机与104共享空间点联合求解的投影对照；每个特征在所有相机中只保留一个XYZ'});
p.resolve(slide.id).speakerNotes.text=d.presentation_notes.join('\n\n')+'\n\n来源：shuolee/current/shared-camera-points-allview-104，包括预先协议、原生Ceres日志、独立留出纹理检查、全部5815个检测的原mask核验。冻结覆盖证据见shared-observation-support-audit-103。104未发布相机，未进行高斯训练，当前可下载的模型仍为098。\n\n公式见配套网页body-smplx.html#shared104-result：u_ik≈π(K_i(R_iX_k+t_i))，同一k的X_k不随相机i变化。';
const candidatePath=build+'/human104-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_全视角共享点核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/human104-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human104-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human104-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
