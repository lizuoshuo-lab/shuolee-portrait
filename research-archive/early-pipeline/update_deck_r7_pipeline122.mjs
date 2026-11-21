import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/surface123-face-observation-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/../robust-portrait-122/native-residual-pipeline/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/r7-pipeline122-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/r7-pipeline122-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/r7-pipeline122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('脸部观测改善了误差，但仍不是静态对齐','静态对齐未通过；训练入口检查完成');
replace('当前眼部权重为0，结果含逐帧变化，不能当作静态验收。','36项工程检查通过；眼部仍缺约束，人像尚未对齐。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第79帧的4个原生320像素裁片：原图、RTMW有效鼻嘴点、FAN有效鼻嘴点、同一皮肤点观测与逐帧变形回放。绿圈观测、紫叉RTMW对照、橙叉FAN对照。图内无文字，未变形。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["R7原生训练入口已完成22项CPU与14项CUDA工程检查。复现并修复了图像缓存标识被通道名覆盖的问题，重复访问现在复用同一原生张量；原先像素值一致，此错误不是已证实的人脸错位原因。完整K、外参和6+2+4输出残余系数随每帧传入训练，缺失系数不默认为0。96点合成场景共12次优化更新，双视角梯度与独立联合损失最大差3.73e-9，原生NPZ/相机回读及RGB/alpha回放一致。新入口对既有102目标、13个失败脸部观察仍拒绝，最大13.049像素未被删除或放宽。没有新人物拟合或训练，静态SMPL-X对齐仍未通过。", "只读审阅00刚完成的FAN逐帧对照18k、全帧验证和退步帧，区分姿态补偿、原相机误差及自由点表面分布；这仍是诊断对照，不直接改成本静态主线。R7已完成的合成检查不重跑。没有新的物理材料/相机证据时不靠重复训练声称修复。", "本页保留此前真实人物证据图；新R7仅为已知96点非零畸变合成工程控制，并未修复或重训人物。", "CPU检查22项，耗时27.57秒；GPU检查14项，耗时61.57秒。总12次小场景优化更新，少于登记16次和120秒CUDA预算。", "缓存变量覆盖导致重复加载但像素相等，不能作为既有人脸错位根因。原始失败记录保留，修复后相同帧实际复用张量。", "完整K/E/D传递，缺失系数拒绝；两视角梯度与独立联合损失最大差3.73e-9；同相机原生回读RGB/alpha逐值一致。", "同一实际失败输入在新拟合/训练入口均返回3，保留102目标与13脸部观察、最大13.0488像素，GPU/初始化/优化器未启动。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\native-residual-pipeline\\conclusion.json"].concat(['此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/r7-pipeline122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生训练入口核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/r7-pipeline122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_r7_pipeline122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/r7-pipeline122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/r7-pipeline122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/r7-pipeline122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/r7-pipeline122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
