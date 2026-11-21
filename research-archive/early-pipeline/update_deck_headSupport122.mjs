import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../clip03-head-level-support/facade_post_context.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/whiteCamera122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/headSupport122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('背景相机约束对面部的影响','头部高度的固定点支持');
replace('三排为背景','三列为248、264、280帧，上排远楼，下排重复围栏。');
replace('假设背景观测噪声','60组全部审阅，唯一暂估近景候选在末帧跳到另一根立柱。');
replace('这是局部敏感度','该轨迹此前未参与相机拟合，不能据此解释现有面部重影。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'远楼与重复围栏立柱的三帧原生上下文，图片无新增文字',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height*1.5)/2,top:frame.top,width:frame.height*1.5,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N12i。保持第一组YAML的4000×3000参考尺寸、下采样和居中裁剪的工作假设。最近的N12g已使用此K。本轮不改K、相机、三维点或原始观测，也不新增检测匹配。", "预登记选择：7帧中6个冻结面部材料族的y范围加减128原生像素，横向取全图。收录任一既有成员触及范围的全部可用背景族，保持先前代表轨迹或最长轨迹加固定哈希排序。选择不依赖残差或深度。60组全部审阅，37FIT、23heldout，原划分不变。", "实际审阅15张256×256上下文页，包含全部60组中心128×128原生裁片及所有可用观测；另看4936的128像素裁片、232原图及先前审阅。未声称另行打开全部10张128裁片页。结果31组远楼重复窗带、18组植被或叶片天空边缘、5组建筑围栏植被混合、3组横幅围栏、1组台阶植被交界、1组树干叶片交界、1组重复立柱换实例。", "暂估几何仅用于深度上下文。候选相机为已完成N12g，没有优化。面部深度参考采用4个FIT标签面部族，它们未进入相机FIT。60组均为正深度，只有4936的深度/脸深度中位比0.833落在描述性区间0.5至2内。其他59组约5至14。这里的区间不是物理验收阈值，深度不是米制真值。", "4936是heldout族，track2196。248、264帧中心立柱对应横幅文字下方，280帧中心立柱上方变成卡通图案，说明实例发生切换。128裁片只见相似立柱、网格与道路白线，256上下文才更容易揭示错误。它在N12d已因围栏/道路深度交界判为不确定，从未进入N12g拟合。因此不能将它归为N12g面部错位原因。", "保留全部原坐标、视频SHA、族标签和观察身份，复用已有material_observations122.py准入模块。280帧身份记为contradicted，早两帧仍unknown，未人为确认；3个观察全部被拒绝拟合，整族留出一致。此为只读模块应用，尚未声称整套训练调用链已经验证。", "本图使用原生context_048_family_4863与context_058_family_4936中248、264、280帧的256像素裁片，组成768×512，无重采样、几何变形或新增图中文字。上排表示远楼支持，下排展示重复结构身份反例。图例用于解释审阅证据，不用于挑选准确率。", "结论仅限现有7帧轨迹：没有建立新的可信近人物、头部高度固定材料支持。灯具可见但现有循环轨迹为零；画面存在物体不等于它已提供可靠测量。不能据此断言整个视频没有固定点，也不能排除相机模型偏差或证明人物微动。", "下一步复核现有相机与训练准入消费者，确保正深度、循环一致、低残差不能覆盖物理身份反证与支持不足。复用已有证据格式，使用这一实际反例。当前候选冻结，不重复求解。新实验需新的可辨识变量或材料证据。未生成新点云，未启动训练。", "来源：robust-portrait-122/video-update-20261027/clip03-head-level-support-plan.json，clip03-head-level-support/result.json、visual-index.json、visual-review.json、observation-contract-check.json、figure-index.json。以下保留历史诊断："].concat([previous]).join('\n\n');
const candidatePath=build+'/headSupport122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122头部固定点支持.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/headSupport122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_headSupport122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/headSupport122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/headSupport122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/headSupport122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/headSupport122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
