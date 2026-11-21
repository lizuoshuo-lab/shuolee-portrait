import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-cvo-control';
const figure=root+'/cvo-native-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/directFisheyeAudit-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.checks_CPU!==22||review.checks_GPU!==15||review.person_training!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeCVO-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('DirectFisheye-GS训练适配','跨视角更新与原生参数保存');
replace('本机合成控制：20万点、1520×2704的一次更新。','同一合成几何的竖屏、横屏全幅与原生裁片。');
replace('CVO可借鉴，主点与全视角增密需要适配。','双视角统计已核验，20万点无损回读与参考一致。');
replace('原生回读仍未通过，尚未验证人像重影改善。','PLY预览仍有响应差，真实人物重影尚未验收。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'本机T8合成双视角控制，两幅全图等比缩略及各372×384原生裁片，图内无文字',fit:'contain'});
image.frame={left:75,top:190,width,height:width*192/672};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 T8 CVO双视角训练组件与无损原生参数包。", "已独立实现CVO关联图与双视角训练组件，22项CPU和15项GPU检查通过。源码思想来自DirectFisheye-GS，不是官方CUDA复现。未运行人物训练。", "关联图只采用FIT中身份已确认且静态可见的前景物理族。整族跨拟合/留出时拒绝输入，一处身份矛盾排除整个物理族。合成循环对每个视角精确曝光两次，关联与随机合格配对的曝光/更新预算一致。尚未执行两策略的重建质量对照。", "每次对两个视角分别按前景与可观察mask计算损失，取平均后只更新一次Adam。与联合目标独立计算的最大梯度差3.73e-9，更新后参数最大差4.39e-7。完整K及外参冻结，遮挡未知处不参与负监督。", "三维射线求值不通过屏幕means2d反传。统计因此明确使用逐视角世界XYZ梯度范数、可见次数、最大投影半径。每个视角独有8个点的控制证明两边都记入，未只保留anchor。实际分裂、克隆或剪枝尚未实现，不能称自适应增密已完成。", "同一组200000已知合成点，两个完整相机1520×2704和2704×1520，完成一次双视角平均目标更新，全部200000中心实际移动。没有修改渲染内核、K或真实图片。PyTorch分配峰值约1238MiB，仅代表本次控制，不是稳定18k训练或总显存保证。", "新的v3推理包以无损线性float32 NPZ为原生参数，另外带完整K/E相机、来源哈希、渲染器合同及标准PLY预览。导出前数组、浮点RGB和alpha完整保留。两种尺寸回读后的五类参数逐位相同，原生RGB和alpha最大差均为0。", "同一参数导出标准PLY后，竖屏RGB/alpha最大差0.00297382/0.00374287，横屏0.00334660/0.00392181。位置与SH不变，四元数最大差1.79e-7，尺度4.66e-10，不透明度7.45e-9。此控制支持PLY表示变化可造成响应差异，不唯一定位到哪个字段或CUDA阈值，也不是像素坐标误差。", "上一轮T7的失败记录和2e-5门槛保留。本轮原生无损回读通过，不把PLY预览宣布为相同响应。现有Spark/旧原生UI尚未接入v3。", "真实面部最大重投影误差仍为13.049px，材料身份和全部训练帧相机未通过。CVO组件不会自动估计相机或修复错误材料对应。实际人物准入、关联图绑定、长期更新和质量对照仍需完成。", "本页图均为同一已知合成几何。前两列是等比例全幅缩略图，后两列是各自372×384原生裁片，未在图上写字。不能把这些图片称为人物或DirectFisheye官方训练效果。", "数值、原图、预登记范围和源码哈希位于robust-portrait-122/native-cvo-control。官方算法思想来源：https://arxiv.org/html/2604.00648v1 ，https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS 。", "图片为本机T8已知几何，不是官方论文图片，也不是人物训练结果。下方继续保留历史研究记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/nativeCVO-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122双视角组件.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeCVO-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeCVO_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeCVO-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeCVO-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeCVO-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeCVO-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
