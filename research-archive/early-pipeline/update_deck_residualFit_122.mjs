import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-fit';
const figure=root+'/visibility-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/residualAdmission-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.checks_passed!==29||review.human_training_started!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/residualFit-122-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/residualFit-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('畸变相机接入材料对应验证','拟合与遮挡使用同一畸变相机');
replace('39项检查通过；人体拟合与遮挡判断仍需接入。','投影、连续自遮挡与衣物轮廓的29项检查通过。');
replace('实拍面部仍为13.049像素，尚未放行训练。','GPU合成投影差约0.000016像素，实拍仍未通过。');
replace('绿圈：观测；蓝点：完整畸变；红叉：忽略畸变（最高6.705像素）。','左：完整畸变。中：零畸变。右：差异位置。未启动人物训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=860;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'已知前后两平面的完整畸变深度、零畸变深度及原像素差异位置，保持原始比例，图内无文字',fit:'contain'});
image.frame={left:210,top:240,width,height:width*128/576};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 R6拟合、连续自遮挡、衣物轮廓与叠加预览的完整输出相机适配。", "已建立使用完整逐帧K、外参和输出畸变的新人体拟合适配版本，关键点投影、连续网格自遮挡、衣物轮廓采样和拟合叠加预览均使用同一相机。29项有界组件及入口检查通过，旧工程代码未修改。", "CPU投影与独立OpenCV计算最大差2.27e-13px，12项坐标梯度有限差分最大差5.82e-9；一次GPU投影与反向的最大像素差1.58e-5px。相机保持常量，没有拟合或优化内外参。", "连续视线直接与网格三角形求交，已核验前后表面、背向法线、画面外及相机后方点。192×128合成深度与独立解析平面的最大差1.33e-15；改变畸变可改变传感器边界，但不应改变同一物理视线的遮挡顺序。", "10082顶点、20160三角形的已知球面，构建与全部顶点射线查询合计0.247秒。这是一次组件性能记录，不是完整人物、464帧或18000步的性能验收。", "衣物表面采样保留原生遮罩像素，按每帧完整相机投影；未知像素继续不计入背景。零畸变采样与旧版一致，掩膜尺寸错误和遗漏畸变字段会拒绝，不隐式缩放。", "新拟合入口已经连接相机适配器，并实际使用失败实拍输入检查：102个目标和13个面部观察全部保留，最大仍13.049px，在GPU、先验和拟合开始前拒绝。没有实际执行SMPL-X拟合、衣物初始化或人物高斯训练。", "可见性策略从屏幕栅格采样改为全部网格顶点和面部/足部表面点的连续射线查询；这不是只改变畸变的一项因果实验。它仍只是模型自遮挡，不识别场景外部遮挡，也不能充当真实材料或亚像素精度证据。", "旧拟合叠加预览也存在针孔依赖，新版本改为原生输出像素的射线网格遮罩。真实全尺寸预览的运行时长尚未实测；合成控制不代表人物质量改善。", "检查驱动最终汇总误复用了一个三角形索引变量，将CPU误差记成9922；原始逐项数值已正确保存。已仅从同一次检查记录修正汇总，保留原结果、原脚本和更正说明，没有重跑计算或修改组件。", "后续还需把正式人物优化器入口接到完整畸变训练组件，以及完善新拟合/表面/训练的来源绑定。当前派发入口明确停止，不回退到针孔旧训练器；真实相机和独立材料准入仍未通过。", "图：每格192×128，左完整畸变，中零畸变，右为深度差异的原始像素位置。绿色近平面，蓝色远平面。颜色与遮挡顺序已实际审阅，图中无文字。合成控制不代表实际人物改善。", "证据：native-residual-fit/result-completed.json、review.json、serialization-repair.json。以下保留历史备注。"].concat([previous]).join('\n\n');
const candidatePath=build+'/residualFit-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122畸变拟合与可见性.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/residualFit-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_residualFit_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/residualFit-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/residualFit-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/residualFit-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/residualFit-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
