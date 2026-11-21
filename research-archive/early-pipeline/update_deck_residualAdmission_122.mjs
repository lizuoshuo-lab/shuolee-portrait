import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-admission';
const figure=root+'/residual-admission-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/residualCapacity-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.checks_passed!==39||review.human_training_started!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/residualAdmission-122-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/residualAdmission-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('完整畸变相机的全尺寸验证','畸变相机接入材料对应验证');
replace('20万高斯，双视角更新与实际界面回放已核验。','39项检查通过；人体拟合与遮挡判断仍需接入。');
replace('两路浮点回放差为0；实拍标定仍未通过。','实拍面部仍为13.049像素，尚未放行训练。');
replace('原生1520×2704与2704×1520，完整K、外参与畸变。','绿圈：观测；蓝点：完整畸变；红叉：忽略畸变（最高6.705像素）。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=460;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'四台已知合成相机的48个留出目标，绿圈为观测、蓝点完整畸变预测、红叉忽略畸变；图内无文字',fit:'contain'});
image.frame={left:410,top:190,width,height:width*3/4};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 R5完整畸变相机正式准入与材料对应验证。", "正式输入检查和独立材料对应验证已新增完整输出畸变相机版本：同一份逐帧K、外参和12个畸变参数贯穿像素到射线、三维三角化和目标重投影。39项CPU检查通过，未运行人物训练。", "四台已知合成相机、48个留出目标中，16个面部类别观察最大误差为2.33e-13px；忽略同一批相机的畸变后，面部最大误差为6.705px。它证明完整模型被计算使用，不证明当前实拍的畸变值已求准。", "三角化仅使用其他视角的支持点；修改目标点不会改变它自己的三维预测。独立轨迹留出、物理身份、微动、测量不确定性和参数哈希检查继续生效。没有按误差删点或改阈值。", "显式零畸变与旧版含噪针孔计算的最大差为1.04e-13px；世界坐标相似变换后的最大像素差为1.17e-13px。测量使用整数像素中心，渲染器使用像素边界坐标，+0.5/-0.5只转换一次。", "原实拍102个目标和13个面部观察全部保留，面部最大误差仍为13.049px，面部通过数0。464帧源视频不能被7个诊断相机替代；实际输入入口在导入GPU或创建优化器前拒绝。", "只读检查还发现旧人体拟合的关键点投影、网格遮挡判断，以及衣物表面轮廓采样仍采用针孔。下一步必须同时接入非线性投影与可见性，不能只换渲染器或只改二维关键点公式。", "新版本目前完成CPU准入和材料验证，尚未串通新人物拟合和训练。旧入口明确拒绝新相机格式；这些工程缺口不能直接认定为历史重影的根因。未修改旧结果、实际畸变参数、相机或观测。", "图：四格按相机0、1、2、3排序，每格320×240原生坐标。绿圈已知独立观测、蓝点完整模型预测、红叉忽略畸变预测。48个留出目标全部保留，误差线未放大。合成观测并非实拍人脸。", "证据：native-residual-admission/result.json、review.json、upstream-projection-audit.json。以下保留历史备注。"].concat([previous]).join('\n\n');
const candidatePath=build+'/residualAdmission-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122畸变材料验证_v2.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/residualAdmission-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_residualAdmission_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/residualAdmission-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/residualAdmission-122-final-validation-v2.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/residualAdmission-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/residualAdmission-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
