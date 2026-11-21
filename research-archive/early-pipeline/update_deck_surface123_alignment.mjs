import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/surface123-18000-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-smplx-audit-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-alignment-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-alignment-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-alignment-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('自由衣物表面训练的18000步结果','SMPL-X错位与固定相机诊断');
replace('29.4万高斯覆盖衣物，面部与鞋边重影仍未通过。','灰衣人物95个视角的面部拟合中位残差约44像素。');
replace('112个相机与点云配套上架，高斯和点云模式均已实测。','放开人体模板后仍约47像素，静态跨视角一致性尚未通过。');
replace('左为全身，右为两种面部视角。每组依次为原图、18000步渲染和实际点中心。','左为原图，中为参与拟合的检测点，右为SMPL-X投影。原视频第375帧。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),height=360,width=height*figureSpec.width/figureSpec.height;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'原生像素同裁剪的原图、检测鼻嘴点和SMPL-X投影，图内没有文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 SMPL-X错位诊断。灰衣人物尚未开始高斯训练，原队列在已有表面审阅节点停止。强鲁棒性目标继续，保留此前两个人物和沙发119。", "已实际审阅6组原生像素裁片：每组依次为原图、参与拟合的RTMW鼻嘴点、SMPL-X对应投影点，图上没有文字。前侧视角的检测点大体沿可见鼻嘴，帧76/95的SMPL-X点却落在脸颊或耳前；帧1投影落到轮廓外。灰衣人体可靠点中位23.85px，面部参与点中位43.99px，帧76面部中位100.42px。它们是检测器残差，不能当独立物理真值。固定全部相机、像素观测和权重，只将29个活跃面部点从人体模板解耦成独立静态3D点后，中位残差47.05px，仍未对齐。该诊断的原损失由4.775降至4.414，说明优化执行了；中位数不等于所优化的加权鲁棒目标，不能挑选均值掩盖大误差。已核验的帧/slot、面部17..67及脚部顺序、独立投影无整体错配；投影数值差<0.0004px。当前拟合缺少面部/脚部解剖可见性检查，灰衣47.1%的面部权重落在当前网格自遮挡/背向点；这是模型派生诊断，网格本身错位，不能直接据此删除观测。粗门槛允许身体中位误差达身高3%，灰衣约50–53px，而且不考核脸手脚；此前只允许它作表面包围盒初始化，不能解释为精准拟合。仅换SMPL-X参数无法说明或解决全部错位。相机实际K/E、检测语义和人物转头/表情贡献仍未唯一分开。已在灰衣表面审阅节点停止旧队列，灰衣尚未开始高斯训练，旧结果及后续人物保留，目标继续排查。", "三人物活跃面部观测的原生像素中位残差：戴帽蓝衣7.299，白T恤32.037，灰T恤43.992。固定相机、观测和权重，仅解除模板耦合后分别6.672、24.601、47.054。所有有效观察逐点低于1像素的目标均未通过。", "解除模板耦合只优化每个面部编号共享的静态三维点，保持原有加权鲁棒目标。灰衣目标4.775085降至4.413785，P90从92.031降至80.785，而中位数增加。优化目标下降不能冒充精准对应通过。有限局部优化不构成全局最优证明。", "数值投影差小于0.0004像素仅验证软件路径一致。条件K的物理正确性、相机外参、检测器解剖语义以及转头和表情变化仍未唯一分离。", "本轮缺少脸脚解剖可见性过滤，粗门槛只检查身体点占框高比例。当前拟合网格的自遮挡分析不是人体真值，不能按该结果或残差直接删观测。下一步先核实同一物理背景点，再做因果区分。", "白T恤独立从头训练已完成18000步，309523个自由高斯和93个相机。六幅同相机及36幅连续未知视角均审阅，面部复影、衣边拖尾和未拍背面破碎未解决。完成计算与点分布改变不代表人体或相机质量通过。", "证据目录：local-clothed-surface-123/smplx-mapping-audit、fixed-camera-face-relaxation、grayT-smplx-audit-review。所有诊断保留原观察和权重，未覆盖相机，未开始新的高斯训练。"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-alignment-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123SMPL-X错位诊断.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-alignment-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_alignment.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-alignment-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-alignment-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-alignment-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-alignment-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
