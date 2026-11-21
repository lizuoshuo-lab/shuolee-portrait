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
const prior=await read(build+'/residualFit-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/winterhat-train-006000-review/actual-review.json');
if(!review.actual_visual_review||review.steps!==6000||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-6000-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-6000-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-6000-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('拟合与遮挡使用同一畸变相机','自由衣物表面训练的6000步检查');
replace('投影、连续自遮挡与衣物轮廓的29项检查通过。','20万点覆盖穿衣轮廓，侧脸和鞋边仍有重影。');
replace('GPU合成投影差约0.000016像素，实拍仍未通过。','自动面部区域留出109/142次误差≥1像素，物理对应尚未验收。');
replace('左：完整畸变。中：零畸变。右：差异位置。未启动人物训练。','左为全身，右为两种面部视角。每组依次为原图、6000步渲染和实际点中心。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),height=360,width=height*figureSpec.width/figureSpec.height;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'6000步同相机原图、自由表面高斯与实际中心分布，左全身、右上正面、右下侧脸。原图保持比例，图内无文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-28 本地表面自由点123，戴帽蓝衣新版视频的6000步实拍检查。此页是训练中途记录，最终18k与其余人物仍需完成。',
 '用户优先要求本地训练数据集并放入查看器，沿用沙发119表面自由点方法。沙发119原结果保留，另外7段视频依次重新提取原生帧、计算新相机、重新拟合、初始化20万自由点和空优化器，不加载旧人物或旧高斯。',
 '本人物557个视频原帧，每5帧取1帧共112个预登记训练视角，112个均真实注册，无相机插值。不是使用全部557帧。首组4000×3000 YAML矩阵按实际1520×2704及显式居中裁剪与半像素约定转换后固定，条件K尚未独立验证，没有二次去畸变或新估K。',
 '静态衣物轮廓体素表面初始化，200000点，12mm网格间距，表面点不绑定SMPL三角形、没有裸模吸引。人体拟合仅提供粗范围与尺度。颜色只在人物遮罩内学习，透明度受轮廓监督。',
 review.summary,
 '1000步至6000步，实际点中心从初始表面移动的中位数由3.89mm到5.46mm，99分位由16.49mm到25.92mm，模型尺度单位。该距离不是到真实衣物表面的误差。',
 '6000步标准PLY原生回读6个相机像素完全一致，证明保存回读没有改变该渲染，不证明相机或人体正确，也不等于最终world/+Y导出或交互UI验收。',
 '固定留出观测分区：人物mask外28419次，中位0.665px。人物内部1854次，中位2.269px。自动面部ROI只有38条轨迹/142次观察，覆盖71/112个视角，其中109次>=1px，最大15.136px。整体中位0.705px会掩盖人物问题。全体仍保留48条非正深度轨迹和全部大残差，没有按误差删点或改变拟合/留出划分。',
 '按轨迹ID等距抽检6条候选的三视角原生128×128裁片，第3、4条为帽子纹理而非面部，第6条主要在下颌/颈部。真实面部样本存在眨眼和表情变化。以上只能确认ROI及外观对应候选，不是真实材料亚像素真值，也不能唯一归因于鱼眼畸变、外参或人物微动。',
 '本次有限研究训练不代表formal_training_admitted或quality_accepted。目标仍要求独立可信面部逐点<1px、全身保护、未知视角与跨人物验证。最终会交付完整协方差/SH的world/+Y PLY与同坐标camera包，并实际OpenFile检查Gaussian/points。此页时仍未交付本轮最终模型。',
 '图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),
 '记录：local-clothed-surface-123/protocol.json、winterhat-train-006000-review/actual-review.json、winterhat-regional-support/result.json、winterhat-material-samples-review/actual-review.json。以下保留此前研究备注。',previous
].join('\n\n');
const candidatePath=build+'/surface123-6000-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123自由表面6000步.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-6000-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_6000.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-6000-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-6000-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-6000-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-6000-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
