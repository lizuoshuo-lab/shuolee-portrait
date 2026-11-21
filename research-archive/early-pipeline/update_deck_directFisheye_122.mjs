import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/directfisheye-research';
const figure=root+'/../native-capacity/capacity-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/nativeEdgeProfile-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.source_files!==21||review.training_executed!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/directFisheyeAudit-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('原生像素定位与模型偏差','DirectFisheye-GS训练适配');
replace('两条建筑轮廓的原生等间隔裁片。','本机合成控制：20万点、1520×2704的一次更新。');
replace('部分轮廓抖动减小，真实定位精度仍未确认。','CVO可借鉴，主点与全视角增密需要适配。');
replace('明暗响应与邻边混合会产生系统偏差，尚未训练。','原生回读仍未通过，尚未验证人像重影改善。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'本机T7合成控制的RGB与alpha整体及原生裁片，非DirectFisheye训练结果，图内无文字',fit:'contain'});
image.frame={left:75,top:190,width,height:width*192/672};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 DirectFisheye-GS官方源码调研与本机T7原生容量控制。", "可以借鉴DirectFisheye-GS的跨视角联合优化CVO。它将相关视角的图像梯度累积到同一组高斯，可用于现有成像模型，论文也给出了针孔输入的对照。此判断不意味着当前人物已经对齐。", "原生鱼眼分支使用KB四系数角度多项式及解析Jacobian，把三维协方差投影成二维高斯，再做屏幕空间混合。与本机gsplat UT/三维响应后端的求值方式不同，应明确分别记录渲染器，不能将两者图片误差混为一项。", "现有MP4已去畸变、缩放并裁剪。不可直接套用原鱼眼系数或把相机类型改成OPENCV_FISHEYE。当前仍以完整条件K和原生像素建模，实际剩余畸变需要独立证据。KB系数全零也不等于针孔，相同f时半径分别为f*theta与f*tan(theta)。", "公开训练循环只优化高斯位置、SH、不透明度、尺度、旋转，以及可选颜色映射。相机K、外参和畸变没有进入优化器，循环里没有真实材料特征点的重投影损失。CVO不能自动修好错误外参、错误对应或人物微动。", "公开加载链丢弃cx/cy，只保留焦距转换的FOV。CPU调用实际读取函数确认：改cx/cy不会改变载入结果。以当前1520×2704条件K计算，居中假设引入水平1.8168px、垂直5.7027px，合计5.9851px偏移。这里只证明直接移植该入口的风险，不能据此断言现有本机模型就是这个错误。", "多视图反向梯度确实累积后统一step，但增密的屏幕梯度、可见性与半径在视图循环外只读取最后一个视角。由于reversed(cam_list)，这个视角是anchor。不能把该代码称为全部视角都参与增密统计。", "公开角度函数对COLMAP世界到相机R使用R*z，而世界中的光轴应为R转置*z。CPU已知旋转控制仅改变世界坐标基，公开两视角夹角由40度变为0度，正确表达仍40度。该问题影响角度排序与auto跳邻居统计，overlap排序本身不读取角度。", "公开l1_loss已先归约为标量，随后乘图像梯度权重并不能逐像素强调边缘。RGBA读取条件误检查shape[1]是否为4，常规C/H/W布局下检查的是高度，不能直接用于我们要求的前景mask监督。", "当前HEAD默认recip策略包含前7000步随机伙伴、后续概率关联与互惠回访、20001至30000步EMA，且按数据自动选skip。只比较bs1与bs2同时改变了图像预算等因素，不足以独立判断CVO。后续受控版本需明确关闭额外混合因素，并固定每个视角曝光预算。", "本轮固定官方提交db9b3a25bff79a59f02ce90f5bd235cb59c03aaa，核验21个源码文件。仅做阅读与隔离CPU控制，没有安装官方CUDA扩展、下载新权重、运行新人物训练或发布新点云。原生GPU性能和实际重影改善仍未验证。", "根LICENSE.md限制为研究与评估用途。训练本身无需下载专门预训练权重，但这不构成可直接商业分发代码的许可。", "CVO对照应固定相机、同一初值和渲染器、每次两个视图、曝光预算及更新预算，单独改变关联与随机配对。关闭额外EMA、颜色映射与相机优化，归一化损失并汇总所有视角的增密统计。这是设计，尚未执行。", "官方论文：https://arxiv.org/html/2604.00648v1 ，项目页：https://yzxqh.github.io/DirectFisheye-GS/ ，官方源码：https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS/tree/db9b3a25bff79a59f02ce90f5bd235cb59c03aaa 。文件、哈希与具体行位置记录在robust-portrait-122/directfisheye-research。", "本页图片来自本机T7已知合成几何测试，不是DirectFisheye-GS或真实人物训练结果。图左两个缩略图依次为RGB和alpha，右侧为同区域456×384原生裁片，无变形、无图内文字。", "T7固定合成200000点与已知针孔相机，在1520×2704原生大小完成一次全属性更新。五类参数梯度均有限且非零，200000个中心均实际移动。没有用人物数据、旧人体或新的相机拟合。", "该次前向1.923秒、反向0.126秒。PyTorch峰值已分配约374MiB、保留510MiB，包含本控制执行期间的张量。它不是总显存、稳定性能基准或18000步训练保证。", "PLY回读最大RGB差0.001623899、alpha差0.002812505，超出预登记0.00002门槛。原试验失败原样保留，横屏控制未执行，没有降低分辨率、点数或放宽门槛重试。这里是0至1响应数值差，不是像素坐标误差。", "原往返位置与SH逐位不变，尺度最大差4.66e-10，不透明度7.45e-9，完整协方差最大差4.66e-12。很小的参数误差也不等于整张图每个像素都满足严格响应门槛。", "T7b只读取已保存的同一个200000点包。与保存的响应相比，再次渲染RGB和alpha逐元素完全相同。直接保存并回读无损NPZ参数也逐位一致，原生渲染差为0。本次没有复现随机渲染差异。", "对这份已量化包再做一次标准PLY往返，仅15个四元数分量发生变化，最大3.73e-9，RGB最大差2.09e-7、alpha最大差2.98e-7。这次没有复现首轮约0.00281的差异。", "首轮导出前完整更新参数与浮点参考未保存，不能凭后续对照唯一确定首轮差异原因。CUDA源码中的alpha低阈值和透射率早停提供不连续性假设，但本轮未证明某个具体像素由该分支引起。", "后续原生验收需要保留无损参数及完整浮点参考，标准PLY仍作为通用预览交付。这是原生渲染与保存链路的工程工作，不能解释或修复真实人像13.049px的映射误差。", "残余畸变D1补偿的建筑留出RMS从1.773变为1.783px，未通过。D2至D4定位到采样及响应模型误差，但仍未验证出可采用的补偿。旧K、原始像素、物理对应与119历史模型全部保留。", "研究结论与T7/T7b完整证据已写入原进展页及原工作日志。以下保留历史技术路线记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/directFisheyeAudit-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122DirectFisheye核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/directFisheyeAudit-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_directFisheye_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/directFisheyeAudit-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/directFisheyeAudit-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/directFisheyeAudit-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/directFisheyeAudit-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
