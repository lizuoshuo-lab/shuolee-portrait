import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const build=process.cwd(),workspace='C:/Users/Administrator/Documents/New project';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/hair-structure-compatibility';
const dist=root+'/dynamic-distillation';
const result=JSON.parse(await fs.readFile(dist+'/result.json','utf8'));
if(result.actual_optimizer_updates!==2360)throw Error('Incomplete experiment');
const sourceOriginal=base+'/reports/02_有限视角重建优化路线.pptx';
const buffer=await fs.readFile(sourceOriginal),sha=crypto.createHash('sha256').update(buffer).digest('hex');
if(sha!=='7e513381f4df064130183c5fdb44269c38ecd49b55cf90379e04d69dd0ba227e')throw Error('Source changed; preserve user edits');
const source=path.join(build,'source-before-hair.pptx');await fs.writeFile(source,buffer);
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==53)throw Error('Unexpected slide count');
const snap=await p.inspect({kind:'slide,textbox,image,shape,layout',maxChars:1000000});
await fs.writeFile('hair-source-inspection.ndjson',snap.ndjson);
const rows=snap.ndjson.trim().split('\n').map(JSON.parse);
await fs.writeFile('hair-before-53.png',new Uint8Array(await (await p.export({slide:p.slides.items[52],format:'png',scale:1})).arrayBuffer()));
function text(s,name,value,left,top,width,height,size=26,bold=false,color='#183342'){
 const sh=s.shapes.add({geometry:'textbox',name,position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={fontSize:size,typeface:'微软雅黑',color,bold};sh.text.insets=0;return sh;
}
function title(s,value){text(s,'title',value,66,42,1148,67,42,true);}
function footer(s,n){text(s,'page-number',n+' / 55',1130,679,86,20,14,false,'#536674');}
async function img(s,file,alt,top,height){s.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',alt,fit:'contain',position:{left:66,top,width:1148,height}});}
for(const row of rows){if(row.kind==='textbox'&&/^\d+ \/ 53$/.test(row.text))p.resolve(row.id).text=row.text.replace('/ 53','/ 55');}
const oldLast=p.resolve(rows.find(r=>r.kind==='slide'&&r.slideIndex===52).id);
const oldTitle=rows.find(r=>r.kind==='textbox'&&r.slideIndex===52&&r.name==='title');
const remaining=rows.find(r=>r.kind==='textbox'&&r.slideIndex===52&&r.name==='remaining');
if(!oldTitle||!remaining)throw Error('Missing conclusion anchors');
p.resolve(oldTitle.id).text='独立衣物与身体观测补充';
p.resolve(remaining.id).text='同一组59张训练观测，两组衣物均为885次更新，原人体和相机保持固定。';
const method=p.slides.add();method.background.fill='#FFFFFF';
title(method,'动态头皮上的发束与模型监督');footer(method,54);
text(method,'method','固定已训练人物，渲染各时刻的颜色与头发掩膜。\n共享 UV 网络解码发束，发根沿用现有头皮运动。',66,121,1148,80,26);
await img(method,dist+'/000023-teacher.png','真实训练观测、原动态模型渲染、模型头发掩膜与实拍语义完整四列对照',214,318);
text(method,'scale','横向标准差  σ⊥ = 10⁻³ sₕₐᵢᵣ ≈ 0.165 模型毫米',66,552,1148,42,29);
text(method,'optimization','每步采样1万条发束，联合拟合颜色、掩膜和方向。\n发型扩散先验作为附加结构约束，身体与背景保持不变。',66,605,1148,64,24);
method.speakerNotes.text=`沿用现有动态人物和真实静态背景，不重建整套系统。人体191595高斯、59张顺序视频观测、主轮18000次更新保持。动态发根使用已经注册的439顶点/820面头皮，每时刻由同一人体运动获得三角形，再通过固定重心坐标产生发根和局部基。这里是动态人体上的本地适配，不是完整OmniRe或Gaussian Haircut官方复现。
官方Gaussian Haircut c18714ddfb799029f7cd8f53984dfdd978e87ee8的run.sh实际流程为：自由3D高斯30000步并学习方向；FLAME配准；局部头发裁切和内部点移除；渲染训练视角；潜变量发束20000步；显式发束10000步。run.sh169-180使用load_synthetic_rgba、load_synthetic_geom、binarize_masks，读取前阶段的颜色、头发/头部掩膜与方向。因此前阶段模型渲染在该阶段充当监督，不能写成真实图像或新增视角。
本轮使用保留的动态人体作为教师模型，并未完成官方自由高斯方向训练阶段。选择12661个既有头发高斯。局部尺度计算使用其中9540个不透明度至少0.5的点，按官方scale_scene_into_sphere.py五次过滤流程：每次去除距当前中心大于中位距离5倍的点，更新中心，记录最大距离。所得局部半径s_hair=0.556308686733世界坐标单位，sigma_perp=0.001*s_hair=0.0005563087世界单位，约0.16497159拟合模型毫米。这是模型坐标，不是实测发丝直径。此前按人体尺度设置1模型毫米约粗6.06倍，本轮纠正该换算。
头皮按三角形面积采样50000个固定候选发根，每步随机选10000，其中1000条解码引导发束，9000条按UV邻域4个引导插值。每条100点99段，共990000段高斯。官方发束解码器和发型扩散先验冻结，训练共享256×256×129 UV网络及外观SIREN。几何64维，外观65维的第一通道按源码保留为未用标签。SH一阶与原人体一致，宽度和不透明度固定。随机UV初始化，外观输出初始化为原头发平均SH，不再把旧头发中心预拟合成潜变量。局部插值公式与动态绑定属于适配。
两组各1180步，59帧各20遍。两组相机和随机发根顺序哈希相同。AdamW学习率0.001，默认weight_decay0.01。总损失 L=0.8 L_teacher_RGB+0.1 L_teacher_mask+0.1 L_orientation+0.5 L_real_face+0.01 L_diffusion；最后一项仅先验组启用。颜色采用完整原生相机渲染后的头部矩形ROI，掩膜为教师合成覆盖超过0.5。由于教师没有学习方向，方向项仍来自38张可靠真实图像的Gabor无向方向，按置信度加权；其余21张保留RGB监督。真实脸部保护使用实拍分割区域。方向和几何先验不是未知视角真值。
每次原生全分辨率渲染后再裁剪，没有用错误局部相机替代。模型输出的头发掩膜与真实语义平均IoU0.8932，侧面耳旁仍有不一致。教师保留原模型的误差，只是一种训练参考。验收必须回到真实照片、同一未知俯仰路径和动态一致性。
源码来源：https://github.com/eth-ait/GaussianHaircut/blob/c18714ddfb799029f7cd8f53984dfdd978e87ee8/run.sh；src/utils/camera_utils.py51-78；src/preprocessing/scale_scene_into_sphere.py38-69；src/scene/gaussian_model_latent_strands.py510-515。本地图源：${dist}/000023-teacher.png。实际配置和指标：${dist}/protocol.json、teacher-result.json、result.json。`;
const conclusion=p.slides.add();conclusion.background.fill='#FFFFFF';title(conclusion,'当前效果与后续约束');footer(conclusion,55);
text(conclusion,'method','校正局部尺度与监督来源后，两组分别训练1,180步。',66,121,1148,40,26);
await img(conclusion,dist+'/000023-trained.png','真实观测、保留动态模型、模型监督发束、附加发型先验的完整同相机对照',174,382);
const sums=result.summary;
text(conclusion,'metrics',`头发训练 L1：原模型 ${sums.parent.hair_L1.toFixed(5)}，发束 ${sums.teacher_control.hair_L1.toFixed(5)}，加先验 ${sums.teacher_diffusion.hair_L1.toFixed(5)}。`,66,573,1148,42,24);
text(conclusion,'remaining','衣物与地面已有局部改善，头发纹理和未知俯仰仍未通过。\n保留原动态模型，后续优先保护观测细节，再约束局部几何。',66,624,1148,58,23);
conclusion.speakerNotes.text=`本页为实拍与实际原生训练渲染，未生成效果图。每组1180步，总2360更新，原人体主轮仍18000步。头发训练区域使用同一真实语义掩膜，指标见${dist}/result.json。教师矩形ROI误差和真实头发区域误差具有不同区域与参考，不能直接混为一谈。固定评测发根，另做多个随机发根的只读渲染以检查采样影响，结果见sampling-audit/result.json。完整视觉判定见review.json。
局部尺度和模型监督来源已按源码纠正，但本轮未恢复原实拍的发丝细节和准确轮廓，没有采用新发束模型。短程结果不能否定完整Gaussian Haircut方法，也不能把权重可加载和梯度可运行称为复现完成。现有教师缺少官方方向阶段，动态头皮和根坐标不是实测3D头发，未知头皮三角形保留未知。先验不能创造真实证据。
已有独立衣物30592点改善部分轮廓，静态和动态衣物的差异小，不宣称动态本身已证明更好。背景恢复9733点改善脚边缺口，但鞋底边缘、脸部、衣领和未知俯仰头顶伪影仍在。当前查看器候选490553点和原18k人体均保留，不在本地整体未通过前部署00。
全部59张均来自水平顺序视频且全部用于训练。没有同时拍摄的多视图真值，没有独立测试集，也没有真实未知俯仰对照。动态形变来自图像优化，不能当作真实运动测量。眼镜专项停止，不继续180000旧任务。下一步以已观察到的纹理为保护条件，对原高斯做小范围结构调整或另选可保留原始外观的路线，不通过无界增加步数或再次放大厚度掩盖误差。`;
await fs.mkdir('hair-final-renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile('hair-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 if((i+1)%10===0)console.log('Rendered '+(i+1)+'/55');
}
await fs.writeFile('hair-final-montage.png',new Uint8Array(await(await p.export({format:'png',montage:true,scale:.22})).arrayBuffer()));
const candidatePath=path.join(build,'hair-candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath:path.join(workspace,'shuolee-slide-output/02_有限视角重建优化路线_20261022_发束修订.pptx'),
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:sha},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'hair-final-validation.json')});
console.log(JSON.stringify(receipt));
