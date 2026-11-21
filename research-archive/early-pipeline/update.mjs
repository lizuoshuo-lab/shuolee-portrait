import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const build=process.cwd();
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background';
const pilot=root+'/surface-gaussian-pilot';
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(build,'source.pptx')));
const inspection=(await fs.readFile('inspection.ndjson','utf8')).trim().split('\n').map(x=>JSON.parse(x));
const set=(id,text)=>{p.resolve(id).text=text;};
function text(slide,name,value,left,top,width,height,size=26,bold=false,color='#183342') {
  const sh=slide.shapes.add({geometry:'textbox',name,position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  sh.text=value; sh.text.style={fontSize:size,typeface:'微软雅黑',color,bold}; sh.text.insets=0;
  return sh;
}
async function picture(slide,name,file,left,top,width,height) {
  const isSvg=file.endsWith('.svg');
  return slide.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:isSvg?'image/svg+xml':'image/png',alt:name,fit:'contain',position:{left,top,width,height}});
}
function clearContent(index,exceptIds=[]) {
  p.delete(inspection.filter(x=>x.slideIndex===index&&['textbox','image','shape'].includes(x.kind)&&!exceptIds.includes(x.id)).map(x=>x.id));
}
function footer(slide,number){text(slide,'page-number',`${number} / 53`,1130,679,86,20,14,false,'#536674');}

set('sh/x4r21kru','动态人体与真实背景的重建目标');
set('sh/ozy1ofad','人体共享一套高斯外观，随时间更新表面位置。静态背景保留原场景坐标，\n与人物共同渲染遮挡。');
set('sh/b29kza94','全部 59 个已标定观测参与本轮训练。191,595 个人体高斯完成 18,000 次更新，\n固定相机检查真实时间变化。');
set('sh/a10jqpsj','固定人物时刻，再改变俯仰和距离。实拍重建与连续新视角共同决定是否采用候选，\n结果汇总见末章。');
p.resolve('sl/hwbqtkby').speakerNotes.text='当前执行以2026-10-22的动态全视角实验为准。59张均为训练观测，没有独立测试集。191595为人体高斯数，258633为冻结真实背景数。旧180000步计划已停止，当前主轮为18000步。眼镜专项关闭。\n来源：'+root+'/run18000/protocol.json\n'+root+'/run18000/result.json';

const s33=p.resolve('sl/nax4b2dk');
clearContent(32,['sh/9o3m9knm','sh/lk3m50nq']);
set('sh/9o3m9knm','2DGS：通过表面交点约束几何');
await picture(s33,'2DGS论文完整Figure3',path.join(build,'2dgs-figure3.svg'),66,130,560,320);
text(s33,'paper-source','Huang 等，2D Gaussian Splatting，2024，Fig. 3。',66,462,560,32,15,false,'#536674');
text(s33,'surface-principle','二维面元由中心、两个切向方向和切向尺度定义。像素射线与面元相交后，按局部高斯值和透过率合成颜色。',666,157,548,153,27);
text(s33,'surface-geometry','同一面元的空间方向跨视角共享。深度畸变损失压缩射线上的贡献分散，法向损失对齐深度与面元方向。',666,340,548,145,27);
text(s33,'dynamic-adaptation','本地适配继续使用已有动态网格绑定与相机。对照组只优化图像，实验组增加几何约束，使用相同 59 个观测与 1,180 步预算。',66,540,1148,100,27);
s33.speakerNotes.text='原始方法：Huang et al., 2D Gaussian Splatting for Geometrically Accurate Radiance Fields, SIGGRAPH2024. 图3完整放置，不裁切。https://arxiv.org/html/2403.17888v1#S4.F3\nhttps://github.com/hbb1/2d-gaussian-splatting/tree/f3e3b9fa67bbd1c75e05167ff37391d8dab2a678\n本地实现采用原始diff-surfel-rasterization e0ed0207b3e0669960cfad70852200a4a5847f61，不修改CUDA。已有3D高斯局部0/2切向轴映射为2D面元0/1轴，法向厚度不参与新表示；保留三角面片绑定。归一化位置、尺度和相机平移使用同一world_scale=3.3721487257，实际源相机投影差小于0.00034像素。平面深度、法向、整数裁剪、有限差分预检通过。\n本地损失为0.8L1+0.2(1-SSIM)+0.25头部L1+0.05轮廓误差+小参数锚定项，几何臂再加0.05法向误差+1000原生NDC深度畸变。前118步关闭几何，随后118步渐增。1000为本地试验权重，当前官方代码默认lambda_dist为0，不冒充官方默认。法向由完整K反投影期望深度，通过邻接射线交叉积求得，使用前景掩膜腐蚀7像素后的区域。原生浮点负畸变截断至0。\nXYZ学习率0.001，DC0.003，头部SH1为0.0005，opacity0.02，切向尺度0.003，旋转0.001，65%预算后乘0.3。位置每坐标限制6模型毫米，尺度相对父模型限制0.5至2，局部四元数向量分量限制±0.25。点数固定191595，不增密，不做眼镜专项。动态与背景冻结，不是独立重新训练18000步，也不是完整官方动态系统复现。\n实际证据：'+pilot+'/preflight/result.json\n'+pilot+'/training/result.json\n'+pilot+'/review/result.json';

set('sh/czq5czux','已有动态结构：人体运动与静态背景');
set('sh/byh43utc','上式为骨骼蒙皮的通用形式。现有 SMPL-X／FLAME 网格提供人体先验。\n高斯保持面片绑定，共享颜色和局部参数。');
set('sh/qx8napc7','本轮复用 ENeRF 试验的局部形变图：128 个节点、每顶点 4 个邻居，\n以三次时间函数表达平滑旋转和平移。');
set('sh/d0zm54bi','固定相机与 258,633 个真实背景高斯，用全部 59 个观测优化人体。\n时间参考固定在第 23 帧，避免全局漂移与局部运动混淆。');
p.resolve('sl/ofu103u5').speakerNotes.text='图中蒙皮公式是通用人体先验原理，不声称本轮逐帧优化骨骼姿态。实际方法：保留12个已注册网格，其余59时刻以插值或端点保持初始化。此前ENeRF残差图代码提供当前动态分支。128节点为80身体+48头部，FPS采样，顶点4近邻混合。时间tau=(frame_index-23)/35，三次多项式不含常数项，参考23帧残差恒等。旋转系数0.035*tanh弧度，平移系数0.01*world_scale*tanh。面片位置与方向随形变更新，局部高斯参数共享。刚性正则0.005和系数幅度正则0.001。\n人体更新XYZ、切向/法向尺度、opacity、DC和头部一阶SH，局部高斯旋转、相机、拓扑、点数、背景固定。学习率XYZ0.0015逐渐降至0.0002，DC0.004，SH0.001，opacity0.035，scale0.0015，motion0.001。主轮18000次实际更新。\n静态背景从此前同坐标系Mip+depth模型分离，原316044点，去除主体后保留258633点。人物与背景按真实深度共同合成，背景参数本轮没有更新。全部59观测均参与训练。动作较小且通过图像估计，不是GT运动或官方OmniRe完整复现。\n来源：'+root+'/run18000/protocol.json\nC:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/dynamic-human-enerf-residual-graph-067/enerf_residual_graph.py';
set('sh/25krep0r','本页为早期固定头像对照，保持原试验条件。当前全身时间形变与真实背景结果见末章。');

const s51=p.resolve('sl/36hsrmts');
clearContent(50,['sh/ihsv6lwb','sh/l8rat03a']);
set('sh/ihsv6lwb','全视角动态训练与真实背景');
await picture(s51,'实拍与18000步动态人体加背景对照',root+'/run18000/review/000000-scene-comparison.png',66,125,690,520);
text(s51,'body-data','59 个真实观测\n191,595 个人体高斯\n18,000 次训练更新',810,178,404,135,28,true);
text(s51,'body-outcome','更多侧面图像参与共享表示优化，补充后脑与衣物侧面的外观约束。',810,366,404,108,26);
text(s51,'body-scene','人物随时间变化，背景固定。左为实拍，中为训练前，右为本轮人物与背景联合渲染。',810,510,404,119,25);
text(s51,'body-evidence','本地实际渲染，2026-10-22。全部观测用于训练。',66,651,1040,25,15,false,'#536674');
s51.speakerNotes.text='主轮证据：'+root+'/run18000/protocol.json\n'+root+'/run18000/result.json\n'+root+'/run18000/root-review.json\n'+root+'/run18000/review/render-check.json\n主模型SHA256 676675cc2f38913ee95e00f4ceb2ba1fb7d488bf6a1fbaa711516d0e69411ec2。19.2万指实际191595有效面片绑定点。人体与背景总450228高斯。原始59观测来自水平绕行，不能说增加了高低角度实拍。此前留出图均加入训练，因此主轮指标只描述训练重建，person L1=0.02275282749，head L1=0.01585944718，mask IoU=0.97199561。图像为未增强原生输出，不使用扩散补图或锐化。虚拟俯仰和距离路径没有GT，不报告不存在的PSNR/SSIM。固定相机时间视频直接使用59时刻估计形变，查看器PLY为时间23的静态快照，不能用PLY切换冒充动态播放。当前模型未达到完整新视角质量验收。';

const s52=p.slides.add({background:{fill:'#ffffff'}});
text(s52,'title','表面高斯在未知俯仰下的实际对照',66,42,1148,67,42,true);footer(s52,52);
await picture(s52,'固定20度俯仰三组实际渲染',pilot+'/training/pitch-15.png',66,133,1148,420);
text(s52,'effect','局部脸部、头皮与衣领更连续。三组沿同一相机路径渲染，人物时刻、点数与运动保持一致。',66,579,1148,74,27);
s52.speakerNotes.text='图片为本地真实20度俯仰路径第15帧，左主3DGS18000步，中原生2DGS纯图像对照1180步，右2DGS几何约束1180步。两个2DGS臂从同一主模型转换，使用同一随机59视角顺序，均执行1180次backward与optimizer.step。完整连续视频位于 http://127.0.0.1:8870/body-dynamic.html#surface 。单帧只展示部分局部连续性，不表示所有视角改善。缺陷和采用判断在最后一页。\n2DGS候选原生渲染不能直接用普通3DGS PLY浏览器验证等价性；因此只发布真实候选图片与视频，主查看器继续保留原3DGS模型。\n来源：'+pilot+'/training/result.json\n'+pilot+'/review/result.json';

const s53=p.slides.add({background:{fill:'#ffffff'}});
text(s53,'title','当前不足与下一轮采用条件',66,42,1148,67,42,true);footer(s53,53);
await picture(s53,'实拍原3DGS与2DGS候选侧身完整对照',pilot+'/review/000058-body-comparison.png',66,143,720,470);
text(s53,'quality-findings','侧身衣服背部仍有缺损。\n候选头发与衣物存在拉丝，实拍纹理也有回退。',820,170,394,138,26);
text(s53,'acceptance','保留当前动态模型。\n下一轮隔离切向尺度、法向和深度权重，检查连续俯仰及侧身轮廓。',820,365,394,157,26);
text(s53,'current-boundaries','未知俯仰没有实拍真值；脚边背景缺口仍在。当前候选未替换查看器，也未部署到 00。',66,626,1148,47,21,false,'#536674');
s53.speakerNotes.text='当前结论不是全部完成。2DGS几何臂相对2DGS纯图像臂，法向误差0.430330降至0.122078，原生NDC深度畸变9.99716e-6降至1.75550e-6。它们是训练内部约束指标，不能证明真实几何精度。训练head L1为0.0191474，相比主3DGS0.0158594回退约20.7%；相比纯图像2DGS0.0181399回退约5.6%。全部59图均参与训练。\n尺度相对父模型95分位为1.92496，局部旋转95分位30.2656度。纯图像臂分别1.58138与24.0236度。这提示覆盖/方向自由度需要隔离检查，但尚不能从相关性断定拉丝原因。下一轮先只读改变切向尺度或法向权重，检验路径与真实图，再决定受限短程对照，不盲目扩大优化预算。候选保存用于必要对照，未按最佳发布。冻结父模型和动态参数，不再开眼镜专项。\n来源：'+pilot+'/root-review.json\n'+pilot+'/review/result.json\n'+pilot+'/training/result.json';

for(const row of inspection)if(row.kind==='textbox'&&/^\d+\s*\/\s*51$/.test(row.text))set(row.id,row.text.replace(/51$/,'53'));
await fs.mkdir('renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(`renders/${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await png.arrayBuffer()));
 if([1,32,33,40,50,51,52].includes(i)){
  const lay=await p.slides.items[i].export({format:'layout'});
  await fs.writeFile(`renders/${i+1}.layout.json`,new Uint8Array(await lay.arrayBuffer()));
 }
}
const finalDir=path.join(workspace,'shuolee-slide-output');await fs.mkdir(finalDir,{recursive:true});
const candidatePath=path.join(build,'candidate.pptx');await (await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath:path.join(finalDir,'02_有限视角重建优化路线_20261022.pptx'),pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:53,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:path.join(build,'source.pptx'),referenceSha256:'432ff8a99b82200dc37f8a6c42921d13f826a8342658cf629f90b7e163de15ae'},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result,null,2));
