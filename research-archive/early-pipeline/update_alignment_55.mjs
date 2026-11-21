import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=process.cwd();
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='27ca89bb6e03486de349b3660841416986d167df67af873aa1bbd49cc12dffe0';
const sha=async f=>crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex');
if(await sha(source)!==expected)throw Error('Canonical source changed');
const run3=base+'/current/gaussianavatars-smplx-confidence-fresh-003';
const review=JSON.parse(await fs.readFile(run3+'/gaussian-run/root-review.json','utf8'));
if(review.step!==18000||review.overall_quality_accepted!==false)throw Error('Unexpected review');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count');
const inspection=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile('alignment-source-inspection.ndjson',inspection.ndjson);
const row=inspection.records[0],slide=p.resolve(row.id);
const set=(old,value)=>{const e=row.elements.find(e=>e.text===old);if(!e)throw Error('Text not found: '+old);p.resolve(e.id).text=value;};
set('初始化视角与几何继承','三维姿态一致性与可靠观测');
set('相同19.2万个初始点，两组各1,770步，仅早期视角安排不同。','共享身体姿态并限制逐帧变化，从20万个新点训练1.8万步。');
set('头部训练 L1：先8帧再59帧 0.01997，全程59帧 0.01925。','共享姿态减轻躯干倾斜，单独置信筛选未改善手脚重影。');
set('两组头顶较旧模型连贯，仰视脸部与颈部仍有伪影。\n新旧差异包含几何重置等因素，下一轮单独区分几何继承与位移限制。','关键点需核对关节定义、遮挡和分数尺度；手脚及未知俯仰仍未通过。\n后续检查稠密图像约束下的姿态配准，所有新实验继续从头训练。');
const imageRow=row.elements.find(e=>e.id.startsWith('im/'));
const img=p.resolve(imageRow.id),frame=img.frame;
img.replace({blob:await fs.readFile(run3+'/comparison/alignment-method-summary.png'),contentType:'image/png',fit:'contain',alt:'真实侧面渲染：独立姿态网格、共享姿态网格、共享姿态高斯、相对置信筛选高斯。各自在人体坐标系中显示。'});
img.frame=frame;img.crop={left:0,top:0,right:0,bottom:0};
slide.speakerNotes.text=`三维姿态一致性与可靠观测

研究问题：逐帧实拍与渲染拼图看起来合理，但导出固定姿态后手臂前伸、躯干倾斜。相机编号、帧索引、网格绑定与PLY导出经核对没有发现错序；原生固定姿态与PLY渲染一致。在高斯训练之前的SMPL-X裸网格已能看到相同局部姿态错误，所以继续调整查看器整体旋转不能解决。

图像：左一是001逐帧独立拟合的裸网格；左二是002从中性模板拟合的共享体态；右二和右一是002、003的18,000步高斯。均固定人体时刻33，显示各自人体坐标系的侧面，不是实拍深度真值。这一总览使用各模型自身的全局坐标标准化，不据此计算像素改善；严格002/003对照另存为comparison/fixed-pose-comparison.jpg，使用完全相同世界相机。

表示与训练：用共享身体姿态加受限逐帧残差表达序列，保持每帧真实相机。拟合从官方中性模板、零形状和新姿态参数开始，运行2,400步。随后重新采样200,000个表面高斯，以全新外观参数和优化器运行18,000步；002最终191,358点，003最终191,041点。没有从旧断点继承学到的几何、姿态、高斯或优化器。原图、遮罩、相机以及公开预训练检测器输出可复用。59张图属于水平顺序视频，各帧存在运动，不能当成59台同步相机。

观测分数：当前RTMW导出的SimCC峰值不是0至1概率。旧的0.45、0.5、0.55门槛让所有手部观测通过；遮挡手部的检测点也可能落到衣服上。重新读取原模型两个坐标轴的响应，复核59帧检测坐标与原始结果一致。MMPose参考实现取两轴峰值的较小值，现有rtmlib取平均。本地003使用较小值除以该关节全序列第90百分位再裁剪至0至1；这只是相对可靠性启发式，不能解释为可见概率。身体、足部、脸部阈值0.65，手部0.70。除了该门控，保留002共享体态、边界和损失设置。

实测结果：002/003共同六帧手部区域MAE为0.0142404/0.0146316，足部0.0149371/0.0152004，脸部0.0146024/0.0147264，均略升。固定姿态的侧面仍有袖口、手部重影和脚边碎片，未知俯仰头部未通过。所有这些MAE来自训练图像，不能充当未拍摄方向的真实几何成绩。相对分数筛选没有带来明确改善，不替换002本地候选，不部署电脑00。

关节语义：公开VHAP_smplx跟踪代码排除OpenPose髋部标注，注明外观标注与SMPL-X内部股骨头关节的定义不同。本地004从中性模板重新拟合，仅排除对应COCO髋部11、12，用于检验这一语义差异；尚未进行高斯训练。为避免删难点制造指标变好，统一比较三个版本相同15个身体点：002/003/004中位重投影误差9.9964/11.4175/10.1558像素。004的手腕与脚踝误差也没有优于002。因此不能只看其排除髋部后的内部拟合损失降低，就宣称三维配准通过。

解决的部分：共享体态约束减轻了001的明显姿态歧义，解释了逐帧渲染正确与固定身体错误可以同时发生。剩余不足：局部遮挡误检、模板关节对应和深度歧义尚未消除；单独更换置信度与忽略髋部都不是完整解决方案。下一步优先核验多帧轮廓、可见表面和稠密图像残差如何约束姿态，避免继续只扫置信阈值。任何新的训练仍从原图与中性模板开始。

公开代码边界：GaussianAvatars_SMPLX公开提交e44c1a6缺少训练入口及SMPL-X高斯模型类。本次是使用其可用原语补齐的本地适配，不能称作者完整实现复现。

参考：
https://github.com/jmaraval/GaussianAvatars_SMPLX
https://github.com/jmaraval/VHAP_smplx/blob/8980d4948d91b05ff96ae91bae29e3f15a238f5e/vhap/model/tracker_smplx.py
https://github.com/open-mmlab/mmpose/blob/main/mmpose/codecs/utils/post_processing.py
https://github.com/vchoutas/smplify-x/blob/master/smplifyx/fitting.py
https://github.com/zju3dv/EasyMocap/blob/master/easymocap/mytools/triangulator.py
EasyMocap多视角三角化要求同步或恰当运动补偿，不能直接套到当前顺序视频并当成几何真值。

本地证据：${run3}/gaussian-run/root-review.json、comparison/metrics.json；${base}/current/gaussianavatars-smplx-joint-mapping-fresh-004/common-joint-evaluation.json；002/observation-audit/confidence-semantics.json。真实图像和方法记录保留，未通过的模型大文件按清单清理。
`;
const candidatePath=path.join(build,'alignment-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_姿态与观测修订.pptx';
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'alignment-final-validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));await fs.mkdir('alignment-final-renders',{recursive:true});
const parity=[];
for(let i=0;i<q.slides.items.length;i++){
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 const name=String(i+1).padStart(2,'0')+'.png',out='alignment-final-renders/'+name;
 await fs.writeFile(out,new Uint8Array(await png.arrayBuffer()));
 if(i<54)parity.push({slide:i+1,identical:await sha(out)===await sha('initialization-final-renders/'+name)});
 if((i+1)%10===0)console.log('Final renders '+(i+1)+'/55');
}
if(parity.some(v=>!v.identical))throw Error('An unchanged slide differs visually');
await fs.writeFile('alignment-first54-parity.json',JSON.stringify(parity,null,2));
await fs.writeFile('alignment-final-inspection.ndjson',(await q.inspect({select:'$.slides[54]',include:['id','index','elements.text','notes.text']})).ndjson);
console.log(JSON.stringify({finalPath,finalSha256:await sha(finalPath),changedSlide:55,slides:q.slides.items.length,first54Identical:true,receipt}));
