import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='d42ffc56c3b3d02661465d22a297488ca50dc4b9c3bcbe076b8221094be32bde';
const phase=base+'/current/smplx-observation-uncertainty-fresh-026';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
if(await hash(source)!==expected)throw Error('Canonical deck changed');
const conclusion=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(conclusion.state!=='detector_weighting_not_adopted')throw Error('Review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count');
const inspected=await p.inspect({select:'$.slides[54]',include:['id','index','layout','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile(build+'/observation026-slide-before.json',JSON.stringify(inspected,null,2));
const row=inspected.records[0],slide=p.resolve(row.id);
const before='纹理残差：足部 30.65 降至 1.36 像素，躯干与腿部 24.23 降至 2.97 像素。\n共享姿态仍未修正侧面形态，需先核验二维关节点。';
const after='纹理残差：足部 30.65 降至 1.36 像素，躯干与腿部 24.23 降至 2.97 像素。\n二维观测重加权未修正侧面形态，下一步比较独立的人体预测。';
const element=row.elements.find(e=>e.text===before);
if(!element)throw Error('Missing text anchor');
p.resolve(element.id).text.replace(before,after);
let notes=row.notes.text;
const oldFuture='下一轮先在实际照片上核验肩、肘、膝的定义、遮挡和裁切/像素坐标，再设计带观测不确定性的从头对照，不直接把任一检测器当真值。';
if(!notes.includes(oldFuture))throw Error('Missing observation plan anchor');
notes=notes.replace(oldFuture,'026已完成实际照片、时间对应、裁切和像素坐标核验，并完成观测均值与方向不确定性对照，详见下一节。');
const newSection=`十、二维观测分歧与方向不确定性
026首先复查全部59帧。源图散列与时间索引一致，按官方流程重建的ViTPose裁切与已有输入逐像素一致，RTMW仿射逆变换最大误差约9.1×10⁻¹³像素。当前RTMW接口接收RGB，与其导出配置相符；没有发现通道、裁切或时间对应错误。半像素投影约定无法解释20至26像素的肘膝分歧。原图显示宽松衣物和远侧遮挡，因此两套检测都不是关节真值。
三臂分别使用原RTMW点、RTMW与ViTPose的等权均值、相同均值加方向性不确定度。每臂独立从原始GVHMR预测重建初始化，新建优化器，各1200步。起始参数逐项相同，原观测臂的全部输出数组与024身体纹理臂散列完全一致。其余相机、遮挡和置信权重、身体/足部纹理、脚底接触、局部手型、姿态与体形约束、阶段及学习率均保持一致。
对于两套二维点u_A、u_B，定义平均点ū=(u_A+u_B)/2、分歧δ=u_A−u_B。以σ₀=20像素构造Σ=σ₀²I₂+δδᵀ，再将投影残差变为r=Σ⁻¹ᐟ²(π−ū)，沿分歧方向降低约束强度，垂直方向保留基础尺度。对每个分量使用ρ(r)=√(1+r²)−1，按原观测权重归一化。Σ是启发式不确定性近似，不是经过校准的统计协方差；共同检测偏差不会被平均消除。
未参与新增损失的身体纹理中位残差依次为2.971、2.815、2.697像素；足部依次为1.312、1.370、1.439像素。对原RTMW点的残差为10.063、10.683、10.327像素，对ViTPose为13.473、11.366、11.393像素。不同观测目标的拟合误差不能直接充当三维准确率。参考时刻23左膝弯曲角依次13.25、12.87、13.89度，右膝6.70、6.32、7.32度；这些数值描述模型，不是真值角度。
已查看真实23/45帧叠图、固定左右90度和六时刻正/侧面快照。人体形态没有明确改善，三臂均不采用，未更新高斯或部署00。本轮3600次身体更新独立于024/025的4800次。
下一条独立参考为SAM 3D Body及官方MHR到SMPL-X转换。已核实源码支持输入现有边界框、遮罩、真实相机内参；先检查原预测在相机中的叠图，再检查单位、坐标方向、转换误差和共享体形。官方权重需要Hugging Face账号访问批准，当前未下载、未运行此素材。不能将论文分数当成本项目效果。

十一、目前解决的问题与不足`;
if(!notes.includes('十、目前解决的问题与不足'))throw Error('Missing closing section');
notes=notes.replace('十、目前解决的问题与不足',newSection);
notes+=`\n026代码与证据：${phase}/phase-conclusion.json、audit/audit.json、review/review-manifest.json。refine_observation_uncertainty_fresh.py与review_observation_uncertainty_fresh.py为本地适配。\nSAM3DBody：https://github.com/facebookresearch/sam-3d-body/tree/b5c765a0d89d789985e186d396315e7590887b94 。MHR转换：https://github.com/facebookresearch/MHR/tree/877cba5ee61a5f77d1457bf4de93a66695132bd9/tools/mhr_smpl_conversion 。相关预印本仅作参考：https://arxiv.org/abs/2604.01447 。`;
slide.speakerNotes.text=notes;
const candidate=build+'/observation026-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_观测核验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/observation026-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/observation026-final-renders',{recursive:true});
for(let i=0;i<55;i++){
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 await fs.writeFile(build+'/observation026-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
}
await fs.writeFile(build+'/observation026-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),candidate,changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
