import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-mask-foreground-fresh-070';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='398a5464224fb1a9f6c8f719b7a2a7c82004af9d4d40c64eef7c80f78f50ccf5';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Report changed externally; preserve it');
const decision=JSON.parse(await fs.readFile(run+'/review/visual-decision.json','utf8'));
if(!decision.actually_inspected)throw Error('Actual review is required');
const numeric=JSON.parse(await fs.readFile(run+'/review/numeric-screening.json','utf8'));
const cameraDecision=JSON.parse(await fs.readFile(base+'/current/static-camera-registration-071/review-decision.json','utf8'));
const previous=JSON.parse(await fs.readFile(build+'/static-deck-receipt.json','utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const text=(i,k,value)=>{const e=snap[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
const note=(i,value)=>{p.resolve(snap[i].id).speakerNotes.text=value;};
const img=async(i,k,path)=>{const e=snap[i].elements.filter(e=>e.kind==='image')[k],v=p.resolve(e.id);v.replace({blob:await fs.readFile(path),contentType:path.endsWith('.png')?'image/png':'image/jpeg',fit:'contain',alt:'实际同相机人物重建对照或数学公式'});v.crop={left:0,top:0,right:0,bottom:0};};
text(19,3,'逐帧相机先由真实背景特征配准，再约束共享静态高斯。\n人物掩码内学习颜色；轮廓、法线与跨视图一致性约束几何。');
note(19,(snap[19].notes?.text||'')+'\n070更新：颜色支持范围改为mask>0.99，L1按有效人物像素与RGB通道归一化。SSIM仅对整个11×11窗口均在前景内的中心求均值。原全图alpha L1与外部系数0.1、几何项和相机不变。这里保留064的同一PGSR适配，未叠加069双背景方案。原数据、fresh初始化逐数组、相机、优化选项和完整采样顺序的核对见person-static-mask-foreground-fresh-070/controlled-protocol.json。');
text(49,0,'人物掩码内部的颜色监督');
await img(49,0,build+'/mask-formula.png');
text(49,3,'颜色由人物区域监督\n轮廓由透明度约束\n保留原始掩码');
text(49,4,'L₁按有效人物像素归一化。\n边缘人物像素仍参与颜色学习。');
text(49,5,'SSIM使用完整前景窗口。\n局部统计排除外部空白颜色，\n透明度继续约束整个图像。');
text(49,6,'单项对照：仅改变颜色监督范围与归一化，其他条件与原静态模型一致。');
note(49,'本页对应070的精确定义。A*为原始分割mask双线性缩放后的值，OmegaF={p:A*(p)>0.99}。L1对有效像素的三个RGB通道平均，保留所有满足阈值的边缘像素。OmegaS要求整个11×11窗口都在OmegaF；SSIM每通道分别计算再平均，Gaussian窗口sigma=1.5。仅SSIM中心需避开边界，没有侵蚀或改写用于alpha监督的mask。约0.75%的软边界不参加RGB目标但仍参加原全图alpha损失。C*保持064的RGB缩放与合成目标，未同时改变数据。\n只读诊断：175帧扩展ROI中69.64%为空白，6固定已见视图中它们贡献11.42%的L1。末态梯度只证明当前机制存在，不能还原训练全过程。预检随机改变外部预测/目标色，loss与内部梯度完全不变，外部图像梯度严格为零。\n来源：current/static-white-edge-audit-068/mask-photometry/diagnosis.json、regression-check.json；070/analysis-before-training.json，公开PGSR源码及本地shuolee_static.py。');
text(50,0,'前景颜色监督的实际对照');
text(50,1,'同一观察相机与像素裁片：实拍原图、完整ROI监督、人物内部监督。');
await img(50,0,run+'/review/head-023.jpg');
text(50,5,decision.ppt_effect);
text(50,6,'两轮均从20万新初始点训练18,000步。全局作用与不足汇总在末页。');
note(50,JSON.stringify({decision,numeric},null,2)+'\n本页实际图来自review/head-023.jpg，三列同像素范围。GT照片与模型图之间没有分别居中、拉伸或几何变换。六个固定观察视图均参与训练，未知视角没有真值。');
text(51,2,decision.ppt_summary);
await img(51,0,run+'/review/backgrounds/below25-064-bg0.00.png');
await img(51,1,run+'/review/backgrounds/below25-070-bg0.00.png');
text(51,4,'同一仰视相机：左为原静态模型，右为人物内部颜色监督。');
text(51,5,decision.ppt_limit);
note(51,JSON.stringify({decision,numeric,cameraDecision},null,2)+'\n按预先协议评价，不能只看白色变暗。没有未知视角GT，六视图误差为已见分割观测诊断。071补齐116帧背景对应配准：EPnP-RANSAC无插值初值，LM内点优化，原59相机、K及世界坐标不变；ID取模划分60%拟合、20%开发验收、20%仅报告。175帧均有观测相机，留出背景中位数0.992→0.647原图像素，p90为2.325→1.940。固定070模型换新相机，轮廓距离仅2.217→2.192训练像素，头部无同步改善，因此尚未证明修复白边。下一步仅改变相机输入，从头拟合/初始化/训练，当前尚未执行。历史069未通过且同时改变三项，只保留必要对照。动态路线废弃，本轮不部署00。');
const changed=[20,50,51,52];
const draft=build+'/mask-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_相机配准核验.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/mask-camera-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));const renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(changed.includes(i+1))await fs.writeFile(build+'/mask-final-'+(i+1)+'.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unintended rendering change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:changed.includes(i+1)});
}
await fs.writeFile(build+'/mask-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:changed,renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed,canonical_replaced:false}));
