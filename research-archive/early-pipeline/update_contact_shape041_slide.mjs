import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-nlf-shape-range-fresh-041';
const contact=base+'/current/smplx-nlf-contact-fresh-040';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='03d579d003d946944be8cdcbc6168a573dcdd165ccfcd1490bd82728646d6424';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='shape_range_change_no_clear_alignment_gain')throw Error('Review required');
const previous=JSON.parse(await fs.readFile(build+'/visible039-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const beforePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(beforePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{const e=row.elements.find(x=>x.text===before);if(!e)throw Error('Missing text '+before);p.resolve(e.id).text.replace(before,after);};
replace('身体先验与可见表面约束','身体拟合中的体形与接触约束');
replace('比较三维表面位置与可见二维对应对身体配准的作用。','体形范围限制身体的变化，地面接触限制脚底的空间位置。');
replace('左组：实拍、三维约束、可见二维约束。右组：同一世界侧面相机。','左组：实拍、体形范围±1、±3。右组：同一世界侧面相机。');
replace('可见二维约束没有带来明显的整体对齐改善。\n两组从头拟合，关节残差均约 9 像素。',
        '扩大体形范围改变胸腹轮廓，整体配准仍待解决。\n取消接触虽降低二维残差，却增大相对地面的偏离。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two images');
for(const [i,file] of ['source-000045.jpg','fixed-left90.jpg'].entries()){
  const im=p.resolve(images[i].id);
  im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',
              alt:i?'固定23帧同一侧面相机，体形范围±1和±3':'45帧实拍与体形范围±1、±3的两组从头拟合'});
  im.crop={left:0,top:0,right:0,bottom:0};
}
slide.speakerNotes.text=`一、身体先验与成像约束
单目网络提供姿态、表面对应及不确定性，SMPL-X提供平均模板、体形空间、关节回归和蒙皮。真实相机将模型投影到图片。身体拟合同时改变共享体形、根变换、尺度和关节姿态，所以二维投影接近观测时，仍可能存在不同的三维解释。人体模板描述裸人体，宽松外套、裤子和鞋子遮挡了真实表面及关节。把裸人体扩张到衣物轮廓会引入另一种偏差。
当前继续使用已经具备的VPoser、GVHMR和NLF。NLF-L原始59帧输出提供本轮初始姿态与1024点局部三维表面先验。这是59张顺序视频帧，不是59台同步相机。

二、共同输入与从头拟合
使用原始视频对应的59张去畸变图、原始遮罩、SfM相机、许可SMPLX_MALE模板及官方原始NLF输出。全部实验从零体形和新的Adam优化器开始，不加载历史个体拟合、高斯、形变或优化器状态。旧模型只在只读比较中出现。
两轮各两组，每组1200步，共4800次身体更新。前300步拟合根旋转、平移与共享尺度，随后开放共享关节/体形，再开放逐帧有界关节修正。共享关节每轴0.30弧度、逐帧0.10弧度，使用SO(3)旋转组合。相机内参和外参固定。
身体、脚、手二维稳健尺度20、18、10像素，权重1、0.2、0.5。HaMeR只提供局部手指姿态，裁切相机的根旋转不复制到身体。保留人物遮罩包含、局部身体姿态与关节先验、鞋纹相对运动、时序平滑和正深度条件。
NLF表面项将原始非参数点减去骨盆并去除整体朝向，与同索引局部人体顶点比较。稳健尺度0.06模型米，不确定性权重为sigma的负1.5次幂，逐帧归一化为均值1。表面项在300至500步渐增到1。预测是有偏差的先验，不能当成测量真值。

三、绝对脚底高度与相对运动
设地面为nᵀv+d=0，n为单位法向量，v为世界脚底顶点，s为每模型米对应的世界单位。相对高度h=(nᵀv+d)/s，以模型米计。
本地接触损失为左右脚平均的ρ([0.01−h]₊/0.015)+ρ([h−0.07]₊/0.02)，ρ(r)=√(1+r²)−1，方括号正部表示max(x,0)。每只脚在59帧、99个脚底点上平均。它是0.01至0.07模型米的软区间，保留了鞋子厚度可能引入的差别，但区间值不是实测鞋底厚度。
地面从原SfM的背景点估计，1393个候选中1284个内点，距离中位0.00747世界单位。没有独立重力参考，不能宣称该平面就是完全正确的真实地面。检查原NEUTRAL脚底索引与MALE模板脚底选区，每侧93点重合，继承区域相对8毫米阈值的最大边缘差约1.5毫米，没有发现大范围索引错位。对照始终使用相同的每脚99点。
相对脚底稳定比较对应顶点的时间变化，尺度0.025模型米、权重0.4。鞋纹项从原始空间特征经踝关节运动传递到各相机，与原始像素比较，尺度10像素、权重0.3。相对运动可以一致，绝对高度仍有歧义，所以不能用鞋纹项自动替代地面接触项。

四、接触消融的实际效果
040只比较绝对高度项权重1与0，其他项、预算和输入一致。关节投影中位数9.007变为8.730像素，95%分位34.200变为35.092像素。未优化鞋纹诊断中位1.393变1.291像素。这些量都来自当前相机与不确定检测观测，不是独立未知视角准确率。
保留高度时，左右脚底采样点低于估计平面的比例为0.63%和0.79%。移除后为30.06%和36.59%。比例包含所有59帧每脚99个点，不能解释为受试者真实穿地的概率。模型尺度变化也参与了结果。
原始45帧叠图、全部59帧联系表、固定23帧正侧面与上下25度，以及0、23、45帧脚部近景显示，取消接触没有带来明显站姿改善。较低二维误差不能支持删除该约束。本地后续保留高度项。

五、体形范围与正则强度
SMPL-X体形向量β有10维。原先五个系数逼近±1，局部导数也显示向边界外的下降方向，因此有必要核对是否是参数范围限制了身体表达。
041比较B=1和B=3，参数化为βᵢ=B tanh(qᵢ/B)。两组q和β从0开始，在0处dβ/dq=1。这样扩大范围时，初始物理灵敏度保持一致。正则固定为Lshape=0.04×mean(β²)，直接约束实际体形系数，没有除以B²，也没有随范围降低先验强度。B是本地优化范围，不是经校准的统计置信区间。
其余损失、姿态界限、脚底高度、相机、表面约束和1200步预算一致。两个041起点逐数组相同。±1控制与040控制的最大顶点坐标差0.000296世界单位，最大关节投影分量差0.06415像素。物理目标等价的两次数值结果接近，未声明逐字节一致。

六、体形变化与实际对齐
扩大到±3后，5个系数超出旧±1范围，最大绝对值2.106，0个系数接近新界限。体形空间确实受到旧范围限制，但这不证明放宽后更准确。
±1与±3两组关节投影中位数9.011和8.939像素，95%分位34.202和32.705像素。未优化鞋纹诊断1.392与1.412像素。固定23帧左膝模型弯曲14.02变13.44度、右膝9.42变8.84度，没有独立解剖真值。
当前页左图显示真实45帧及两组网格，右图来自同一世界侧面相机。胸腹、腰部和四肢轮廓随体形空间发生变化，整体肢体配准没有明显突破。所有59帧联系表及正侧面、俯仰均已对照。原生模板重放误差为0仅确认输出接口一致。

七、适用边界与后续方向
取消接触和扩大体形范围两项均不作为整体几何通过。没有新高斯训练、查看器模型替换或00部署。完整身体需要结合真实原图、连续视角和局部几何判断，不能靠较小二维误差或更直的膝盖自动验收。
停止重复扫描接触/表面权重和体形范围。下一步用现有VPoser V02_05对原始NLF身体姿态进行原生编码与解码，固定相机、模板、根、尺度和手指，核对其旋转与投影变化。该检查判断先验是否兼容，不将解码结果当成真实人体。此前VPoser实验使用旧SMPLer-X起点且未通过，不直接重复。只有原图支持新的姿态方案时才建立从头对照。
几何验收后，完整高斯实验使用20万新点和18000步。未通过的模型与重复初始化在证据保存后清理，原始图像、相机、官方预测、许可模板、源代码及必要对照保留。

八、来源
NLF官方代码：https://github.com/isarandi/nlf 。SMPL-X官方代码：https://github.com/vchoutas/smplx 。VPoser来自human_body_prior，当前使用已核验的V02_05。模板由用户提供。NLF权重的默认许可限制非商业研究，本轮仅本地方法验证。
本地接触证据：${contact}/contact-input-audit.json、review/comparison.json、phase-conclusion.json。
本地体形证据：${phase}/bound1/protocol.json、bound3/protocol.json、review/comparison.json、phase-conclusion.json。照片和网格均来自真实输入与数值计算，无生成式图像补全。`;
await fs.writeFile(build+'/contact-shape041-edited-layout.json',JSON.stringify((await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']})).records,null,2));
const draft=build+'/contact-shape041-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_体形与接触_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
  pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
  verifyArtifactToolImport:true,receiptPath:build+'/contact-shape041-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/contact-shape041-final-renders';await fs.mkdir(renderdir,{recursive:true});const checks=[];
for(let i=0;i<55;i++){
  const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
  await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
  const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
  if(i<54&&!unchanged)throw Error('Unintended change '+(i+1));checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/contact-shape041-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),
  changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,unchanged_first54:true,review_required:renderdir+'/55.png'}));
