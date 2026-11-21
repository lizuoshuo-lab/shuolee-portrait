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
const expected='ada0c18063ac1c763930cd2d96a4d5578f6906bf3486c786939ed0e98de55f3c';
const sha=async f=>crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex');
if(await sha(source)!==expected)throw Error('Canonical source changed');
const run5=base+'/current/gaussianavatars-smplx-dense-alignment-fresh-005';
const result=JSON.parse(await fs.readFile(run5+'/gaussian-run/result.json','utf8'));
if(result.step!==18000||result.quality_accepted!==false)throw Error('Unexpected training state');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count');
const inspection=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile('dense-alignment-source-inspection.ndjson',inspection.ndjson);
const row=inspection.records[0],slide=p.resolve(row.id);
const set=(old,value)=>{const e=row.elements.find(e=>e.text===old);if(!e)throw Error('Text not found: '+old);p.resolve(e.id).text=value;};
set('三维姿态一致性与可靠观测','全身姿态拟合与几何验收');
set('共享身体姿态并限制逐帧变化，从20万个新点训练1.8万步。','关键点、可见表面颜色和人物轮廓共同约束姿态，再从20万点训练1.8万步。');
set('共享姿态减轻躯干倾斜，单独置信筛选未改善手脚重影。','手腕重投影误差由7.14降至6.40像素，脚踝由24.38降至23.06像素。');
set('关键点需核对关节定义、遮挡和分数尺度；手脚及未知俯仰仍未通过。\n后续检查稠密图像约束下的姿态配准，所有新实验继续从头训练。','裸网格整体姿态仍错误，本轮判定为拟合失败，局部误差下降不能替代几何验收。\n先核验骨盆、躯干和四肢的三维姿态，再从头训练高斯。');
const imageRow=row.elements.find(e=>e.id.startsWith('im/'));
const img=p.resolve(imageRow.id),frame=img.frame;
img.replace({blob:await fs.readFile(run5+'/comparison/dense-method-summary.png'),contentType:'image/png',fit:'contain',alt:'实际固定姿态、相同相机对照：共享姿态和加入表面约束后的侧面与俯视。'});
img.frame=frame;img.crop={left:0,top:0,right:0,bottom:0};
slide.speakerNotes.text=`三维姿态配准与表面约束

问题定位：001在逐帧照片对应的相机和人体姿态下可以获得清楚的渲染，但固定同一个人体时刻再移动相机，就暴露手臂前伸、躯干倾斜与局部重影。高斯训练前的裸SMPL-X网格已有相同姿态错误。帧号、相机矩阵和网格绑定未发现错序。固定姿态原生渲染与标准PLY一致，整体坐标转正无法修复局部关节错误。这里确认的是拟合存在姿态歧义，不将某一种先验直接认定为真实三维答案。

基本表示：身体姿态 θ_t = θ_shared + δθ_t，用共同体态约束各时刻，并限制逐帧变化。保留每帧真实相机和整体人体运动。59张照片来自水平顺序视频，各时刻不同，不能当成59台同步相机。002先从中性模板拟合2400步，再从20万个新高斯训练18000步，减轻了明显的倾斜和前伸，但袖口、手脚与未知俯仰没有通过质量审阅。

观测校验：二维关键点响应分数不是可见概率。单独用两轴较小SimCC响应及相对归一化筛选观测的003对照未改善实际渲染。004按VHAP参考排除有语义歧义的二维髋部点，统一评价相同关节后也未优于002，因此未进入高斯训练。不能通过排除难点或降低内部损失制造改进结论。

005方法：本轮重新从中性SMPL-X、零形状和新姿态参数开始，重新执行2400步关键点拟合。没有读取002人物参数作为初始化。每个表面顶点设跨视角共享RGB颜色，初值0.5，先做300步颜色预热，再做900步有界几何和颜色联合优化。该新粗拟合与002参数哈希相同，是确定性重新计算的结果。其后高斯训练使用新的200000点、新颜色及新优化器，运行18000步，最终190728点。

优化目标：在原关键点目标上加入3倍可见表面颜色残差和0.5倍人体包含约束，同时使用0.03倍邻接颜色平滑、0.02倍增量先验、0.015倍时间平滑。颜色项在实际可见的网格表面比较共享顶点RGB插值与原图，躯干权重1、手部3、足部2。裸模型缺少头发与眼镜几何，所以头部不参与新增颜色项，原头部关键点约束保留。这里没有新增眼镜专项。

人物轮廓的含义：裸人体应位于穿衣人物的遮罩内，外部距离超过0.5个低分辨率像素后施加单边惩罚。不能把裸皮肤强拉到宽松外套或裤子的轮廓上，也不能把轮廓包含关系当作内部深度真值。渲染采用640乘360像素网格，几何在同一真实相机下投影。新增稠密渲染器的像素中心与透视插值通过独立回投检查，最大误差约0.000122像素。

几何边界：共享身体和手部轴角增量各限制0.12弧度，逐帧身体0.02、手部0.03弧度。整体旋转共享0.03、逐帧0.02弧度。平移共享0.01、逐帧0.003模型米，形状系数增量0.3，对数尺度增量0.02。这些是本地实验设置，不是论文默认最优值。模型尺度也不等于经测量的真人尺度。

拟合结果：相同59帧、相同点集下，手腕平均重投影误差7.1445降到6.3964像素，脚踝24.3842降到23.0556像素，共同15个身体点的中位误差9.9964降到9.2339像素。原始分辨率下投影点超出人物遮罩2像素的比例，手部1.782%降到1.734%，足部5.255%降到4.731%，躯干反而从2.850%升到3.092%。这不是所有部位一致改善。

高斯实际效果：相同六张训练照片局部MAE，002与005手部为0.0142404和0.0140922，足部0.0149371和0.0142472，脸部0.0146024和0.0147604。对应手部下降约1.04%、足部下降约4.62%、脸部升约1.08%。这些只是训练照片区域误差，不是测试集或新视角真值。005多了900步几何优化和300步颜色预热，所以也不能仅凭此对照分离新增损失与额外优化预算的影响。

图像说明：四张图来自002与005原生渲染，同一人体时刻33、同一世界相机。前两列侧面，后两列俯视，使用共同裁剪与缩放，没有改变模型形状。另存的六方向对照以及72帧、6秒相同路径视频用于诊断。已查看六方向、手部训练区域和视频9个时间采样，未宣称全部连续帧通过。两版仍有袖口、手脚碎片和未拍摄角度伪影，005不替换当前002候选。

解决的部分与不足：本轮把稠密表面约束接入了从零拟合流程，局部二维误差下降，但裸网格整体姿态依然错误。用户明确指出躯干、骨盆和四肢的形态不正确，因此本轮判为拟合失败，002也只保留为必要诊断对照，不能称可用修复版。共享纹理仍可能吸收位置误差，遮挡点和衣物也不服从裸人体表面。下一步优先SMPLer-X全身三维姿态推理，重新从原图建立独立身体初始估计并核对相机坐标。先通过原图网格叠加、固定姿态的正侧背面和三维关节关系检查，再开始新的高斯训练。HaMeR和Mano2Smpl-X仅为后续局部辅助，不以单独修手代替全身姿态修正，不直接接受网络预测为真值。

公开代码边界：GaussianAvatars_SMPLX公开提交e44c1a6缺少训练入口和SMPL-X高斯模型类。本次是使用其公开原语补齐的本地适配，不称作者完整实现复现。VHAP原版针对头像，SMPL-X分支的轮廓项处于注释状态，本轮单边包含损失是本地适配。

来源：
https://github.com/jmaraval/GaussianAvatars_SMPLX
https://github.com/ShenhanQian/VHAP
https://github.com/jmaraval/VHAP_smplx/blob/8980d4948d91b05ff96ae91bae29e3f15a238f5e/vhap/model/tracker_smplx.py
https://github.com/open-mmlab/mmpose/blob/main/mmpose/codecs/utils/post_processing.py
https://github.com/geopavlakos/hamer
https://github.com/VincentHu19/Mano2Smpl-X
https://github.com/MotrixLab/SMPLer-X
https://github.com/MotrixLab/SMPLest-X

本地实测：${run5}/dense-fit-protocol.json、fit-review.json、native-containment-evaluation.json、comparison/metrics.json、gaussian-run/standalone-audit/audit.json。
`;
const candidatePath=path.join(build,'dense-alignment-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_全身拟合判定.pptx';
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'dense-alignment-verdict-validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));await fs.mkdir('dense-alignment-final-renders',{recursive:true});
const parity=[];
for(let i=0;i<q.slides.items.length;i++){
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 const name=String(i+1).padStart(2,'0')+'.png',out='dense-alignment-final-renders/'+name;
 await fs.writeFile(out,new Uint8Array(await png.arrayBuffer()));
 if(i<54)parity.push({slide:i+1,identical:await sha(out)===await sha('alignment-final-renders/'+name)});
 if((i+1)%10===0)console.log('Final renders '+(i+1)+'/55');
}
if(parity.some(v=>!v.identical))throw Error('An unchanged slide differs visually');
await fs.writeFile('dense-alignment-first54-parity.json',JSON.stringify(parity,null,2));
await fs.writeFile('dense-alignment-final-inspection.ndjson',(await q.inspect({select:'$.slides[54]',include:['id','index','elements.text','notes.text']})).ndjson);
console.log(JSON.stringify({finalPath,finalSha256:await sha(finalPath),changedSlide:55,slides:q.slides.items.length,first54Identical:true,receipt}));
