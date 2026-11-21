import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='7fa7c9f94def87abfd9dc82d3eda3424303d8289befe05a3ef08e10347fb165d';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
if(await hash(source)!==expected)throw Error('Canonical deck changed.');
const conclusion=JSON.parse(await fs.readFile(base+'/current/smplx-gvhmr-prior-fresh-019/phase-conclusion.json','utf8'));
if(conclusion.state!=='independent_prior_reviewed_alignment_pending')throw Error('Review required.');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count');
const inspected=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile(build+'/gvhmr019-slide-before.json',JSON.stringify(inspected,null,2));
const row=inspected.records[0],slide=p.resolve(row.id);
const replacements=new Map([
 ['身体先验与运动一致性','时序人体先验与场景配准'],
 ['关节姿态、场景接触与时间一致性共同约束人体，着地点稳定仍需配合全身几何。','时序先验联合图像特征、二维关节和相机运动，提供独立的人体姿态预测。'],
 ['同一时刻、同一侧面相机：延长拟合后膝部略有变化，整体姿态仍未通过。','同一时刻、相同侧视方向：新预测屈膝较小；各模型仅平移到骨盆中心。'],
 ['脚部纹理传递残差由43.36降至7.32像素，整体姿态仍未通过。\n后续用时序人体预测检查髋、膝、踝与整体位置的联合解释。','真实相机叠图仍有头部、手腕和脚的位置偏差，尚未完成场景配准。\n下一步固定新先验的局部姿态，从原始观测重新求尺度与整体变换。'],
]);
for(const [before,after] of replacements){const e=row.elements.find(e=>e.text===before);if(!e)throw Error('Missing anchor '+before);p.resolve(e.id).text.replace(before,after);}
const imageElement=row.elements.find(e=>e.id.startsWith('im/'));
if(!imageElement)throw Error('Missing image');
const im=p.resolve(imageElement.id);
const geometry=im.geometry,borderRadius=im.borderRadius,rotation=im.rotation;
await im.replace({blob:await fs.readFile(base+'/current/smplx-gvhmr-prior-fresh-019/review/geometry-left90.jpg'),contentType:'image/jpeg',fit:'contain',alt:'实际网格侧面：旧5000步拟合、独立GVHMR原始预测、官方后处理；骨盆居中，不表示世界配准通过'});
im.frame={left:(1280-382*1560/826)/2,top:174,width:382*1560/826,height:382};
im.crop={left:0,top:0,right:0,bottom:0};im.geometry=geometry;im.borderRadius=borderRadius;im.rotation=rotation;im.fit='contain';
slide.speakerNotes.text=slide.speakerNotes.text.replace('该模型尚未在本素材运行，没有新的实验成绩。','此前尚未运行；本轮019已完成独立推理，当前结论见下文。')+`

当前增量019：独立时序人体预测（2026-10-23）

研究问题：继续增加身体拟合步数只能有限改变膝部，故用独立时序模型检查新的关节解释，而不是把旧个体拟合继续微调。模型为GVHMR，官方代码ee960bb6e2ea2d381aa97f08e9b71ef320b624b1。VPoser V02_05仍完整保存在本地并通过CUDA核验；下载先验不等于当前人体已经对齐。

输入与机制：HMR2.0a提取每帧1024维人物图像特征，ViTPose-H产生17个二维关节点及响应。人物框和真实图像内参提供透视条件，相邻世界到相机旋转的相对变换按官方六维旋转表示输入时序Transformer。网络预测身体局部关节、共享体形、相机空间根朝向与位移，以及重力-视角坐标中的根朝向、局部平移速度和静止关节置信度。人体由SMPL-X生成；没有把SMPL手模型或HaMeR权重冒充身体特征模型。

默认demo以30fps重编码视频且估计焦距。本素材实际为24.029053fps，容器解码177帧，每帧41.616ms。59张标定图对应零起点索引1、4、7直到175。这里限制在标定覆盖区间，按30fps最近邻取218个输入时刻，实际175个不同实拍帧，重复43个位置。没有生成新的俯仰图像。相机旋转仅在59个标定节点之间做球面插值，相机中心线性插值，最终按选中图像的真实时间关联；插值相机不是真值。旧记录的176帧是读取到最后标定索引175时的数量，本轮完整解码数为177。

图像原尺寸1520×2704，使用原SIMPLE_RADIAL标定去畸变到1514×2694，fx=fy=1461.4907695px，cx=757px，cy=1347px。59张实际校准图与重新解码去畸变结果逐像素一致。人物框来自原始二维遮罩，在实拍节点间插值，按官方规则扩大1.2倍；256×256 RGB裁切经ImageNet均值方差归一化，网络中央宽192像素。ViTPose启用左右翻转热图平均和UDP坐标解码。两大图像模型均批量4处理；没有移除背景或读取旧人物参数。相机绕人物运动，static_cam=False。

官方模型加载：GVHMR247个状态张量、HMR2的backbone和smpl_head、ViTPose全部严格匹配，无缺失或额外学习张量。核心权重约164MB，图像编码器约2.71GB，关键点模型约2.55GB。来自camenduru/GVHMR公开镜像固定版本21b32d5389e2e59c0737d4c4095bbc0b8c23f66b，三个SHA256与LFS记录匹配，另一公开镜像的LFS摘要相同；没有作者独立发布的哈希，因此仅声明传输/镜像一致性。weights_only方式加载，未执行检查点任意代码。

运行环境：隔离venv，复用现有Torch2.7.1/CUDA12.8和RTX5080支持；不修改原有身体训练环境。官方推荐旧Torch2.3/CUDA12.1不直接套用到5080。Hydra1.3.2、timm0.9.12、PyTorch-Lightning2.3、PyAV13与官方Python源码结合。有标定输入替换了默认YOLO/视觉里程计输入，因此这是本地适配，不声称官方demo完全原样复现。

对照一为原始网络输出postproc=False；对照二为官方postproc=True、static_cam=False。后处理先根据静止关节置信度修正全局平移，再对髋/膝/踝等做CCD逆运动学；手、脸仍为默认参数。其最低关节归零与预测地面是启发式，不等于本场景实测地面。两个结果独立保存，没有后处理覆盖原始预测。

本页图固定原始时刻23。三组使用相同观察方向、人体模板米制单位，分别只把骨盆平移到中心；旧场景模型除以原拟合的世界尺度转换回模板单位。没有分别旋转到更好看的姿态，也没有按身高归一化。因此可比较局部形态，但不能用本图证明场景世界配准。旧结果仅在推理完成后被读取作比较，不是初始化。

左右膝几何弯曲：018控制5000步20.01/21.05度，019原始预测13.69/10.53度，官方后处理13.47/9.19度。原始侧面网格中的屈膝确实减轻，关节真实角度未知。这不是准确率，也不能断言更直一定正确。59个对应时刻的原始预测膝弯曲中位数13.97/9.92度，旧拟合17.67/15.91度。

真实标定相机叠图没有人为调整位姿：正面整体较接近，右侧45/58的头部、手腕与脚仍偏离实拍，默认张开的手也没有表达握拳。第45帧裸网格超出衣服/鞋掩膜的比例，旧控制1.68%，新原始5.19%，后处理5.31%；该数字不是真实裸人体误差，因为掩膜是衣服、鞋和头发的外轮廓。已实际审阅裁切、正面/侧面网格和00/23/45/58叠图；其他生成图片及完整时间序列未宣称全部通过。

采用范围：保留GVHMR作为新的候选先验，尚不采用为最终身体，不训练新高斯、不部署00。本轮个体拟合0步、高斯0步，只有官方预训练推理。下一轮020从原始观测、许可SMPL-X模板和该官方独立预测建立尺度与场景根变换。先固定局部骨架检查标定相机下的整体配准，再做受控的局部修正与手部融合。禁止从018旧个体结果初始化，也不把示意骨盆居中展示当成世界对齐。

源代码许可：GVHMR默认教育、研究及非营利用途；不因用户对GaussianAvatars/模板已有商业授权就推断本项目也已获公司部署授权。当前为本地研究比较，部署前再核对具体授权。

官方来源：https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1。
证据：${base}/current/smplx-gvhmr-prior-fresh-019/phase-conclusion.json；preflight.json；crop-receipt.json；prediction-receipt.json；review/review-manifest.json。
`;
const candidate=build+'/gvhmr019-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261023_时序先验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/gvhmr019-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/gvhmr019-final-renders',{recursive:true});
for(let i=0;i<55;i++){const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});await fs.writeFile(build+'/gvhmr019-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(build+'/gvhmr019-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),candidate,changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
