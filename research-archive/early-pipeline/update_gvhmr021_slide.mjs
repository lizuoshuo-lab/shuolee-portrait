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
const expected='e46c82b323095f5c1df4120c5edaafd559791db40b2c64efc7a41bba047106cd';
const phase=base+'/current/smplx-gvhmr-articulation-fresh-021';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
if(await hash(source)!==expected)throw Error('Canonical deck changed.');
const conclusion=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(conclusion.state!=='staged_fitting_reviewed_partial_improvement_not_accepted')throw Error('Review required.');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count');
const inspected=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile(build+'/gvhmr021-slide-before.json',JSON.stringify(inspected,null,2));
const row=inspected.records[0],slide=p.resolve(row.id);
const replacements=new Map([
 ['时序人体先验与场景配准','时序先验与分阶段人体拟合'],
 ['时序先验联合图像特征、二维关节和相机运动，提供独立的人体姿态预测。','固定真实相机，先估计整体尺度与位置，再联合约束关节、手部与体形。'],
 ['同一时刻、相同侧视方向：新预测屈膝较小；各模型仅平移到骨盆中心。','同一时刻、同一世界相机：原拟合 → 时序预测 → 整体配准 → 关节与手部拟合。'],
 ['真实相机叠图仍有头部、手腕和脚的位置偏差，尚未完成场景配准。\n下一步固定新先验的局部姿态，从原始观测重新求尺度与整体变换。','可见关节点残差中位数：30.15 → 22.79 → 8.13 像素（拟合观测）。\n手腕贴合改善；侧面体形和遮挡肢体尚未通过，继续核验。'],
]);
for(const [before,after] of replacements){const e=row.elements.find(e=>e.text===before);if(!e)throw Error('Missing anchor '+before);p.resolve(e.id).text.replace(before,after);}
const imageElement=row.elements.find(e=>e.id.startsWith('im/'));
if(!imageElement)throw Error('Missing image');
const im=p.resolve(imageElement.id);
const geometry=im.geometry,borderRadius=im.borderRadius,rotation=im.rotation;
await im.replace({blob:await fs.readFile(phase+'/review/fixed-left90.jpg'),contentType:'image/jpeg',fit:'contain',alt:'真实网格同一世界相机侧面：此前拟合、GVHMR原始预测、独立整体配准、独立分阶段关节及手部拟合。整体尚未通过。'});
im.frame={left:(1280-382*2080/826)/2,top:174,width:382*2080/826,height:382};
im.crop={left:0,top:0,right:0,bottom:0};im.geometry=geometry;im.borderRadius=borderRadius;im.rotation=rotation;im.fit='contain';
slide.speakerNotes.text=`时序先验与分阶段人体拟合：当前有效方法及证据（2026-10-23）

问题与范围
同一人物的训练视角可以被高斯颜色拟合得较好，但错误的身体载体会让未知侧视、俯仰与导出模型出现变形。本页处理前置的人体配准：整体尺度、根变换、局部关节、体形和手部的联合解释。当前没有重新训练高斯，不把裸网格通过或较低二维误差当作最终人物重建通过。所有新个体实验从原始图像、相机、许可模板和官方预训练输出开始；旧人物参数只在完成后读取作对照。

一、两类身体先验的作用
VPoser V02_05：32维潜变量经固定解码器给出21个身体关节旋转，约束姿态分布；不能决定场景尺度、相机、全局位置、手指或真实衣物。模型已下载，哈希和RTX5080运行核验完成。前序VPoser、可见性、接触及较长预算对照已有记录，整体尚未通过。
GVHMR：HMR2.0a提取图像特征，ViTPose-H提供二维关节，相机相对旋转与透视条件输入时序Transformer。预测局部身体姿态、体形、相机空间根变换及重力坐标运动。默认全局后处理通过静止关节和逆运动学修正运动，但预测地面不能替代本场景的地面证据。本轮采用原始网络输出作独立先验，不用旧个体拟合初始化。

二、输入和坐标必须一致
原视频1520×2704、约24.029fps，完整解码177帧。59张标定图对应原视频零起点索引1、4、7至175，重新解码并按SIMPLE_RADIAL去畸变后，与既有1514×2694图逐像素相同。真实内参fx=fy=1461.4907695px，cx=757px、cy=1347px。先验模型按30fps最近邻输入218个采样位置，只有175个独立实拍帧；重复不增加新视角。插值相机旋转用球面插值、相机中心线性插值，按实际选中图像时间关联。相机绕人物运动，static_cam=False。这里以原遮罩人物框和已标定相机替代默认检测/里程计，是明确的本地输入适配。

三、官方关节点不等于同名骨骼节点
使用官方SMPL-X到SMPL的稀疏映射，再用COCO17表面回归矩阵求图像监督点。它与SMPL-X同名骨架/标志点不同，肩部投影中位差约16–20像素，膝部约11像素。保留官方权重而不擅自归一化；回放官方COCO输出的最大差约5.4×10^-7模板米。网格和骨架回放均小于5×10^-7米，世界转相机回放小于10^-7米。这消除一种定义歧义，但不能证明它是此前失败的唯一原因。

四、从头确定世界尺度和整体变换
由各标定相机及官方预测骨盆位置，稳健求共享初始尺度与近似静止骨盆位置。近似静止只用于初始尺度，不强制全身完全不动。初始尺度3.18005世界单位/模板米。原始局部顶点去骨盆并撤销相机空间根旋转，再按各真实相机逆变换进入统一世界。
在020中，两组独立从同一起点各优化800步，局部身体旋转、体形、手姿态固定。图像组只求每帧根旋转/位移，固定尺度以避免纯图像的尺度规范自由度；场景组加入共享尺度、脚底高度带与脚底位置稳定。固定全部59个真实相机，不逐个移动展示模型。结果可见关节中位残差原始30.15、图像组23.01、场景组22.79像素。整体配准改善部分头部位置，未解决关节和手型。

五、独立的分阶段关节与手部拟合
021重新读取官方原始预测和原始观测，重新生成初始化，未加载020学习数组。1200次更新：1–300仅根变换，301–700开放共享局部姿态与共享体形，701–1200再开放受限逐帧姿态。相机始终固定。局部旋转在SO(3)上组合：R_tj=Exp(0.30 tanh(a_j)+0.10 tanh(b_tj)) R^prior_tj；这是轴角向量每分量界限，不是总旋转角硬上限。共享体形相对原预测中位数每个beta最多改变1。根旋转校正每分量±0.35弧度，位移每分量±0.30倍初始尺度，共享尺度以±0.30对数幅度变化。
HaMeR提供77个可见手样本的局部手指旋转和二维点。遮挡时取最近可见局部手姿态，只作先验；按SMPL-X非平手均值正确相减，不复制裁切相机的绝对手腕根朝向。身体上臂、前臂与手腕继续联合求解，避免只把手指旋转换成握拳却放在错误位置。

六、目标函数和优化细节
总损失：L=L_body+0.2 L_foot+0.5 r L_hand+2 L_contain+0.15 L_pose+0.04 L_shape+L_contact+0.4 L_stable+0.05 L_smooth+L_front。
二维残差用rho(x)=sqrt(1+x²)-1；身体、脚、手分别以20、18、10像素归一化。身体响应按各关节90%分位归一化并平方，面部点再乘0.35，髋部投影不直接约束；侧面被遮挡肘腕和膝踝点显式剔除。脚点也按可靠性及遮挡筛选。手部权重r在300–500步从0升至1。
轮廓只惩罚裸人体伸出衣物遮罩，允许5像素余量，再除15进入稳健损失，不要求裸皮肤填满外套/裤子/鞋。为避免默认开手拉扯身体，包含项排除手部顶点，手由专门的可见点约束。
姿态先验同时约束对原始GVHMR局部旋转的校正（0.15弧度尺度）与局部骨骼位移（0.06模板米尺度）；共享形状变化做均方惩罚。脚底相对估计地面的软高度带1–7厘米，是为鞋与估计误差留下余量，并非真实测量。静止足底中心用2.5厘米尺度约束。相邻姿态/根旋转修正以0.04弧度平滑，位移修正以1.5厘米平滑；没有冻结上身所有真实运动。
Adam：根参数初始学习率0.008，共享0.006，逐帧0.004；第1000步后乘0.4。全参数梯度范数裁剪10，检查有限值与相机前方。总计020的1600次及021的1200次身体更新，0次新高斯更新。硬件RTX5080，Torch2.7.1/CUDA12.8，独立环境，不改原主训练环境。

七、实际效果与采用边界
同一组参与拟合的可见身体观测，原始/整体配准/关节拟合残差中位数30.15/22.79/8.13像素，90%分位52.36/40.54/25.17像素。手腕中位数约3.53/4.62像素；左/右肘仍30.36/24.88像素，其中左肘比刚体配准更差。不能把参与损失的预测点当作独立GT。第23时刻左右膝弯曲13.96/10.06度，接近原始先验13.69/10.53度，没有回到此前约20/21度；真实膝角未知。
已实际审阅59帧联系表、6张完整源图叠加和同一时刻7个世界相机方向。此页四个模型完全使用相同世界相机，没有分别居中、转正或按身高归一化；与019的骨盆居中图不能混为同一种世界配准展示。手腕和部分腿部位置改善，但侧面整体倾斜、体形和遮挡肢体仍未通过。中性裸模板不是衣物外表面，宽松衣物不能确认真实胸腹和膝部形状。轮廓越界也未逐视图改善，例如第11帧场景组1.18%而关节组2.42%。
参数审计：局部旋转校正最大分量0.3903弧度，均值旋转角约9.06度；体形最大分量变化0.9306，接近设定界限。脚底相对估计平面高度约0–7.8厘米，不能当绝对物理量精度。完整固定相机时序回放尚未验收，故仍为候选，不替换查看器或部署00。

八、下一步和最终不足
先核对肘/袖口与头部观测的不确定性、局部关节与根变换的耦合，并检查固定相机时序形变及脚底接触。不默认靠继续增加步数解决，也不强行把裸人体撑到宽松衣物边界。待裸网格多视角和时间一致性通过，再从20万新采样高斯、新优化器做18000步有界训练；禁止恢复旧个体学习结果或180000步任务。未拍摄俯仰缺乏真实观测，单目先验只给合理解释，不保证真实隐藏形状。

源码和权重
https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1
GVHMR核心、HMR2.0a、ViTPose-H三份权重来自固定公开镜像版本21b32d5389e2e59c0737d4c4095bbc0b8c23f66b，SHA256与其LFS记录相符，严格加载无缺失；没有作者独立哈希。GVHMR默认教育研究非营利用途，不因其他模板/项目已有授权推断本仓库公司部署授权。当前仅本地研究比较。
证据：${phase}/phase-conclusion.json；parameter-audit.json；protocol.json；review/review-manifest.json。源码：GVHMR-shuolee/shuolee/register_fresh.py、refine_fresh.py、review_registration.py。
`;
const candidate=build+'/gvhmr021-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261023_分阶段人体拟合_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/gvhmr021-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/gvhmr021-final-renders',{recursive:true});
for(let i=0;i<55;i++){const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});await fs.writeFile(build+'/gvhmr021-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(build+'/gvhmr021-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),candidate,changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
