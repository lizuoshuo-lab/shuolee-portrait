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
const expected='662e09379511d2fc6a3a987cce06b61afa48bd9a9a2179e23d5377c02c8257bf';
const phase=base+'/current/smplx-gvhmr-shoe-motion-fresh-023';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
if(await hash(source)!==expected)throw Error('Canonical deck changed.');
const conclusion=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(conclusion.state!=='component_improved_body_geometry_not_accepted')throw Error('Review required.');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count');
const inspected=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile(build+'/gvhmr023r3-slide-before.json',JSON.stringify(inspected,null,2));
const row=inspected.records[0],slide=p.resolve(row.id);
const replacements=new Map([
 ['时序先验与分阶段人体拟合','人体先验与纹理运动约束'],
 ['固定真实相机，先估计整体尺度与位置，再联合约束关节、手部与体形。','分阶段拟合关节与体形，再用真实纹理对应约束足部相对运动。'],
 ['同一时刻、同一世界相机：原拟合 → 时序预测 → 整体配准 → 关节与手部拟合。','左：实拍；中：接触约束；右：加入纹理约束。白圈为观测，红叉为投影。'],
 ['可见关节点残差中位数：30.15 → 22.79 → 8.13 像素（拟合观测）。\n手腕贴合改善；侧面体形和遮挡肢体尚未通过，继续核验。','未参与约束的 33 条足部附近纹理：30.65 → 1.36 像素。\n局部运动一致性改善；身体侧面、体形和遮挡肢体仍未通过。'],
]);
for(const [before,after] of replacements){const e=row.elements.find(e=>e.text===before);if(!e)throw Error('Missing anchor '+before);p.resolve(e.id).text.replace(before,after);}
const imageElement=row.elements.find(e=>e.id.startsWith('im/'));
if(!imageElement)throw Error('Missing image');
const im=p.resolve(imageElement.id);
const geometry=im.geometry,borderRadius=im.borderRadius,rotation=im.rotation;
await im.replace({blob:await fs.readFile(phase+'/review/shoe-feature-comparison.jpg'),contentType:'image/jpeg',fit:'contain',alt:'原始实拍纹理与模型运动的投影对照：左为原图，中为接触控制，右为新增纹理运动约束。包含鞋和裤脚。整体人体仍未通过。'});
im.frame={left:(1280-382*1500/852)/2,top:174,width:382*1500/852,height:382};
im.crop={left:0,top:0,right:0,bottom:0};im.geometry=geometry;im.borderRadius=borderRadius;im.rotation=rotation;im.fit='contain';
slide.speakerNotes.text=`人体先验与纹理运动约束：方法、训练条件与实际效果

问题与表示
有限视角的颜色可以被高斯拟合，但人体姿态与运动不一致，会在未知视角和导出模型中暴露错误。本页是完整GaussianAvatar训练前的SMPL-X配准阶段。保留同一组59个真实相机，求共享体形和尺度、逐帧根变换、局部姿态与手部；后续高斯应重新初始化，不能从旧人物或高斯状态续训。裸人体模板提供骨骼和形变载体，不等同衣物、头发或鞋的外表面。

一、身体先验
VPoser V02_05以32维潜变量解码21个身体关节旋转，约束姿态分布，不给出场景尺度或全局位置。已下载并完成RTX5080运行验证。
GVHMR将HMR2.0a图像特征、ViTPose-H二维关节、相机运动和透视条件输入时序Transformer，提供人体姿态、体形与根变换预测。本地使用原始网络输出作为独立先验，没有复制此前学习到的个体形状。其接触和逆运动学后处理产生的地面不替代真实场景证据。当前不是完整官方端到端复现：检测框与相机里程计分别由原始人物遮罩和已有标定相机替换。

二、输入时间与相机
原视频1520×2704、24.029053fps，完整解码177帧。59张标定图对应零起点1、4、7至175帧；逐像素复核去畸变后的1514×2694图。内参fx=fy=1461.4907695像素，cx=757、cy=1347像素。先验按30fps最近邻重采样218个位置，实际只有175个独立实拍帧，重复不增加视角。相机旋转球面插值、中心线性插值，按图像实际拍摄时间关联。相机绕人运动，static_cam=False；插值相机不是新测量。
初始化在模板米中去除骨盆平移、撤销预测的相机空间根旋转，再经真实相机变换进入统一世界坐标。使用近似静止骨盆求初始尺度3.18005世界单位/模板米，不把这个初始化假设强加为所有身体部位绝对静止。相机全程固定。

三、观测点、遮挡和手部
图像COCO17关节点与SMPL-X同名骨架点不完全相同，使用官方SMPL-X到SMPL稀疏映射及COCO17表面回归矩阵，保留其原始权重。肩部两种定义的投影差约16–20像素，说明关节点定义必须先统一。官方网格、骨架和COCO输出回放误差均小于6×10^-7模板米。
身体点按检测器各关节90%分位归一化响应并平方；面部5点再乘0.35，髋部不直接作为二维约束。遮挡的另一侧肘、腕、膝、踝按原图可见性剔除，脚点同样筛选。HaMeR原始预测提供77个可见手样本的局部手指姿态和图像手点，遮挡时仅用最近可见局部手姿态作先验。减去SMPL-X非平手均值，不把裁切相机中的绝对手腕方向移植到场景。

四、可优化参数与分阶段训练
每组从同一份官方原始预测、相机、遮罩、检测器观测和许可模板重新初始化，优化器也新建。023与独立022控制的初始化数组逐项相同；不加载022拟合结果作为起点。
共享体形β相对原始逐帧预测的中位数，每维最多改变1。局部旋转在SO(3)上组合：R_tj=Exp(0.30tanh(a_j)+0.10tanh(b_tj))R_prior,tj。界限是轴角向量各分量，不是总旋转角。根旋转校正每轴±0.35弧度，位移每轴±0.30倍初始尺度，共享尺度用±0.30对数幅度变化。
1200次更新：第1–300步仅优化根变换和尺度；301–700开放共享关节与体形；701–1200开放有界逐帧关节。Adam学习率根参数0.008，共享0.006，逐帧0.004；第1000步后乘0.4，梯度范数裁剪10，检查非有限值和相机前方。

五、图像、接触与时序目标
L=L_body+0.2L_foot+0.5r_hand L_hand+2L_contain+0.15L_pose+0.04L_shape+L_contact+0.4L_stable+0.05L_smooth+0.3r_tex L_tex+L_front。
稳健残差ρ(x)=sqrt(1+x²)-1。身体、脚、手二维误差分别以20、18、10像素归一化；手部权重在300–500步渐增。姿态先验约束原始GVHMR局部旋转校正（0.15弧度尺度）和局部骨架偏移（0.06模板米）。体形差为均方惩罚。
人体包含项只惩罚裸网格超出衣物遮罩：5像素容差后除15，手部排除并使用专门手点。不能要求皮肤填满外套、裤腿和鞋。接触项是估计地面上方1–7厘米的软高度带，容许鞋底厚度与误差；不是测量真值。脚底对应顶点在时间上以2.5厘米尺度约束，优于只约束中心，但仍不能替代真实运动证据。
相邻姿态/根旋转修正以0.04弧度平滑，位移修正以1.5厘米平滑，不冻结整个上身。全部59帧均用于图像拟合，没有独立未知俯仰GT。

六、真实纹理怎样约束运动
从原视频标定对应中取经静态一致性检验的足部附近特征，按原始先验的空间距离分到左右脚。它们包括鞋和部分裤脚，不是经过语义标注的纯鞋面；不推广到明显摆动的裤脚。66条轨迹按每只脚内点ID排序、交替分为33条优化轨迹与33条未优化诊断轨迹。左右脚各占一半损失，各轨迹先按自身可见观测次数归一化，避免某只脚或长轨迹支配目标。
G_t为当前踝部刚体变换。世界旋转使用外部根旋转依次乘髋、膝、踝的局部旋转，原点为当前踝关节：G_t=[R_t,o_t]。对原始世界特征X_p，以时刻23为参考，X_hat(t)=G_t G_23^-1 X_p，再用真实相机π_t投影，与实拍像素x_tp比较。此式约束相对运动，不强行把裸脚网格拉到鞋的外表面；参考时刻投影不会因这个相对传递式而改变。
L_tex以10像素归一化的稳健二维误差计算，总权重0.3，在第100–300步从0升至1。优化使用33点666次观测，另一半33点623次观测不进入损失。两组之前已被检查，原始相机也使用过这些特征，故不是盲测或独立三维精度。
坐标核验：原先膝—踝—脚尖构造坐标系混入了小腿方向。现改用SMPL-X真实运动链，与官方batch_rigid_transform旋转最大差1.52×10^-7；参考时刻传递恒等误差1.19×10^-7世界单位；独立NumPy/SciPy与PyTorch投影误差最大0.00042像素。COLMAP特征采用原始像素坐标，不减去关节观测管线的0.5像素偏移。小腿构造坐标系的旧指标不与本页正确运动链指标混用。

七、效果与解决的问题
023与022均独立从头1200步，主要差别是增加真实纹理运动项。022仅从中心稳定改为点稳定，运动链核验下的足部特征残差中位数仍约30.83像素，不能称整体突破。
在未参与新增约束的33条轨迹623个观测上，023相对022的中位残差30.6541→1.3626像素，90%分位39.6664→4.2159像素。此页实拍11与45的白圈为真实特征，红叉为模型传递后的投影，黄色线段连接两者；对应帧残差中位数41.26→2.15及33.72→1.10像素。图像保留原比例，三列使用相同裁切。
身体观测中位残差8.1267→8.7126像素，90%分位25.03→26.23像素，说明存在小幅拟合代价。手腕3.59/4.52像素，肘30.40/24.45像素。固定时刻23左右膝弯曲12.38/6.89度，角度变小不等于更准确。局部纹理的运动一致性改善，不能把它写成全身配准或高斯质量完成。

八、仍存不足与下一轮
59帧联系表、6张完整源图叠加、7个固定世界相机方向、两条时间视频的6时刻快照和两张纹理近景已检查。完整连续视频尚未验收。固定世界相机不逐模型居中、转正或身高归一化。侧面胸腹形态、整体倾斜及遮挡肢体仍有疑问，裸模板与宽松衣物的真实内部关系没有测量GT。
下一轮先核对躯干与下肢真实纹理，检验逐帧自由度是否把拍摄视角变化误解释成人体运动，再从头比较共享躯干/下身姿态约束。旧SMPLer-X阶段共享下身未通过；新GVHMR先验和实际纹理约束不同，须重新做匹配对照，不继承旧结果。增加优化步数不是默认修复。
022与023本次共2400次身体更新，0次新高斯更新。整体几何尚未采用、不替换查看器、不部署00。通过身体可观察对齐和时间一致性后，才重新采样20万高斯、新建优化器训练18000步，禁止恢复旧个体状态或180000步任务。

源码与证据
GVHMR官方：https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1 。权重固定公开镜像的LFS摘要相符，无作者独立哈希。GVHMR默认研究非营利许可，当前为本地研究比较。
本地新增纹理项是适配代码，不归为GVHMR原论文方法。实际证据：${phase}/phase-conclusion.json、shoe-protocol.json、shoe-contract.json、motion-review/motion-audit.json、review/shoe-feature-figure.json。对应源码：refine_shoe_motion_fresh.py、shoe_texture_constraint.py、audit_foot_frame_contract.py。`;

const candidate=build+'/gvhmr023r3-candidate-r2.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261023_纹理运动约束_定稿3.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/gvhmr023r3-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/gvhmr023r3-final-renders',{recursive:true});
for(let i=0;i<55;i++){const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});await fs.writeFile(build+'/gvhmr023r3-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(build+'/gvhmr023r3-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),candidate,changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
