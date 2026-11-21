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
const expected='90e34be744e409aca7b26c81d43d10cb0d378d110fa1c5119c6c65a3981c9061';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical deck changed.');
const review=await read(base+'/current/smplx-floor-contact-fresh-016/review/review-manifest.json');
if(review.visual_review_pending||!review.control_contact_initialization_identical||!review.control_exactly_reproduces_015B)throw Error('Unreviewed or mismatched comparison.');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count.');
const inspected=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
const row=inspected.records[0];const slide=p.resolve(row.id);
await fs.writeFile(build+'/floor-slide-before.json',JSON.stringify(inspected,null,2));
const replacements=new Map([
 ['身体先验与全身姿态对齐','身体先验与场景接触约束'],
 ['VPoser 用32维潜变量表示21个身体关节；真实相机与图像观测约束人物位置。','姿态先验约束关节配置；真实图像、地面与接触关系约束人物在场景中的位置。'],
 ['同一时刻、同一侧面相机：取消单目三维吸引项后，姿态变化仍有限。','同一时刻、同一地面参考相机：脚掌有所放平，膝部弯曲基本未变。'],
 ['四组从头拟合各1,400步，膝部、脚部与手部仍未通过对齐检查。\n先检验跨视角观测与三维结构，再从20万新高斯训练1.8万步。','接触约束未使整体通过；身体关节图像残差由8.12增至9.25像素。\n下一步核查跨视角姿态与整体运动，通过几何验收后再训练高斯。'],
]);
for(const [before,after] of replacements){const e=row.elements.find(e=>e.text===before);if(!e)throw Error('Text anchor missing: '+before);p.resolve(e.id).text.replace(before,after);}
const im=p.resolve('im/l4jex8bi');
const geometry=im.geometry,borderRadius=im.borderRadius,rotation=im.rotation;
await im.replace({blob:await fs.readFile(base+'/current/smplx-floor-contact-fresh-016/review/fixed-left90.jpg'),contentType:'image/jpeg',fit:'contain',alt:'同一人物时刻23及地面参考相机，无接触项、含接触项与真实纹理稀疏三维支持'});
im.frame={left:(1280-382*1560/826)/2,top:174,width:382*1560/826,height:382};
im.crop={left:0,top:0,right:0,bottom:0};im.geometry=geometry;im.borderRadius=borderRadius;im.rotation=rotation;im.fit='contain';
slide.speakerNotes.text=`身体先验、真实观测与场景接触

阶段目标：先恢复可信的人体姿态和场景配准，再从20万个新高斯与新优化器开始18000步训练。所有拟合均从原始59帧、固定已标定相机、许可SMPL-X模板及官方预训练预测重新初始化，不读取旧人物拟合结果。此页图是裸网格诊断，不是高斯渲染或完整方法复现。

VPoser V02_05以32维潜变量描述21个身体关节；固定网络解码为63维轴角并送入SMPL-X。整体旋转、平移、共享体形、尺度和手指分别估计。官方代码版本6b883ecc3000697409a8caa2744523afe041e48a。官网返回502后从公开镜像下载权重，散列与该镜像LFS一致，没有取得官网独立散列。已验证严格加载、CUDA前向、梯度与SMPL-X接口。

012/013四组各1400步：轴角、VPoser联合、VPoser分阶段、去除单目三维吸引项。VPoser编码/解码使其初始姿态不同于轴角，因此只在VPoser组间严格比较相同起点。可见身体预测残差8.87/12.31/11.37/12.13px；四组整体均未通过。先冻结整体再拟合手部、取消单目三维吸引项也没有明显修复。该结果只评价本地有界适配，不能推广为VPoser方法无效。

014从真实图像特征检查可见表面。原始COLMAP中存在同一3D点同一图像的重复观测；每帧最多保留一组：坐标范围不超过1px时合并，歧义组排除。候选要求至少10帧、所有投影在人物遮罩内、内部距离中位数至少8px、观察射线跨度至少8度。按连续10帧分块留出，用其余至少4帧和4度跨度做soft-L1三角化，鲁棒尺度2px。稳定条件为至少5个留出观测、误差中位数小于2px、P90小于5px且正深度。943条候选中312条通过；按粗略图像高度分区，脚部66、下腿72条。所有脚部候选留出残差中位数1.39px，下腿1.79px。衣物或鞋面纹理不等于皮肤或关节；相机曾使用原特征估计，以上不是独立标定精度。

遮挡处理：逐张检查59帧，0至11帧的右膝/踝和足部、42至58帧的左侧对应观测被近侧裤腿遮挡，损失权重置零；已有肘腕遮挡策略不变。初始化仍读取相同原始预测。015对轴角控制、剔除腿部遮挡的轴角、剔除腿部遮挡的VPoser各拟合1400步。前两组在共同可见子集中的膝残差约10.54/10.02降至8.08/8.57px，但侧面膝弯曲和脚掌姿态未明显修复，VPoser组也未通过。所有数字均是训练预测残差。

地面参考：选择至少出现3帧、处于图像底部8%且低于人物遮罩12px的原始SfM点。1393个候选经2000次RANSAC和SVD得到1284个内点，阈值0.035世界单位。单位法向量为(0.107932,-0.985047,-0.134287)，平面偏移4.588044；内点距离中位数0.007473世界单位。法向朝向相机。原检查相机竖直参考与该法向相差10.883度，因此重新按地面方向设置七个公共检查相机，人体网格没有旋转。地面参考不是IMU重力，也不能单独判断身体姿态。

016使用完全相同初始数组比较无接触项/增加接触项，每组1400步，控制结果逐字节复现015可见性轴角组。前250步只优化整体配准，接着500步开放共享体形和姿态，最后650步允许受限逐帧残差。相机固定；手指使用可见HaMeR局部姿态，遮挡时沿用最近的可见手指姿态但没有图像损失；不复制裁切相机的绝对手腕方向。身体、足部、手部重投影以及单向人物遮罩、单目关节与旋转约束均与控制组一致。

脚底接触模型：模板踝/脚蒙皮权重和大于0.7、模板纵向距脚最低点8毫米以内的点，每脚99个。对每点计算h=(n·v+d)/s0，s0为原始初始化世界尺度3.0378457943794817，而不是优化后的尺度。用rho(x)=sqrt(1+x^2)-1：平坦项rho(max(|h-mean(h)|-0.012,0)/0.015)，下界项rho(max(0.01-h,0)/0.015)，上界项rho(max(h-0.07,0)/0.02)，再平均所有点、帧和双脚。接触权重1，250步后用200步渐入。厘米单位来自模型尺度，不是实测深度。1至7cm鞋内高度和12mm脚底起伏是有界假设；当前只针对看起来双脚着地的素材，不是通用动态接触检测器。

固定时刻23，左右脚底平面与地面的夹角由31.76/22.45度变为13.33/10.84度；左右膝几何弯曲约23.37/22.19度变为23.39/21.63度。可见身体关节残差8.12升到9.25px，手部预测残差6.23降到6.05px。脚底平坦性被直接约束，因此改善不是独立准确率。已检查正面/侧面实拍叠图、固定侧面和俯视四张关键图，脚掌有所放平但膝部与整体仍未通过，不升级、不训练高斯、不部署00。

本阶段015/016共7000次新的身体拟合更新、0次高斯更新。保留当前诊断对照，失败历史模型仅留精简证据。下一步核查原始单目根旋转、逐帧位置和腿部姿态是否随观察方向漂移，用真实纹理轨迹区分运动与检测变化。准备共享下身位置与受限逐帧运动的从头比较，同时允许头部、手臂真实运动；不得强制全身静止或将衣物点直接用作关节真值。

官方参考：https://github.com/nghorbani/human_body_prior；https://github.com/jmaraval/VHAP_smplx 。GaussianAvatars_SMPLX公开仓库缺少关键SMPL-X入口，本地适配不称作者完整复现。
本地证据：${base}/current/smplx-observation-support-014/track-audit.json；smplx-leg-visibility-fresh-015/review/review-manifest.json；smplx-floor-contact-fresh-016/review/review-manifest.json及foot-floor-diagnostic.json。
`;
const candidate=build+'/floor-contact-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261023_场景接触_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/floor-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/floor-final-renders',{recursive:true});
for(let i=0;i<55;i++){const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});await fs.writeFile(build+'/floor-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(build+'/floor-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),candidate,changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
