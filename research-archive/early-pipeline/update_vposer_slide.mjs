import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='03862c97fbb5f445e0774a493a5b5884ed4e4b07130e7b74d7fbabdc345bb12a';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical deck changed; inspect before modifying.');
const r12=await read(base+'/current/smplx-vposer-fresh-012/review/review-manifest.json');
const r13=await read(base+'/current/smplx-vposer-prior-release-013/review/review-manifest.json');
const joint=await read(base+'/current/smplx-vposer-prior-release-013/joint-consistency/diagnostic.json');
const bg=await read(base+'/current/smplx-vposer-prior-release-013/joint-consistency/background-camera-check.json');
if(!r12.vposer_pair_initialization_identical||!r13.vposer_pair_initialization_identical)throw Error('Matching initial states not verified.');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count.');
const report=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
const row=report.records[0];const slide=p.resolve(row.id);
const replacements=new Map([
 ['全身姿态拟合与几何验收','身体先验与全身姿态对齐'],
 ['关键点、可见表面颜色和人物轮廓共同约束姿态，再从20万点训练1.8万步。','VPoser 用32维潜变量表示21个身体关节；真实相机与图像观测约束人物位置。'],
 ['手腕重投影误差由7.14降至6.40像素，脚踝由24.38降至23.06像素。','同一时刻、同一侧面相机：取消单目三维吸引项后，姿态变化仍有限。'],
 ['裸网格整体姿态仍错误，本轮判定为拟合失败，局部误差下降不能替代几何验收。\n先核验骨盆、躯干和四肢的三维姿态，再从头训练高斯。','四组从头拟合各1,400步，膝部、脚部与手部仍未通过对齐检查。\n先检验跨视角观测与三维结构，再从20万新高斯训练1.8万步。']
]);
for(const [before,after] of replacements){const e=row.elements.find(e=>e.text===before);if(!e)throw Error('Missing text');p.resolve(e.id).text.replace(before,after);}
const im=p.resolve('im/l4jex8bi');
const oldFrame=im.frame,oldGeometry=im.geometry,oldBorderRadius=im.borderRadius,oldRotation=im.rotation;
await im.replace({blob:await fs.readFile(base+'/current/smplx-vposer-prior-release-013/review/fixed-left90.jpg'),contentType:'image/jpeg',fit:'contain',alt:'同一时刻23、同一侧面相机，轴角、VPoser与取消单目三维吸引项的网格对照'});
// Set the frame to the measured image ratio explicitly: imported image frames
// can otherwise retain the previous picture's stretch geometry on export.
im.frame={left:(1280-382*1560/794)/2,top:174,width:382*1560/794,height:382};
im.crop={left:0,top:0,right:0,bottom:0};
im.geometry=oldGeometry;im.borderRadius=oldBorderRadius;im.rotation=oldRotation;im.fit='contain';
slide.speakerNotes.text=`身体姿态先验与观测约束

当前阶段先拟合裸人体的共享体形、逐帧关节与整体运动，再重新采样人体高斯。早期高斯在部分实拍相机下可以拟合清晰外观，但固定同一人体时刻查看侧面时，仍有肢体姿态和衣物破碎。更换坐标方向或增加点数无法证明这些局部几何正确。所有新实验均从原始59帧、固定标定相机、许可SMPL-X模板和官方预训练输出开始，不读取旧拟合参数或旧高斯。

VPoser V02_05把21个身体关节的旋转表示为32维潜变量。官方网络保持冻结，优化潜变量，经解码得到63维轴角，再输入SMPL-X。根节点旋转、平移、模型尺度、10维共享体形、手指和表情分别处理。VPoser约束姿态分布，不能自动确定场景中的位置、重力方向或真实肢体深度。使用的官方代码版本为6b883ecc3000697409a8caa2744523afe041e48a；检查点取自公开镜像，散列与镜像LFS对象一致，未取得官网独立散列。

本地表示：z_t = z_0 + 2 tanh(a) + 0.5 tanh(b_t)。z_0由原始共享姿态的编码均值取得；a是共享潜变量修正，b_t从各帧原始预测编码与z_0之差初始化。以上边界是本地诊断设置，并非官方最优参数。VPoser编码/解码会改变初始姿态，因此与轴角组的初始网格不同；VPoser联合组、分阶段组与去除单目三维吸引项组的初始数组逐项相同。

目标包含真实相机下的身体与足部重投影、可见手部预测、单向人物遮罩、共享体形与时间平滑，以及0.01倍潜变量均方惩罚。手部由77个可见HaMeR预测提供局部手指姿态，隐藏手部只沿用最近的可见关节旋转；其图像损失权重为零。没有把裁切相机中的绝对手腕旋转直接复制到场景。遮罩只惩罚裸人体越过外部边界，不要求裸身填满衣服和鞋。所有观测均来自训练视频，未增加同步多视角或俯仰真值。

对照预算：轴角联合、VPoser联合、VPoser分阶段三组各1400次更新；另从头运行一组1400步，仅把单目预测的三维关节和旋转吸引项从0.12设为0。共5600次身体拟合，无新增高斯训练。轴角控制逐字节复现先前011B结果。共同阶段为250步整体配准、500步共享体形与姿态、650步开放受限逐帧残差。分阶段组前750步不使用手部位置损失，随后冻结整体配准和体形，用200步逐步加入手部损失。冻结后的参数逐项未变，但潜变量仍耦合身体关节，并非独立手臂优化。

结果：可见身体关节残差中位数为8.87、12.31、11.37、12.13原图像素；相应手部预测残差为6.20、7.60、9.01、7.60像素。这些是参与拟合的检测预测残差，不是真值精度。三组012的六张原图叠加、八个手部裁切与七个固定方向均已检查；013检查了正面与侧面原图、固定侧面和俯视。VPoser改变手臂配置，但膝部、足部和手部边缘/尺寸仍不匹配；去除单目三维吸引项没有明显修复。四组均不采用，不部署00，不将本地适配结果推广为VPoser完整方法无效。

独立于人体拟合的诊断：将原始二维关节按连续10帧分块留出，训练块仅拟合一个静止三维点，使用soft-L1且尺度8像素。左膝留出残差中位数${joint.joint_records.find(r=>r.joint==='left_knee').withheld_block_median_px.toFixed(2)}像素，右膝${joint.joint_records.find(r=>r.joint==='right_knee').withheld_block_median_px.toFixed(2)}像素。该残差混合人物运动、检测偏差与标定误差，不能直接当作关节真值或据此修正身体。现有COLMAP背景特征126487个观测的重投影中位数${bg.background_median_px.toFixed(2)}像素，人物遮罩内31311个特征为${bg.person_region_median_px.toFixed(2)}像素。这些特征参与过原始SfM，仅说明相机、特征与缩放在存储数据中一致，不是独立标定精度。

下一步利用真实图像对应，区分关节检测的跨视角偏差与人物运动，并检查人物区域SfM三维支持。裸人体、宽松衣物和鞋面必须分别解释，不能把衣物表面当作皮肤位置。当前示意相机竖直参考初始模型，没有独立重力标定，因此不能把画面中的整体倾斜直接当作身体弯腰。完成几何和时间一致性核验后，再初始化20万个高斯、按用户纠正的18000步预算训练。

参考代码：https://github.com/nghorbani/human_body_prior；https://github.com/jmaraval/VHAP_smplx 。GaussianAvatars_SMPLX公开仓库缺少关键SMPL-X入口，本地补齐与诊断不称作者完整复现。
本地依据：${base}/current/smplx-vposer-fresh-012/review/review-manifest.json；smplx-vposer-prior-release-013/review/review-manifest.json；其joint-consistency/diagnostic.json与background-camera-check.json。
`;
const candidate=build+'/vposer-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const preview=await p.export({slide,format:'png',scale:1});
await fs.writeFile(build+'/vposer-slide-after.png',new Uint8Array(await preview.arrayBuffer()));
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261023_身体先验验证_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/vposer-final-validation-v3.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/vposer-final-renders',{recursive:true});
for(let i=0;i<55;i++){const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});await fs.writeFile(build+'/vposer-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));}
await fs.writeFile(build+'/vposer-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
