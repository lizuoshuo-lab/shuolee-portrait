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
const expected='c5a43e6672286c2d22f03b208e86b24e2a1ba9d4c9e7b169dbb40828e813c90f';
const phase=base+'/current/smplx-gvhmr-shared-world-fresh-025';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
if(await hash(source)!==expected)throw Error('Canonical deck changed.');
const conclusion=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(conclusion.state!=='shared_world_hypothesis_not_adopted')throw Error('Review required.');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count');
const inspected=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile(build+'/gvhmr025-slide-before.json',JSON.stringify(inspected,null,2));
const row=inspected.records[0],slide=p.resolve(row.id);
const replacements=new Map([
  ['人体先验与纹理运动约束','人体运动约束与几何检查'],
  ['分阶段拟合关节与体形，再用真实纹理对应约束足部相对运动。','真实纹理约束跨帧运动，固定世界相机检查身体形态。'],
  ['未参与约束的 33 条足部附近纹理：30.65 → 1.36 像素。\n局部运动一致性改善；身体侧面、体形和遮挡肢体仍未通过。','纹理残差：足部 30.65 降至 1.36 像素，躯干与腿部 24.23 降至 2.97 像素。\n共享姿态仍未修正侧面形态，需先核验二维关节点。'],
]);
for(const [before,after] of replacements){
  const e=row.elements.find(e=>e.text===before);
  if(!e)throw Error('Missing text anchor '+before);
  p.resolve(e.id).text.replace(before,after);
}
// Preserve the actual foot-feature figure, full crop, layout and first 54 slides.
let notes=row.notes.text;
if(!notes.includes('七、'))throw Error('Missing notes boundary');
notes=notes.split('七、')[0]
  .replace('人体先验与纹理运动约束：方法、训练条件与实际效果','人体运动约束与几何检查：方法、训练条件与实际效果');
slide.speakerNotes.text=notes+`七、从足部运动到躯干和下肢运动
足部对照023用33条未参与新增损失的轨迹、623次观测作诊断，残差中位数从30.654降至1.363像素。身体可见关节残差从8.127升至8.713像素。PPT图仍是这组真实足部示例，不能把图误解为后续共享身体的结果。
躯干和下肢对照024沿用同一1200步流程和其他损失，只增加权重0.3、10像素归一化、100至300步渐增的身体纹理运动项。轨迹来自原始SfM和静态一致性筛选，不从旧拟合提取。参考帧23中，距原始先验表面不超过0.20模板米，最大蒙皮权重至少0.65，再按空间高度选择躯干和腿部。选中108条，按关节组内点ID交替分成55条优化轨迹（973次观测）和53条未优化轨迹（930次观测）。六个骨骼组为骨盆、左右髋、左右膝和上躯干，先组内按轨迹平均，再组间等权。中间两个脊柱组没有满足置信条件的轨迹，因此剔除，没有降低门槛补点。以单根骨骼变换近似衣物纹理运动有适用范围，不等于裸人体表面监督。
两臂各从头优化1200步，初始化数组逐项相同。控制臂最终数组逐项复现023，确认新增项之外没有改变。未优化身体纹理残差中位数24.231降至2.971像素，90%分位37.880降至7.832像素。与此同时，身体关节残差8.713升至10.063像素，左膝7.578升至17.714像素。固定世界侧视仍没有修正形态，因此没有采用这组模型。

八、共享世界姿态的匹配对照
025检验逐帧自由度是否把相机视角变化解释为身体运动。两组均从同一份原始预测重建起点：世界根旋转和局部1至11号关节取SO(3)平均，根平移和体形取中位数，保留原始逐帧头颈、肩臂预测。所有初始网格和参数逐项完全相同，优化器新建。自由组允许逐帧根变换和有界局部修正；共享组只保留一套世界根旋转、位置，并把1至11号关节的逐帧修正置零。头颈、肩臂与手部仍可变化。两组均保留身体及足部纹理、图像点、轮廓、接触与先验项，各1200步。
未优化身体纹理残差中位数2.493降至0.953像素，足部1.283降至0.677像素。身体二维关节残差10.071升至12.013像素，右肩13.943升至26.031像素，左膝18.880升至28.425像素。侧面形态仍有疑问，共享整段身体的假设未采用。此对照与024的初始化不同，不能把两个实验的分数拼成连续优化曲线。
相对变换在参考时刻满足G_r乘G_r的逆为单位变换，因而运动项可约束跨帧一致性，却不会直接修正参考姿态的绝对形状。其点、相机和静态筛选复用原视频的SfM支持，未优化轨迹也不是独立三维GT。

九、检查参数边界与二维观测
共享组根旋转、平移和尺度修正远离边界，膝和躯干修正不到各自分量上限的约28%和23%。手腕接近上限，应独立处理。这排除了本轮膝和躯干仅因碰到硬边界而无法继续移动的简单解释，不能据此排除姿态先验权重的影响。
用当前可见性权重，逐关节拟合独立静止三维点，以连续10帧分块留出。原始左、右膝检测的留出残差中位数约21.88与22.49像素。该诊断不施加人体关节链，仍混合人物运动、检测偏差与相机误差，不能当作人体真值或严格误差下界。
复用已有官方ViTPose-H预测，依据原始时间映射取59帧，保持相同可见性权重，仅比较二维坐标，不额外训练或替换模板。两套检测的肘部中位分歧约22.60/23.34像素，膝部26.42/19.74像素。ViTPose左膝的静止点留出误差约31.94像素，并非普遍优于旧检测。下一轮先在实际照片上核验肩、肘、膝的定义、遮挡和裁切/像素坐标，再设计带观测不确定性的从头对照，不直接把任一检测器当真值。

十、目前解决的问题与不足
真实纹理项改善相对运动一致性，脚底接触与轨迹有独立作用。身体先验文件、CUDA推理及SMPL-X接入已验证，缺文件不再是当前障碍。VPoser权重来自固定版本公开镜像，SHA256与其LFS对象相同，尚无官网独立摘要。
024和025共4组从头各1200步，合计4800次身体更新，0次新高斯更新。已检查024的23/45源图、正面和左90度、两条时间快照及纹理图；025检查00/11/23/45/58源图、左90度和时间快照。全部59视角已渲染，并未逐张人工验收，也未完成完整连续视频验收。固定世界相机不逐模型居中、转正或按身高归一化。
侧面胸腹形态、整体倾斜与遮挡肢体尚未确认正确，没有独立身体测量或未拍摄俯仰GT。本轮不替换查看器、不部署00。身体几何和时间一致性通过后，再从头采样20万个高斯并训练18000步。

源码与证据
官方VPoser：https://github.com/nghorbani/human_body_prior 。固定代码6b883ecc3000697409a8caa2744523afe041e48a。权重目录D:/GS-SLAM-Data/models/vposer/V02_05，download-receipt.json和verification.json保存来源及加载验证。
官方GVHMR：https://github.com/zju3dv/GVHMR/tree/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1 。权重摘要匹配固定公开镜像的LFS对象，无作者独立摘要。默认研究非营利许可，当前为本地研究对照。
本地纹理、共享姿态和诊断代码属于适配，不归为GVHMR论文方法。${base}/current/smplx-gvhmr-body-motion-fresh-024/phase-conclusion.json，${phase}/phase-conclusion.json、parameter-bounds-audit.json、joint-observation-audit.json、vitpose-joint-observation-audit.json。对应源码refine_body_motion_fresh.py、refine_shared_world_fresh.py、body_texture_constraint.py、audit_shared_joint_support.py、audit_vitpose_joint_support.py。`;

const candidate=build+'/gvhmr025-candidate-r2.pptx';
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261023_运动与几何核验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
  pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
  layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
  verifyArtifactToolImport:true,receiptPath:build+'/gvhmr025-final-validation-r2.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir(build+'/gvhmr025-final-renders',{recursive:true});
for(let i=0;i<55;i++){
  const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
  await fs.writeFile(build+'/gvhmr025-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
}
await fs.writeFile(build+'/gvhmr025-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),candidate,changed_slides:[55],rendered_slides:55,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,final_sha256:await hash(finalPath)}));
