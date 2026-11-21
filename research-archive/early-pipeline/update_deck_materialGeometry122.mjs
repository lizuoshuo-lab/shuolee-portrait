import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../clip03-head-level-support/facade_post_context.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/headSupport122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/materialGeometry122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('头部高度的固定点支持','对应证据接入三维评测');
replace('三列为248','沿用围栏反例，三列为248、264、280帧。4000基准K保持。');
replace('60组全部审阅','逐视角检查三角化支撑点，三次立柱观察均被拒绝。');
replace('该轨迹此前','已修复0.024像素的坐标依赖，面部误差仍达13.05像素。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'远楼与重复围栏立柱的三帧原生上下文，图片无新增文字',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height*1.5)/2,top:frame.top,width:frame.height*1.5,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N13。第一YAML K以4000×3000为当前首选输出假设，经下采样和居中裁剪得到视频K，N5/N6/N12g已经这样计算。该尺寸字段自身证明标定参考尺寸，未独立证明实际K_rect。当前1520×2704的fx1494.442548、fy1493.094180、cx761.316834、cy1357.202690保持原值。原视频不再去畸变。", "真实调用链核查：旧N3 Observation合同此前主要由诊断及报告使用。新增material_geometry122.py固定相机几何评测器复用同一数据类，检查目标及每个三角化支撑观察，同族任一反证传播到所有目标。保留每个原观测数值、冻结划分和所有失败。没有创建另一套观测格式。", "一项工程组件，正式122训练入口尚未实现。旧113 admission.py/cli.py/training.py内容哈希保持不变，不改变另一00对话的部署项目。相机文件绑定证据并不等于物理标定真值。", "一次有界CPU运行初轮在旧数值一致性检查失败后停止。failure.json和初始源码保存。诊断显示重设世界原点后99个目标的DLT预测最大变化0.023671011899px，误差值最大变化0.019660799455px。相同固定三维点的坐标变换投影差仅9.0949e-13px；恢复原216坐标后旧误差值一致到2.2867e-12px。不是相机矩阵转置错误的证据。", "必要修复只改新评测器：每个留出目标用最早支撑相机作局部坐标，最大支撑基线作尺度，做齐次代数DLT后转回世界坐标。待测目标从三角化与规范选取中排除。没有更新K、相机或原始像素。此规范消除输入世界Sim3依赖，但不将代数DLT变成几何重投影最优估计。", "修复后28项CPU组件检查全部通过：已知合成几何、完整偏心主点、目标与支撑物理身份、同族跨轨迹反证、FIT/留出泄漏、缺相机、未知静止性或测量不确定性、错误源视频、重复身份、不合法位姿以及世界旋转平移尺度变换。实际102个目标全部保留，99个历史目标恢复原规范核对。输入和旧入口哈希不变。", "新局部规范下输入世界坐标变化的预测最大差2.2737e-12px；规范改变导致误差数值相对旧评测最大变化0.055904140079px，如实保留。13个面部留出目标仅2个数值<1px，最大13.048757643629px，物理准入0个。0.024px的数值问题不足以解释13px偏差，不能宣称重影根因已解决。", "实物反例仍是N12i的族4936，248/264为横幅文字下立柱，280变为卡通图案下的另一实例。该族三次观察都因整族反证拒绝，原数值误差412.211、195.908、363.890px保留。该族从未用于N12g相机拟合，不能据此归因N12g面部误差。", "本页图片是上一轮已审阅的原生上下文，非本轮新增图像或训练结果。维持原像素比例，图内无新增文字。左至右248、264、280帧，上排远楼，下排围栏。", "下一步只读检查现有native_portrait_renderer122和119自由衣物表面训练实现，登记必要的合成工程接入验证。当前仅7诊断帧，全训练帧尚未注册，物理材料/静止性/输出K均有缺口，不开始人物训练。合成通过不等于实拍亚像素或多人物泛化通过。", "来源：robust-portrait-122/video-update-20261027/material-geometry-consumer-plan.json；material-geometry-consumer/failure.json、parity-diagnosis.json、repair-plan.json、review.json与gauge-consistent-repair/result.json/evaluated-observations.json。以下保留历史诊断："].concat([previous]).join('\n\n');
const candidatePath=build+'/materialGeometry122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122对应证据接入.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/materialGeometry122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_materialGeometry122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/materialGeometry122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/materialGeometry122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/materialGeometry122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/materialGeometry122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
