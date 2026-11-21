import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../../native-training-integration/existing-backend-path/synthetic_comparison.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/materialGeometry122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeTraining122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('对应证据接入三维评测','原生训练与小高斯剔除错误');
replace('沿用围栏反例','96点合成：目标 / 初始化 / 120步；上排训练视角，下排留出。');
replace('逐视角检查','20万点被误剔除；仅改裁剪后，30681点获得位置更新。');
replace('已修复0.024','生产修复待做。该后端尚未用于历史人物，不能归因旧重影。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'已知合成目标、从头初始化和120步结果，上为训练视角，下为留出视角；图片无新增文字',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height*1.5)/2,top:frame.top,width:frame.height*1.5,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 T0a/T0b。保持用户确认的第一YAML4000×3000基准，经下采样和居中裁剪。实际K_rect及整数裁剪实现仍未独立实证，本轮没有修改人物相机、重新BA或人物训练。", "只读源码核查：历史119自由衣物中心训练已经直接优化世界XYZ，但使用传统屏幕空间gaussian_renderer。新native_training122组件连接已有gsplat1.5.3 UT和ray-3D内核，使用自由XYZ、完整协方差、SH3、空Adam、固定完整K与外参、准确前景RGB和轮廓有效区域、1000步审阅钩子。没有修改119、113或运行中的原生查看器。", "T0a预登记一次已知合成96点、4训练视角和2不参与优化视角、120步。两个启动实现错误分别是Windows到WSL路径斜杠和未加入已存在gsplat源码路径，都在合成优化前停止并保留日志；仅做必要修复，无依赖下载。随后实际优化120步完成，每个训练视角30次，优化器由0状态到5参数组。全部96中心移动，位移中位数0.037419世界单位。", "训练目标比初始0.122605338544，留出目标比初始0.421893873318；图中左右分别为已知目标/初始/训练后，上下为训练相机0及留出相机4，每格160×160原始像素。实际审阅可见形状和颜色改善，但留出结果仍有模糊与差异。12个轨道视角已审阅，是粗糙合成椭球，不代表人体新视角能力。教师与学生同用已知后端，不是独立物理标定真值。", "T0a有11项先行控制通过，包括前景外RGB无损失/梯度、遮挡区RGB和alpha不监督、全部训练视角/空优化器/中心更新、6记录相机完整协方差SH的PLY原生回读、审阅拒绝在第2步停止、缺1000步审阅拒绝、旧输出不可覆盖、合成KB4有限非零几何梯度。RGB回读最大9.54e-7、alpha1.07e-6。整套T0a仍失败：200000点64×64一步检查所有位置梯度为0，不能以小规模通过作为完整接入通过。", "T0b注册单因子诊断，冻结同一20万点、相机K/外参、尺度0.004/0.005/0.003、透明度0.1、eps2d0.3、SH及所有CUDA代码，仅切换fully_fused_projection_with_ut的可选透明度参数。原后端与底层算子重接原规则的RGB、alpha及位置梯度完全一致，均全部剔除。", "源码ProjectionUT3DGSFused.cu约169至180行在包围范围计算中先以opacity乘二维模糊补偿，小于1/255则剔除。但classic ray-3D后续使用原opacity。20万个点的补偿后透明度最小0.001563876、中位0.002363345、最大0.003685487，全部低于1/255。仅关闭该提前裁剪后20万点全部获得有效范围，30681点有非零位置梯度且在相同Adam学习率下一步更新，所有参数梯度有限。未通过改相机/放大尺度/提高透明度或改阈值获取结果。", "另一个独立NumPy控制将一颗同尺度各向异性高斯放在精确像素射线上，按完整偏心K和刚体相机直接计算Mahalanobis射线响应。预期中心alpha约0.1，原后端给0，关闭错误裁剪的CUDA给0.100000024，整图RGB/alpha最大差2.235e-8。5项T0b控制通过，解释的是新合成零梯度错误。这是响应数值差，不是重投影像素误差。", "生产修复仍待做；本轮低层连接只是诊断，没有替换native_portrait_renderer122/scene包约定或当前查看器。该后端尚未训练历史人物模型，不能称这是历史多鼻子/全身白边的根因。人物独立材料、静止性、实际输出K及全帧注册仍不通过，当前7帧面部最大13.05px。", "下一步登记响应一致的保守包围范围适配，明确新后端版本并同步原生包回读，验证正常尺度/SH/协方差/针孔/KB4/梯度/回读和这同一小高斯反例。不要修改运行源码、静默替换后端或整套重复T0a，不新增人体拟合/训练或参数扫描。原生生产接口、实拍全分辨率容量、增密、初始化与训练准入尚未完成。119历史点云保持。", "来源：robust-portrait-122/native-training-integration/plan.json、existing-backend-path/result.json和synthetic_comparison.png、subpixel-culling-diagnosis/plan.json/result.json与review.json。以下保留历史诊断："].concat([previous]).join('\n\n');
const candidatePath=build+'/nativeTraining122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生训练接入_v3.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeTraining122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeTraining122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeTraining122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeTraining122-final-validation-v3.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeTraining122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeTraining122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
