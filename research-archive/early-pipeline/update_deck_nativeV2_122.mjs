import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-renderer-v2';
const figure=root+'/native_v2_views.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/nativeTraining122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.driver_controls!==41||review.CLI_controls!==2)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeV2-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('原生训练与小高斯剔除错误','原生渲染修复与点云回读');
replace('96点合成：','上排针孔，下排已知鱼眼。三列为相同三组相机。');
replace('20万点被误剔除','20万点恢复30,681点更新。PLY回读最大响应差4.59×10⁻⁶。');
replace('生产修复待做','组件41项和命令行2项通过。交互界面与实拍标定仍待验收。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'6颗已知各向异性高斯在3组相机下的原生图像，上排针孔下排KB4；无文字叠加',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height*2)/2,top:frame.top,width:frame.height*2,height:frame.height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-28 T1原生渲染修复。延续第一组YAML在4000×3000下的K，经下采样和居中裁剪映射到MP4。YAML尺寸证明参数参考像素坐标，实际K_rect仍是条件假设。本轮没有修改内参、外参、人物图像或真实材料对应，也没有再次去畸变或重估K。',
 'T0b已证明gsplat1.5.3的UT提前剔除将opacity乘二维模糊补偿，但classic ray3D响应使用原opacity，因此原20万小高斯全被误删。T1独立v2适配仅给UT投影传入opacity=None，保持eps2d0.3、3.33σ几何范围、相机、初始参数、1/255射线阈值、原始opacity及CUDA源码不变。旧v1适配、当前运行原生服务和历史119人物源码均保留。',
 'v2渲染器、v2场景包与v2训练组件共同使用版本122-ray3d-geometry-bounds-v2。包记录完整K、唯一w2c、坐标/像素约定、完整协方差和SH、ray3D/UT参数、gsplat1.5.3与torch2.7.1cu128、适配/加载器/验证器/编解码器/包装器及CUDA源码树SHA。新旧schema互相拒绝，不会静默改变旧模型的求值规则。',
 '一次预登记GPU驱动测试在52.30秒完成41项检查。重用6颗各向异性SH3合成高斯、3完整偏心K相机，针孔和已知KB4均与所有高斯覆盖所有图像tile的射线参考精确一致。该参考只绕过UT剔除，仍共享CUDA射线内核。独立NumPy公式证据来自已完成T0b，不把共享后端当独立物理真值。',
 '图中每格320×240，原生输出无新增文字，上排针孔下排KB4，三列对应相同3相机。实际检查六格中6颗彩色各向异性椭球和随视角变化的颜色，无全黑遗漏。样本接近视野中心，不能推广到所有大视角/边缘鱼眼。UT3.33σ范围仍是近似，本次通过不是任意镜头的全支持证明。',
 '12项固定中心差分检查覆盖两个投影中的means x/z、log尺度、原始四元数分量、opacity logit及SH系数，固定epsilon0.001，阈值abs1e-4加2%相对误差，全部通过。最大绝对差0.00011592，不能宣称在所有遮挡/可见性不连续边界都正确。',
 '同一冻结20万点64×64反例直接输出与T0b仅关闭opacity剔除的诊断完全相同。200000点获得范围，30681点有非零位置梯度，并由空Adam一步实际更新，所有属性梯度有限。没有放大点、提高透明度、改相机或扫描eps。更新后完整PLY/相机回读最大RGB/alpha差4.58956e-6。原生64×64图可见小椭球的暗内部及彩色边缘，远非高分辨率人物外观验收。',
 '针孔与KB4正常尺度包的最大回读差分别2.08616e-6、1.84774e-6。GPU驱动退出后两个独立CLI进程逐个加载包中camera0，最大响应差分别1.90735e-6、1.43051e-6。响应差单位是RGB/alpha数值，不是重投影像素。v2实际交互界面尚未接入验收。',
 '96点合成组件重新创建空优化器，同一4记录视角12步各3次，目标由0.15449578降到0.04362841，比值0.28239225。完整训练包回读差8.34465e-7。图像仍明显模糊，代表接线通过，不能冒充真实人物或与T0a120步比较。1000步缺审阅回调会拒绝，审阅拒绝于第2步停止，已有参数修改或重复优化器会拒绝。完整输入来源准入仍另需验证。',
 '拒绝控制涵盖已去畸变图重新套鱼眼、8未知系数冒充KB4、相机梯度、反射外参、不完整SH、旧schema、错误bounds、后端SHA不符、资产SHA和越界路径。41项驱动加2CLI均通过。旧代码及冻结数据哈希不变。本轮不重跑旧人物或旧相机候选。',
 '目前实拍材料身份、静止性、输出K和完整帧注册仍未通过。N12g7帧面部探索性最大误差约13.049px，并无真实逐点小于1px证据。新后端未训练历史119，因此本次修复不能解释历史多鼻子/白边或标注已解决。4人物跨人物与新视频泛化也未通过。',
 '下一步独立登记v2原生交互包选择、记录相机渲染与下载，复用现有合成包，不再运行优化器或修改正在运行的v1服务。后续完整帧训练入口需实际接入材料与相机准入。全分辨率20万容量、自适应密度、人物初始化、Spark/SIBR原生等价及00部署分别保留未通过状态。',
 '来源：robust-portrait-122/native-renderer-v2/plan.json、result.json、cli-result.json、review.json、native_v2_views.png；实现位于native_portrait_renderer122_v2.py、native_scene_package122_v2.py、native_training122_v2.py。以下保留既有诊断历史。',previous
].join('\n\n');
const candidatePath=build+'/nativeV2-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生渲染修复.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeV2-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeV2_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeV2-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeV2-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeV2-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeV2-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
