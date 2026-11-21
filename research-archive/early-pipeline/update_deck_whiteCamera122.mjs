import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../clip03-camera-observability/background_face_feet.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/whiteGeometry122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/whiteCamera122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('固定背景相机下的面部错位','背景相机约束对面部的影响');
replace('三排为面部原图','三排为背景、面部、鞋面留出投影，三列为200、232、280帧。');
replace('背景留出中位数','假设背景观测噪声为1px，面部相机敏感度约为鞋袜7倍。');
replace('面部最大13.05px','这是局部敏感度，尚不能认定为13.05px面部错位的唯一原因。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'背景、面部、鞋面原生留出投影，蓝圈观测，粉色十字预测',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height)/2,top:frame.top,width:frame.height,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N12h。使用第一组YAML在4000×3000参考尺寸下的既有条件K，按原N4下采样与居中裁剪到1520×2704。尺寸字段说明标定参考坐标，真实去畸变输出K尚无独立导出记录。本轮不改内参、相机、三维点或二维观测。", "在N12g完成结果的原216相机规范下只读线性化。40背景FIT族、207观测、414残差坐标。固定216位姿与216到232的单位基线，消去120个三维点变量后，相机35自由度数值满秩，列归一条件数141.10。未用heldout或面部建立信息矩阵，未加入掩盖空方向的正则。", "假设每个背景拟合像素坐标存在独立1px标准差，传播至目标留出三角化预测的主轴标准差增益：背景86目标中位0.531687、最大1.081199；面部41目标中位2.675785、最大3.532888；鞋袜47目标中位0.377976、最大0.493526。面部与鞋袜中位比7.079。单位噪声只用于敏感度归一，实际噪声未知。这些数值不是实测相机误差、置信区间或逐点1px合格结论。", "背景信息矩阵独立确定的最弱模式，在拟合残差总L2变化归一为1时，背景留出投影变化中位0.050934px、最大0.464654px，面部中位0.784479px、最大2.284266px，鞋袜中位0.056924px。此为局部线性响应，通过极小参数扰动有限差分核验。没有用面部误差选方向或拟合新相机，也没有把单位幅度假想相机输出为候选。", "固定点支持存在深度不均衡。FIT地面深度中位9.804874个初始基线单位，视差中位27.639°。非地面固定点深度中位107.938670，视差1.278853°。这是诊断候选的相对几何，不是米制测量，也不保证物理K正确。", "图中背景family282、面部family800、鞋面family1454均为冻结heldout族。三列200、232、280帧，每格128×128原生像素，原SIFT蓝圈与N12g不使用目标观测三角化所得粉十字。图不加文字，不修改训练图。背景按最小可覆盖三帧的familyID选取，面部及鞋面沿用N12g例图，未按残差选择。", "结论：背景、鞋面低残差不足以保证人脸受约束。相机弱约束需要保留为风险，不能直接归因人物微动。本轮也未证明它足以解释全部13.05px面部误差。输出K、残余投影模型偏差、材料定位偏差与人物变化仍未完全分离。未生成新相机、点云或训练。", "数值核验：解析Jacobian相对差1.21e-10；相机消元协方差与全联合SVD相对差3.57e-13；相机传递导数两步长差1.69e-9；原规范与184规范投影差9.09e-13px。2.65秒CPU，一次运行，输入哈希保持。数值控制通过不等于物理标定通过。", "下一步预登记并核验既有原生图像和轨迹中，是否有处于人物同等深度、头部高度的可信固定场景物体及独立留出。若没有，明确证据缺口，不重复BA或用人脸拟合后断言人物微动。", "来源：robust-portrait-122/video-update-20261027/clip03-camera-observability-plan.json，clip03-camera-observability/result.json、FIT-information.json、target-sensitivities.json、structure-support.json、figure-index.json、visual-review.json。以下保留历史诊断："].concat([previous]).join('\n\n');
const candidatePath=build+'/whiteCamera122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122相机空间敏感度.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/whiteCamera122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_whiteCamera122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/whiteCamera122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/whiteCamera122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/whiteCamera122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/whiteCamera122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
