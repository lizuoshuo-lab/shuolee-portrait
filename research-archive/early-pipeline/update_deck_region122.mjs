import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/material-region-comparison/same_camera_face_clothing_ground_native.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/projection122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/region122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('残余畸变与共享三维对齐','人像与背景的跨视角偏差');
replace('上排针孔','三排依次为脸、上衣和地面。绿圈为观测，紫色为投影。');
replace('修复求解后','固定相机后，脸与上衣候选中位误差为3.11和3.02像素。');
replace('加入径向项后','共同偏移仍在。原脚部区域混入裙摆和地面，根因尚未唯一确定。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'相同固定相机的脸部上衣和地面投影对照，原生裁片无文字',fit:'contain'});
image.frame=frame;image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N7/N8：按用户最新判断将第一组K对应4000×3000、下采样、居中裁剪作为主处理路径。此前N5/N6已经采用4000基准。图示4096与1532保留为历史示意数值，实际新版MP4为1520×2704。YAML尺寸本身不能独立证明导出K，这仍为明确工作假设。", "本页三排依次为face seed5、clothing seed249、ground seed9。六列为230、246、262、294、310、326帧。每格160×160原生像素，等比显示，没有图中文字或二维图像拉齐。绿圈为观测，紫十字为由其他视角三角化后投影的预测，橙线为残差。278只在本页排布省略，完整七视角及全部观测保留在报告。", "N7只读分析N5c保存的935个FIT点，PCA主平面绝对距离中位0.10856个基线单位，脸部目标排除三角化点离该平面中位5.25034。第一基线固定1，不是米。存在离面背景点，不能把全背景说成平面。背景形状和深度依赖当前K/相机，不是独立真实深度。最大视差角的轨迹中位11.64度，不能称所有轨迹没有视差。", "最近留出背景点距离每视角脸部中心337至720原生像素。7张最近背景裁片已逐张查看，部分具有水面波纹、反光和岸边边界，不能自动作静止物理材料。没有按残差删点。", "57脸候选的逐视角共同平均残差占总平方和96.9307%。这只是统计分解，没有把该向量用于校正观测。固定相机、假定支持点各向同性独立1px误差的线性敏感度，中位最大轴0.631px、最大1.591px，未包括相机误差、偏差和运动，不能当实测不确定性。", "N8复用N1全部307轨迹、原有效性状态和N5c同7相机及完整K。没有新提点、重跟踪、优化K/相机、动态网络或训练几何。每目标使用其余有效视角DLT，至少2个支持视角。全部错误和缺失保留。与旧57脸预测逐值复核误差小于1e-6px。", "脸部9轨迹共63槽位，57可计算、6原跟踪无效。57误差中位3.105781px、P90 8.286258px、最大10.142233px，全部>=1px。蓝上衣38轨迹266槽位全可计算，中位3.022049px、P90 8.232741px、最大8.697617px，255个>=1px。", "原footwear_candidate20轨迹140槽位全可计算，中位3.660489px、P90 11.514281px、最大24.890418px，121个>=1px。但原ROI覆盖裙摆、靴筒边缘和地面，这不是脚部解剖准确度。全部20锚点和预登记6条跨视角裁片已查看。历史N1把287/291描述为脚部材料并以落到裙摆证明错配的语义不充分，现记录更正而保留原始数据。", "地面240轨迹1680槽位，其中1588可计算、86原跟踪无效、6支持不足。中位0.446233px、P90 0.940609px、最大2.792084px，139个>=1px。这些N1地面与N5背景SIFT可能对应相同物理材料，因此仅为描述性同场景负对照，不冒充全新独立留出。", "第294帧脸部平均残差(+7.877,+0.842)px，上衣(+8.252,+0.903)px，地面(+0.100,+0.018)px。脸与上衣存在相近同向偏移，说明现象不限于面部表情点。其他视角竖直偏差并不完全相同，不能据此说整个人是同一刚体平移。", "已查看5张预登记裁片页，包含全部9脸候选和各区域均匀索引选出的6条，共27条。衣物样本仍在相应蓝色上衣材料区，但毛绒织物、金属扣反光和精细像素对应仍有测量不确定性。所有307轨迹逐点物理身份尚未认证。", "结论：离面相机支持不足、真实对应偏差和人物微动仍是竞争解释，尚未唯一归因。单径向参数失败不能排除更一般输出投影残差。保留4000主假设，先核查既有离面静止材料与实际脚部覆盖，再登记单变量验证。未启动新人像训练，未发布相机/PLY，119历史模型保留，严格<1px验收未通过。", "证据：robust-portrait-122/video-update-20261027/background-geometry/residual-structure/result.json与visual-review.json；material-region-comparison/result.json、observations.json、visual-review.json、patch-index.json、foot_candidate_anchor_all20-index.json。", "以下历史说明保留溯源："].concat([previous]).join('\n\n');
const candidatePath=build+'/region122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122区域残差诊断.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/region122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_region122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/region122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/region122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/region122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide'+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/region122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
