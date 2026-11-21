import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/material-quarantine/same_native_before_after.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/region122v2-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/quarantine122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('人像与背景的跨视角偏差','跨视角错配与重影根因');
replace('三排依次为脸','绿圈为观测，紫色为隔离前投影，蓝色为隔离后投影。');
replace('固定相机后','隔离已核实的背景错配后，面部中位误差仍为3.14像素。');
replace('共同偏移仍在','单条错配不是主因。继续检查人物周围的场景约束。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'隔离背景错配前后相同原生观测的脸部上衣地面对照',fit:'contain'});
image.frame=frame;image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
'2026-10-28 N9/N10与N11准备。继续按用户说明采用第一组K对应4000×3000、下采样和居中裁剪；此前N5/N6已使用4000基准，不把4096示意图视为新的标定事实。实际导出K与裁剪取整尚无独立记录。',
'N9只读审阅935条FIT轨迹合并后的608个近重复族。按事先登记的条件PCA平面绝对距离最高10%选取53族代表，9页159张原生160像素裁片均实际查看。52/53族最大视差小于3度，选集视差中位1.189934度、条件深度中位74.5526个首基线单位。依赖当前K和相机的估计不是独立真实深度或米。未用重投影误差挑选或剔除观测。',
'唯一大视差代表为族399，最大视差11.3034度。实际查看三个320像素上下文裁片：246和262帧中心位于近沙洲/植被边界；278帧中心跳到远岸，原近沙洲仍在该裁片左下。三视角循环匹配仍可能连接不同物理材料。其他52族不能因为未发现明显跳转就认证为静止真值。',
'N10唯一因果变化为隔离已确认错误的完整FIT族399，即1轨迹3观测。原文件保留。其余934条FIT、190条整轨留出、7帧、完整固定K、Huber1与优化设置不变。由原始二维观测重新计算首对PoseLib相对姿态、PnP及DLT，未读取旧优化位姿或几何作为初值。新首对R/t与旧原始首对逐值一致，全部934条FIT初始化成功，全部681个冻结背景留出目标保留。',
'本页三排为face seed5、clothing seed249、ground seed9；六列为230、246、262、294、310、326。每格160×160原生像素，图像没有变形或平移，2:1全图等比显示。绿圈为观测，紫色十字为隔离前投影，蓝色十字为隔离后投影，两种预测几乎重合。每个目标均由其他有效视角三角化后投影，未用目标本身拟合点。',
'N10脸部57个有效候选中位误差由3.105780899变为3.143951708像素，最大10.101921704，全部57个仍不小于1像素；原63槽位中的6个原跟踪无效仍保留。上衣266个中位3.022049072变3.011339990，最大8.699291677。混合脚部ROI140个中位3.660488619变3.662915719，最大24.909323399，该ROI不是纯脚部解剖材料。地面1588个中位0.446233296变0.447523771，最大2.805201723；地面不冒充与拟合材料无重叠的独立真值。冻结背景681个中位0.434743489变0.435557541，最大3.239984121。',
'N10新FIT最终cost163.125788；相机最大旋转变化0.013258度、平移变化0.002521个首基线单位。拟合cost下降没有改善独立候选偏差。结论是族399的材料错配成立，但单独隔离它不能解释主要的人像重影，不能据此唯一归因于K、外参、人物运动或特征精度。脸部材料身份与测量不确定性仍未全部认证，严格逐点小于1像素的准入未通过。',
'N11仅完成场景支持遮罩准备，不是新相机或新训练。旧人物排除矩形为[350,900,1100,2400]，把周围大量背景一并排除。使用已存在YOLOv8n与SAM2.1 tiny在CPU生成7帧人物轮廓，原生像素膨胀32。7张全图覆盖均实际审阅，帽子、头发、外衣、裙摆、红包和靴子包含于排除区；这不是像素级人工真值。每帧恢复约75至77万原矩形背景像素，其中仍包含天空、水面、植被，面积不等于静止物理点数量。',
'下一项必须先登记一次仅在恢复背景区域的原生特征补提，保持K与匹配设置，隔离旧轨迹近重复族及留出泄漏，审阅材料支持后再判断是否允许固定K新鲜初始化的相机诊断。当前尚未补提特征或启动新BA，不把遮罩通过当作几何改善。未启动新人像训练或发布新PLY/相机，119历史模型保持。',
'证据：background-geometry/offplane-support/result.json、visual-review.json、track399-context-index.json；material-quarantine/result.json、comparison-controls.json、visual-review.json、figure-index.json；scene-support-masks/result.json与visual-review.json。',
'以下历史说明保留溯源：',previous].join('\n\n');
const candidatePath=build+'/quarantine122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122材料隔离诊断.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/quarantine122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_quarantine122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/quarantine122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/quarantine122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/quarantine122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/quarantine122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
