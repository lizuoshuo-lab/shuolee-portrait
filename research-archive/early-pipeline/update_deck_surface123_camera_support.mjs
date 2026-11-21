import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/surface123-alignment-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-camera-support-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-camera-support-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-camera-support-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-camera-support-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('SMPL-X错位与固定相机诊断','SMPL-X错位与面部约束缺口');
replace('灰衣人物95个视角的面部拟合中位残差约44像素。','灰衣95帧：相机优化中9个脸区观测，逐点审阅均为背景。');
replace('放开人体模板后仍约47像素，静态跨视角一致性尚未通过。','25条脸区候选轨迹未进入最终相机优化；人体拟合仍约44像素。');
replace('左为原图，中为参与拟合的检测点，右为SMPL-X投影。原视频第375帧。','鼻部候选轨迹的原生三视角。全图误差0.39像素不能证明面部准确。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),height=360,width=height*figureSpec.width/figureSpec.height;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'同一鼻部候选轨迹的三个原生裁片和独立点位标记，图内没有文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 灰衣相机支持审阅，未改变K、外参、点坐标、原始观测、拟合/留出划分或任何训练参数。灰衣高斯仍未开始。", "灰衣相机优化最终使用53761个观测，其中52179个（97.06%）位于人物遮罩外。面部区域内只有9个观测，属于7条轨迹、覆盖4/95帧；逐个原生裁片与精确点位复核，9个全在墙边或栏杆等背景，脸区与人物mask交集的最终BA观测为0。原始SIFT有556个脸区与mask交集特征；拟合分组有25条全部节点落在该区域的候选轨迹，最终相机BA一条也未保留。抽检3条未入选候选，点位分别在鼻部和眼周，仍未完成全部物理身份或亚像素测量验收。保留的匹配日志没有逐轨迹剔除原因；25条中10条原长度少于最终入选的最短6次观测，但另15条更长，不能把全部丢失归因于数量截断。此外，留出轨迹ee91c8b1f0c2从眼周连接到背景栏杆，已确认跨物体错配；另外两条抽检为嘴角区域及重复栏杆，分别有形变/身份歧义。这解释了为何全图拟合中位0.393px不能作为人脸准确性的证据。它证明流程中存在人体支持丢失与验收缺口，尚不能唯一归因于K、外参或人物微动。灰衣SMPL-X面部检测残差中位43.99px，固定相机并放开模板约束后47.05px；当前人体拟合未通过，灰衣高斯尚未开始。", "53761个最终BA观测中52179个在人物mask外；脸ROI只有9个观测、7条轨迹、4/95帧。全部9个原生裁片和精确点位已查看：均为墙边植物或栏杆，不是脸部物理点。", "原生特征总数467584，脸ROI与mask交集556个。FIT分组含35条涉及该区域的组件，其中25条全部节点在该区域；最终BA的该区域观测数为0。25条只是候选，不能视为全部同名物理真值。", "三条未入选FIT全脸区候选按ID均匀选择后实际查看，点位位于鼻部、额侧/眼眉和眼周。图示为083ced88259e，完整原生图像和源帧在grayT-track-attrition-review/index.json。", "留出轨迹ee91c8b1f0c2从眼周变成背景栏杆，证实跨物体错配。嘴角轨迹344fc01b6ef5仍存在表情变化/语义身份歧义，其帧76检测残差27.884px不单独证明K或外参错误。", "所有高残差与负深度观察保留。背景外区域并非自动静态真值；某些抽样叶片可能运动。低全图FIT残差有原先几何筛选偏差，不能替代独立有效面部逐点<1px验收。", "SMPL-X灰衣活跃面部检测残差中位43.992px；冻结相机和检测观测，独立静态3D点诊断仍47.054px。该结果与本轮支持缺口尚不能唯一分离条件K、外参、检测语义和人物微动。", "保留同一份52页报告，其他51页及历史研究记录不改变。查看器当前WhiteT18k只是已发布研究结果，质量未通过。无新PLY、无公司00部署或SIBR GUI验收。"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-camera-support-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123面部约束核查.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-camera-support-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_camera_support.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-camera-support-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-camera-support-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-camera-support-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-camera-support-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
