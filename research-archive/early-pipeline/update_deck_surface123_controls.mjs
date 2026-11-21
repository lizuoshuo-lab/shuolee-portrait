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
const prior=await read(build+'/surface123-material-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-skin-localization-review/post-geometry-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-controls-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-controls-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-controls-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('皮肤材料点仍有跨视角错位','修正局部定位后，静态错位仍在');
replace('灰衣：35个留出视角中位18.9像素，最大32.3像素。','灰衣：35个留出视角中位18.83像素，最大33.10像素。');
replace('同帧地面观测0.16–0.96像素，相机残差与人物微动仍需分离。','定位最大修正2.91像素；近景路面18次检查仍有2次超过1像素。');
replace('青圈：皮肤点观测。橙十字：固定相机下同一静态三维点的投影。','青圈：亮度定位。橙十字：固定相机下静态点投影；相机与微动未分离。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'原帧58、60、62、65的皮肤材料点观测与冻结相机静态三维点投影，图内没有文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["SMPL-X仍未对齐：灰衣参与拟合的面部检测残差中位43.99像素。原几何门槛仅检查全身，允许身体中位误差约50–53像素，不能用于精细人脸验收。本轮补查6条原留出场景轨迹的全部110个节点：5条路面候选在同帧58/60/62/65共18次检查，中位0.687、最大2.446像素，2次超过1像素；另一条为远处树叶，不认证静止。最接近皮肤点估计深度的路面轨迹也有2次超过1像素，而且全部可信候选仍在地面，没有靠近头部高度的非共面固定结构，不能宣布相机亚像素通过。另外固定全部相机和原41帧，仅将皮肤色素点定位改为13px原生亮度模型。定位相对原光流最大修正2.910像素，全部41点实际审阅未见跨材料跳接。仍用原6锚帧拟合一个静态三维点后，35个原留出视角中位18.833、最大33.098像素，全部大于10像素；原光流对应中位18.925像素。局部定位修正未消除大错位。结论：当前相机、材料观测与静态人像假设仍不一致，不能只调整SMPL-X体型或继续高斯迭代。精确K/外参误差与人物局部运动尚未唯一分离。亮度模型的条件置信区间不是物理真值。灰衣高斯未启动，原点云和相机保留。", "本页为两项分开登记的控制结果。近景仅复用原HELD轨迹、原3D点和原K/E，无新提取或几何优化。按同帧覆盖、估计深度距离、ID排序，读取误差前审阅全110节点。六条中五条为路面候选，一条为树叶；树叶误差完整保留，但不能当静止负对照。", "路面在58/60/62/65共有18个检查，16个<1px，2个>=1px，中位0.686861、最大2.445777px。最接近旧皮肤点估计深度的路面比值1.138–1.177，三个检查0.706/1.850/2.446px。深度为估计，非物理测量；这些控制仍主要共面。", "观测定位单变量：13x13原生亮度斑点模型（负椭圆高斯+平面背景，8参数），中心以旧LK为初值且边界±3px，41帧一次运行，无调窗、重试或误差筛点。位置变化中位0.428064px、最大2.909808px。", "合成模型正确性：无噪声中心误差2.48e-16px，固定噪声样本误差0.006247px；这不验证真实材料亚像素精度。亮度Hessian条件95%半径中位0.247、最大0.352px，只在模型和噪声假设下有效。", "全部41帧原生64px及最近邻2倍定位图实际审阅，未见跨材料跳接。随后原6锚帧拟合一个静态3D点，35HELD均未用于点估计。留出中位18.832509、P9027.350449、最大33.097690px，35个均>=10px。旧LK对应中位18.925094px，局部观测修正没有消除大错位。", "原生同帧58/60/62/65亮度定位静态投影误差17.962/14.345/12.691/14.438px。照片裁片均256x256，未变形；图内无文字，青圈/橙十字仅为对照标记。", "SMPL-X帧映射与投影数值复核通过不等于拟合正确。灰衣面部检测残差中位43.99px，原粗身体门槛允许50–53px且未精细验脸。旧报告已明确quality=false；本轮不将它改称物理通过。", "无新相机、SMPL-X或高斯训练；当前WhiteT和winterhat仍是未通过质量验收的研究模型。未实测公司00、SIBRGUI或独立Spark页。仅更新同一52页PPT的第52页，前51页不变。", "下一步：在后续本地队列增加与粗初始化分开的对齐准入检查。几何原因仍缺近头部非共面固定点及多处独立刚性皮肤材料证据，不能声称全部由相机或头部运动引起。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\grayT-near-scene-support-review\\result.json", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\grayT-skin-localization-review\\geometry-result.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-controls-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123近景与定位控制.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-controls-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_controls.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-controls-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-controls-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-controls-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-controls-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
