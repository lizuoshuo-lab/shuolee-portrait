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
const prior=await read(build+'/surface123-camera-support-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-skin-texture-probe/post-geometry-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-material-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-material-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-material-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('SMPL-X错位与面部约束缺口','皮肤材料点仍有跨视角错位');
replace('灰衣95帧：相机优化中9个脸区观测，逐点审阅均为背景。','灰衣：35个留出视角中位18.9像素，最大32.3像素。');
replace('25条脸区候选轨迹未进入最终相机优化；人体拟合仍约44像素。','同帧地面观测0.16–0.96像素，相机残差与人物微动仍需分离。');
replace('鼻部候选轨迹的原生三视角。全图误差0.39像素不能证明面部准确。','青圈：皮肤点观测。橙十字：固定相机下同一静态三维点的投影。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'原帧58、60、62、65的皮肤材料点观测与冻结相机静态三维点投影，图内没有文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["灰衣SMPL-X面部拟合残差中位约44像素，原粗门槛仅检查全身，未完成精细人脸验收。25条FIT和3条HELD脸区候选的全部279个观测已逐一查看，均为眼睑/眼裂、唇齿嘴角或轮廓附近，未认证为静止皮肤材料。原3条留出候选12次检查有11次超过1像素，最大27.88像素。新增一次皮肤色素点诊断：在原生帧40–80追踪同一可见深色皮肤标记，冻结全部相机，用预登记6帧只拟合一个静态三维点。其余35帧全部超过10像素，中位18.93、最大32.35像素。同一帧58/60/62/65中，皮肤点分别17.12/13.43/11.74/13.69像素，现有地面留出点分别0.96/0.16/0.44/0.69像素。这表明当前观测、相机与静态人像假设存在不一致，不能只靠调整SMPL-X体型解决。边界必须保留：皮肤跟踪仍可能有数像素中心漂移，0.452像素光流往返误差不等于物理定位精度；地面控制只有一个材料候选、4帧，位置与深度不同。当前证据不能唯一排除K/外参/残余畸变，也不能直接判定全部由头部微动导致。灰衣高斯仍未开始，旧模型保留。", "本次单变量观察变更：由眼睑、嘴角、轮廓候选改为明确可见的皮肤深色标记。原生帧40–80共41帧，6个预登记锚点拟合一个静态三维点，35帧独立于该点拟合。原K/外参、原SIFT分组、原几何、SMPL-X和训练参数未修改。", "皮肤跟踪以31pxLK窗口、3层金字塔、固定阈值运行一次。往返最大0.451923px只反映算法重复性，原生像素检查显示仍可能有数像素中心漂移。不能声称物理定位已经小于1像素。", "预先6锚点的静态三维点拟合中位15.944px。35留出中位18.925px、P9027.613px、最大32.345px，全部大于10px。41个点深度均为正。点拟合仅3参数、13次评估，不作为训练初始化。", "同帧58/60/62/65皮肤残差17.119/13.431/11.743/13.687px，原地面轨迹留出0.962/0.159/0.444/0.686px。两侧检查点均未参与各自三维点估计。地面控制仅一个候选，在图像边缘，估计深度约为皮肤点2倍，不能据此证明全部相机精度。", "已实际检查25FIT与3HELD的全279节点、皮肤全41帧、4个皮肤投影以及地面全9节点。没有按误差删点或将留出转入拟合。", "手工初始像素坐标误写到邻近皮肤，在第一次LK前经64px原生裁片和最近邻放大纠正。原计划错误坐标文件保留，未发生多次跟踪或阈值搜索。", "灰衣高斯未开始，WhiteT及winterhat旧18k结果仍为质量未通过的研究结果。未验证00、独立SIBR或Spark。只有报告第52页更新，其他51页保留。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\grayT-skin-texture-probe\\geometry-result.json", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\grayT-material-identity-review\\actual-review.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-material-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123皮肤材料点核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-material-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_material.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-material-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-material-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-material-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-material-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
