import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/surface123-simcc-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-facial-articulation-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-articulation-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-articulation-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-articulation-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('连续解码未解决SMPL-X错位','增加表情仍未修复SMPL-X错位');
replace('6帧鼻点位置变化中位0.84像素，最大1.82像素。','固定相机对照：面部残差中位43.99→48.69像素。');
replace('逆仿射误差小于0.0001像素，遮挡与材料对应仍需验证。','同视频两轮内参一致，皮肤留出误差仍约19像素。');
replace('第80帧：原图、原解码青点、连续解码橙点。原拟合保持。','原图／观测青点／原模型紫点／共享表情候选橙点。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第80帧四个原生320像素裁片，依次原图、观测青点、原模型紫点、共享表情与下颌候选橙点。图内没有文字，没有形变。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["已确认拟合源码的表达限制：29个鼻嘴点参与损失，但SMPL-X的10维表情和3维下颌未进入优化器，保持默认零值。固定95帧相机、身体、体型、根姿态、观测和权重，只增加一组共享表情/下颌，完成一次CPU对照；回放投影差0.000339px、梯度差9.61e-11。加权损失4.775→4.646（降2.70%），2732个有效语义残差中位43.99→48.69px；9个表情系数接近边界。6帧原生图实际审阅仍有侧脸鼻嘴落在脸颊，因此不能将开启表情当成对齐修复，也不能把微小损失下降等同质量提高。同时只读核对00对话独立完成的同视频95帧：换回原生坐标后K一致至2.3e-13，相机相对旋转却中位差1.54°、最大6.89°。00语义面部残差较低（原生17.71px），但相同皮肤标记35留出残差18.83→19.15px，两组35/35均≥1px。分辨率、特征集、遮罩和拟合代码同时变化，这不是单因素归因；较低的语义点残差仍没有证明物理对齐。研究侧未改变历史相机、模型或开启新高斯训练。", "当前证据支持“相机—静态人物—语义观测尚不一致”，不是遗漏jaw/expression单独造成的。保留本次负对照，不能放大表情边界继续吸收误差。接下来只核对独立复现的背景支持、相机条件与同一物理材料轨迹；先区分相机差异与头部相对运动，不将更低拟合损失作为训练准入。已有近景控制主要共面，头部高度非共面场景支持仍是具体证据缺口。", "独立CPU单项对照，95帧、29个活跃鼻嘴语义点，2732个正权重观测。共享身体体型、根姿态、世界位置/尺度、K/E、所有观测与权重冻结。仅变化10维共享表情与3维共享下颌，用3*tanh和0.35*tanh限制；不是每帧动态模型，不是新的正式人物拟合。", "零参数回放与原输出的最大投影差0.000338710px，顶点差5.45e-7世界单位，13参数自动微分与中央差分的目标梯度最大差9.61e-11。仅验证计算，没有验证真实标定。", "48次目标计算，单次5.48秒CPU。加权鲁棒目标4.7750856→4.6460193（下降2.7029%）。脸部语义残差中位43.9922→48.6926px；鼻部45.8649→46.3916px；嘴部43.3939→50.2146px。不能用鲁棒损失下降代替逐点质量结论。", "9个表情维接近预先登记的边界，未放宽边界、未重新选点、未再次运行。6帧原生320px对照已实际审阅，侧脸鼻嘴仍落在脸颊，未发布候选模型，未开启新高斯训练。", "00对话独立复现路径：/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2。研究侧只读，没有改动它的源码或结果。该对话独立负责其18k训练、导出与质量审阅。", "两路源帧序号/时间一致，95帧。研究侧1520x2704，00为899x1600。严格半像素变换回原生坐标，有效K最大差2.2737e-13。第一帧外参均为单位矩阵；相对旋转差中位1.54359度、P90为6.62733度、最大6.88835度。这个差异不是相机真值误差，也不是人的运动角度。", "两路同语义观测差中位1.25164px、最大4.96115px；00面部语义拟合残差原生17.7061px，低于研究侧43.9921px。但输入分辨率、特征、遮罩和代码同时变化，不是单因素实验。", "复用既有M0皮肤候选的41个原生观测，仍用40/48/56/64/72/80六个ANCHOR，只在各自冻结相机下拟合一个静态3D点；35个HELD全部保留、不进入点拟合。研究侧中位18.8325px最大33.0977px，00中位19.1477px最大30.2174px。两边35/35均>=1px，不能称亚像素对齐。既有留出点被复用，是次级验证，不是新增独立真值。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\facial-articulation-conclusion.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-articulation-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123表情与跨轮核查.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-articulation-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_articulation.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-articulation-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-articulation-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-articulation-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-articulation-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
