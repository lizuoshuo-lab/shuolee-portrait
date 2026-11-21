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
const prior=await read(build+'/surface123-controls-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-head-rigid-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-head-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-head-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-head-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('修正局部定位后，静态错位仍在','局部姿态能减错，相机仍不能替换');
replace('灰衣：35个留出视角中位18.83像素，最大33.10像素。','灰衣：皮肤留出中位18.83→6.93像素，33/35仍超1像素。');
replace('定位最大修正2.91像素；近景路面18次检查仍有2次超过1像素。','局部调整若套给场景相机，路面中位误差0.69→326.90像素。');
replace('青圈：亮度定位。橙十字：固定相机下静态点投影；相机与微动未分离。','青圈：皮肤观测；橙：静态；绿：头部逐帧。仍未达到亚像素验收。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'原帧58、60、62、65的皮肤观测、静态与鼻部FIT代理逐帧刚体预测，原生裁片无文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["灰衣人物的SMPL-X仍未对齐。已有鼻部语义点只作FIT代理、皮肤材料点独立验证：固定原K/外参和头部形状，从所有帧共享刚体调整改为每帧刚体调整后，35个皮肤留出视角中位误差由18.833降到6.934像素，最大仍12.217像素，33/35达到或超过1像素。这说明当前静态头部假设存在可被逐帧刚体自由度吸收的不一致，但没有证明全部来自真实头部运动。拟合出的相对旋转最大50.69度，不能称作已经测准的“小幅微动”；鼻部形状近似、遮挡推测点及局部相机误差也会被吸收。将这组局部变化按同一参考帧解释为整幅场景的相机修正，会把18次路面检查的中位误差由0.687放大到326.903像素，不能直接替换场景相机。实际审阅全部41帧鼻部原生裁片，侧脸末段部分检测点贴近或越过可见鼻部轮廓，原权重仍有效；这些语义点不能冒充物理材料对应。头部形状、真实运动和精确相机仍未唯一分离。当前所有精细准入仍拒绝，灰衣高斯未开始；历史模型、相机和训练图均未改动。", "同一个预登记诊断：固定原相机K/E、SMPL-X头部9鼻部三维点、原权重和41帧。只改变所有帧共享6DoF与每帧6DoF。鼻部代理是FIT，独立皮肤定位从未进入姿态优化。此前35皮肤HELD已用于研究对照，不声称新未见测试集。", "先核对帧和IBUG27..35映射，独立投影最大差0.000327px；41帧9点都具数值Jacobian rank6，不能等同物理可辨识。合成SE3及参考帧控制通过后，一次CPU2.563秒完成1共享+41逐帧优化，无新相机/特征/模型训练。", "所有41帧原生鼻部裁片在拟合前实际审阅，侧面末段部分鼻梁检测点越过可见面部边界。没有按残差删点，也没有把这些语义代理称物理真值。", "共享刚体变换不会改善静态材料点多视角一致性；重估皮肤三维点后与旧静态投影最大差1.00e-6px。逐帧刚体后35HELD中位6.933834、P9010.810861、最大12.216955；只有2个<1px，33个>=1px。", "相对参考40的拟合头部旋转中位19.3037度，最大50.6867度。这是模型所需修正，不能报告为准确的真实人物微动。模板形状、遮挡代理与相机误差均可被局部自由度吸收。", "场景反事实只将E_i H_i inv(H_40)应用于原先保存的HELD背景三维点；不优化点或K。18次路面误差中位0.686861→326.902566，最大397.025960像素。只排斥这组头部修正作为整场景相机的解释，不能排除所有相机模型。", "工程修复：local123拟合前、表面初始化前、训练前和1000步复核均增加精细准入；27项合成边界合同和4调用入口检查通过，实际WSL三人物两阶段共6次拒绝缺失证据。没有用合成测试授权真实训练。", "当前灰衣仍未启动高斯，WhiteT/winterhat历史18k研究模型保留quality=false。未改原图/相机/拟合/PLY，未实测00、官方SIBR或独立SparkGUI。", "下一步只做原生可见性与多物理材料支持核查，不调整点权重、不复跑本次逐帧姿态。头部真实运动、精确标定及模板形状尚未唯一分离。", "结果：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\grayT-head-rigid-review\\result.json", "工程证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\alignment-admission-conclusion.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-head-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123头部刚体诊断.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-head-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_head.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-head-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-head-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-head-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-head-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
