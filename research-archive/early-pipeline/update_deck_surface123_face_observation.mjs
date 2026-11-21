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
const prior=await read(build+'/surface123-fixed-direction-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-face-observation-cross-check/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-face-observation-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-face-observation-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-face-observation-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('单一栏杆方向无法验证全部外参','脸部观测改善了误差，但仍不是静态对齐');
replace('两组留出方向RMS约0.57和0.59像素，仍不能保证人脸对齐。','同一皮肤点35留出中位5.12→2.42像素，32个仍≥1像素。');
replace('数学控制：绕栏杆方向额外旋转7°，线方向预测保持不变。','当前眼部权重为0，结果含逐帧变化，不能当作静态验收。');
replace('原图／边缘测量／两套相机预测。最大线偏差2.35像素。','原图／RTMW有效点／FAN有效点／同一皮肤点回放。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第79帧的4个原生320像素裁片：原图、RTMW有效鼻嘴点、FAN有效鼻嘴点、同一皮肤点观测与逐帧变形回放。绿圈观测、紫叉RTMW对照、橙叉FAN对照。图内无文字，未变形。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["00新面部观测对照已只读回放：同一物理皮肤标记的35个原留出视角，从全身RTMW驱动逐帧拟合的中位5.119、最大9.569原生像素，降到脸部FAN驱动的中位2.417、最大3.797像素，但32/35仍≥1像素。独立公式回放差小于0.000058像素，95帧身份、K换算和原观测划分一致，未重新拟合。两条路线仍使用每帧19个头、颈、下颌、表情变量，共1805变量；40–80帧头部参数最大相对差约24.69°与25.79°，不能称已测准的小幅微动或静态模型。当前权重只覆盖鼻部852次、嘴部1900次，眼部和眉部均为0。FAN拟合报告中的1.039像素属于899×1600工作坐标，原生面部语义残差中位实际1.757像素。六个原生裁片显示侧脸鼻嘴观测确有变化，但语义点和重复使用的一处皮肤标记不足以证明全脸材料准确。相机、模板和真实运动仍未唯一分离，本静态路线未准入。", "不重复已完成的FAN/RTMW拟合，也不把逐帧变形结果换成静态模型发布。固定相机下，一处皮肤材料点的静态一致性此前已经失败，单纯换检测器不能消除这一证据。下一步检查尚未完成的R7原生残余相机训练入口与调度接入，推进可独立完成的工程；同时只读审阅00正在运行的结果，保持它与本静态目标的验收分开。", "Company control refitted the6anchor material point with Huber f_scale1 working pixel. It is not numerically identical to the earlier original company00_M0 baseline.", "这是00对话已有两次逐帧拟合的只读回放，没有新特征、推理、三维点拟合、相机优化或高斯训练。全部源哈希保持，95帧与原M0的6anchor/35HELD划分匹配。", "两组回放最大差分别0.00004334和0.00005796原生像素。FAN皮肤留出最小0.6784、中位2.4173、P90 3.5839、最大3.7972像素，32/35仍≥1像素。RTMW中位5.1193、最大9.5693、35/35≥1像素。", "全体2752个有效语义观测中，鼻部852、嘴部1900；眼部和眉部均0。FAN原生语义中位1.757、最大7.250像素，RTMW为2.299、13.636。FAN报告的1.039中位属于899×1600工作坐标，不是原生1520×2704像素。", "每帧19个颈部、头部、下颌、表情变量，共1805。40至80帧FAN颈部参数最大两两差20.77度、头部25.79度；这些包含模型和相机误差吸收，不能冒充真实微动测量。其他身体关节只出现1.2e-7数值差。", "原RTMW拟合未保留fit-script.py，FAN源码哈希7f1771d0fa962d9252f3ad9a479c9ceecfe9c2cb045a7b8545e79355d383785a与报告吻合。未称严格单因素实验。", "6个原生整数裁片41/49/57/65/73/79实际审阅；闭眼和侧脸遮挡存在。该皮肤标记重复用于前序验证，不是新封存材料，也不覆盖全脸或未见人物。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\face-observation-conclusion.json", "41/49帧两组鼻嘴点总体位置接近，口角和鼻翼仍存在差异。", "57帧可见闭眼；当前损失对眼眉没有任何有效观测，不能据鼻嘴值宣称眼部对齐。", "65/73/79帧侧脸的FAN鼻翼与嘴部相对RTMW有明显变化，存在不可见侧语义推测，未认证为物理材料真值。", "第四列为同一眼角附近皮肤标记；橙色回放较接近，但仍有大于1px的偏差。", "RTMW原运行未保留fit-script.py，FAN执行源码与报告哈希吻合，不能声称两次历史运行严格单因素复现。"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-face-observation-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123面部观测回放.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-face-observation-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_face_observation.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-face-observation-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-face-observation-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-face-observation-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-face-observation-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
