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
const prior=await read(build+'/surface123-scene-support-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-fixed-direction-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-fixed-direction-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-fixed-direction-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-fixed-direction-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('头部附近场景验证仍不充分','单一栏杆方向无法验证全部外参');
replace('两组已有留出残差中位1.19和1.50像素，材料身份待核实。','两组留出方向RMS约0.57和0.59像素，仍不能保证人脸对齐。');
replace('局部区域实际BA支持为0条与1条，尚不能据此判定外参错误。','数学控制：绕栏杆方向额外旋转7°，线方向预测保持不变。');
replace('00复现的一条栏杆轨迹：54、58、61帧。原图保持比例。','原图／边缘测量／两套相机预测。最大线偏差2.35像素。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第80帧原生裁片，依次为原图、有效边缘测量、研究与00冻结相机方向预测。图内无文字，原图比例不变。预测线通过观测中心，不能检验绝对位置。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["灰衣人物已完成一次冻结相机的栏杆方向控制：56和72帧的6条边缘只用于估计一条公共空间方向，64和80帧的6条边缘、480个采样点留出。研究侧与00相机的留出方向残差RMS分别0.569和0.593原生像素，最大2.200和2.350像素。这里每条预测线都通过观测中心，检验的是方向，不能当成绝对重投影或人脸误差。数学负对照表明，绕这条公共方向旋转7°，预测线方向仍保持不变；任何平移也不进入此检验。原始栏杆边缘的上下半段斜率还相差0.12°至3.88°，其物理直线性和精确平行性未认证。因此这项控制无法选择哪一套相机正确，也无法把人物微动与外参误差分开。相机和SMPL-X均未修改，正式静态训练仍未准入。", "本轮单一方向控制已完成，不追加线段或放宽门槛。下一步只读复核00对话已在运行的灰衣对照及原有皮肤留出，区分它的表观改善与本静态目标的实际准入。相机仍缺能同时约束旋转和平移的可信固定场景证据；后续原生训练工程只推进尚未完成的部分，不重复已完成的控制或用未合格人物启动训练。", "独立方向控制：56和72帧6条anchor边缘，64和80帧6条held边缘，共480个有效原生采样。只为各冻结相机求一个公共三维方向，没有优化相机或三维点。实际审阅4组完成图和4组初始测量图。", "第一轮第64帧的三个上端种子来自阴影边缘，测量在相机比较前停止。保留原始失败记录，只纠正3个起点x，复用其余9条测量。未改变物理边缘、行范围、搜索宽度、门槛或留出划分。", "研究侧held方向误差median0.321704 RMS0.569363 max2.199605像素；00为median0.332691 RMS0.592848 max2.350204像素。这里不是面部误差，也不是完整相机绝对重投影。", "7度公共方向旋转负对照对线系数影响小于7e-13；线方向不能识别这一旋转分量以及任何平移。合成2度俯仰控制被检出，只用于检查实现。", "上下半段边缘方向差0.120372至3.883690度，物理杆弯曲、光照、采样及残余图像扭曲未分离。端点正负0.5像素敏感度0.392970至0.658604度，不是统计置信区间。", "仅2个held视角。没有完成全部95帧相机或人脸小于1像素验收，不判定鱼眼、K、外参或微动中的某一项为唯一原因。正式静态训练未准入，历史人物点云保持。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\fixed-direction-conclusion.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-fixed-direction-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123固定方向核查.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-fixed-direction-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_fixed_direction.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-fixed-direction-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-fixed-direction-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-fixed-direction-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-fixed-direction-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
