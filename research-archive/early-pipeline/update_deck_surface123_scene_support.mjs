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
const prior=await read(build+'/surface123-articulation-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-scene-support-cross-run-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-scene-support-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-scene-support-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-scene-support-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('增加表情仍未修复SMPL-X错位','头部附近场景验证仍不充分');
replace('固定相机对照：面部残差中位43.99→48.69像素。','两组已有留出残差中位1.19和1.50像素，材料身份待核实。');
replace('同视频两轮内参一致，皮肤留出误差仍约19像素。','局部区域实际BA支持为0条与1条，尚不能据此判定外参错误。');
replace('原图／观测青点／原模型紫点／共享表情候选橙点。','00复现的一条栏杆轨迹：54、58、61帧。原图保持比例。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'00复现实际进入相机BA的同一栏杆候选轨迹，54、58、61帧的原生256像素方形裁片。青圈定位，没有图内文字，没有形变。材料身份未通过精细验证。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["两组灰衣人物相机都存在尚未通过的局部验证：预登记的头部附近图像区域、40至80帧中，研究侧实际BA支持为0条，00复现仅1条、7个观测。该区域原有留出残差中位分别为1.19和1.50原生像素，22与50个重建观测出现非正深度。实际查看14组三视角裁片后，重复栏杆、树叶和平滑金属的材料身份仍有歧义。两条原始FIT族的错误末端在BA前已剔除，不能把它们称为已参与优化的错误点。00的特征确实位于人物排除遮罩之外。当前发现的是头部附近独立场景验证不足，不是已经证明外参错误；远处或其他区域的特征也能约束相机。相机误差与人物微动仍未唯一分离。", "先检查现有原图中固定栏杆/建筑的直线方向是否能提供独立旋转检验；只做材料与可辨识性检查，不重新提特征、不拟合新相机。这一检验只可能约束部分旋转，不能单独确定平移或证明内参正确。若不存在可确认的固定方向，保留缺口。灰衣正式静态训练继续保持未准入。", "00以899×1600工作，研究侧为1520×2704。同样3工作像素门槛在00约为5.07原生像素；分辨率、特征、遮罩和筛选容差共同变化，不能作单变量归因。", "只读审阅两轮全部95帧的原始轨迹、最终BA成员及既有留出，未提取新特征、未拟合三维点、未优化相机。全部源哈希保持。预登记头部附近图像ROI来自帧40至80，不代表物理深度。", "研究侧此区域实际BA为0条/0观测，00复现1条/7观测。已有留出分别284条/398观测与556条/839观测。原生像素误差中位1.1942364494和1.5033483312，P90为26.45491548和31.19300769，最大279.85318895和14045.83494935，至少1像素220/398和510/839。包含非正估计深度22/50个观测，未按误差筛除。", "非正深度或较大留出误差可能由错误材料对应、弱三角化或相机误差引起，不是单一外参错误的证明。局部支持少也不是相机不可估计的充分条件，其他区域固定点仍能约束外参。", "实际审阅14组三视角裁片（42个裁片位置），含11组原始分量、3组实际BA节点。原始FIT树木与路面轨迹的错误末端分别57/54帧未进入BA，不能将原始错误末端归为优化污染。实际树叶仍未证明静止，栏杆具有重复和平滑纹理歧义。", "00背景mask排除检测到的人物后，全部保存特征均处于允许区域，违规数0。背景不等于天然静止或材料身份确定。", "本页图像为00实际BA的d0326813da630086ad4eca3286b6246e569944c2f25a451b3a3609222779777e轨迹，全部7帧54/55/56/58/59/60/61保留，抽检54/58/61。图中支架固定结构看似相同，亚像素材料对应尚未认证。", "当前灰衣研究训练未准入，历史白T和蓝衣研究模型保留。00对话另有授权的独立实验，不作为本静态目标已通过。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\scene-support-cross-run-conclusion.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-scene-support-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123场景支持核查.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-scene-support-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_scene_support.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-scene-support-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-scene-support-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-scene-support-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-scene-support-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
