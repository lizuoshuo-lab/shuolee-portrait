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
const prior=await read(build+'/surface123-6000-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/winterhat-train-018000-review/actual-review.json');
if(!review.actual_visual_review||review.steps!==18000||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-18000-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-18000-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-18000-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('自由衣物表面训练的6000步检查','自由衣物表面训练的18000步结果');
replace('20万点覆盖穿衣轮廓，侧脸和鞋边仍有重影。','29.4万高斯覆盖衣物，面部与鞋边重影仍未通过。');
replace('自动面部区域留出109/142次误差≥1像素，物理对应尚未验收。','112个相机与点云配套上架，高斯和点云模式均已实测。');
replace('左为全身，右为两种面部视角。每组依次为原图、6000步渲染和实际点中心。','左为全身，右为两种面部视角。每组依次为原图、18000步渲染和实际点中心。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),height=360,width=height*figureSpec.width/figureSpec.height;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'18000步同相机原图、自由表面高斯与实际中心分布，左全身、右上正面、右下侧脸。原图保持比例，图内无文字',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-28 本地表面自由点123，戴帽蓝衣新版视频已从头完成18000步。294054个最终高斯，初始200000个，112个真实注册的预登记训练帧。',
 review.summary,
 '本轮实际审阅全部6幅同相机全身、面部与点中心。原帧0的自动面部框落在帽子，不能作为有效人脸。另逐张查看36个10度间隔环绕渲染，背面明显空洞和破碎，没有背面实拍真值。',
 '最终标准PLY完整协方差和三阶SH逆变换通过，6个相机导出前后原生渲染差异在已登记格式阈值内。所有112个相机的完整K、位姿和原视频帧身份，经标准camera数组及绑定provenance联合回读核验一致。相机格式转换不代表物理标定正确。',
 '已在本机Shuolee查看器用打开文件选择实际ZIP，确认294054个高斯、三阶SH和112个SHA绑定相机，并实际选择原帧330查看Gaussian与points。右侧latest已指向本轮，下载包包含PLY和camera。未实测官方SIBR、独立Spark页面或公司00，未断言跨渲染器像素等价。',
 '冻留出自动面部区域142次观察中109次误差>=1像素，真实材料身份尚未逐点认证。训练结束仍不能声明重影修复或强鲁棒性。新的K仍为条件K，外参、对应、人物微动的唯一根因尚未确定。',
 '白T恤人物的独立93个相机、粗拟合与衣物初始表面已准备并审阅，继续同一队列，其他人物仍独立处理。沙发119作为旧对照保留。',
 '最终模型SHA256=814003a9411e0d6eeb9e4cd83b518b487e6e0a7be229e1811b8cc01f121301af。证据目录local-clothed-surface-123，winterhat-quality-review.json，winterhat-viewer-ui-review.json，winterhat-package-camera-parity.json。',
 '图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),
 '以下保留此前研究备注。',previous
].join('\n\n');
const candidatePath=build+'/surface123-18000-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123自由表面18000步.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-18000-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_18000.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-18000-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-18000-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-18000-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-18000-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
