import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/static-observation-consensus-081';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='7108a04ddeef34742f2f51692be487ba67d850bb08e33ca13be067a5efeee74c';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json');if(!d.actually_inspected||d.gaussian_optimizer_updates!==0)throw Error('Diagnostic evidence required');
const previous=await read(build+'/camera080-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[51]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records[0];
const edit=(k,value)=>p.resolve(snap.elements[k].id).text.replace(snap.elements[k].text,value);
edit(2,d.ppt_summary);edit(4,d.ppt_caption);edit(5,d.ppt_limit);
p.resolve(snap.id).speakerNotes.text=snap.notes.text+
'\n\n081审阅更新（2026-10-24；优先于前文080的未完成计划与区域简称）：\n'+d.summary+'\n'+d.consensus+'\n'+d.joint+'\n'+d.visual_conclusion+'\n'+d.source_findings+'\n'+d.next_step+
'\n区域口径更正：前文080及早期记录中的“鼻口”主要指检测框派生的下半脸分区，包含下颌轮廓及衣领。它们不是已经确认的固定解剖点、三维真值或独立相机真值。本轮原图接触表实际核查12幅，不是所有轨迹全部人工标注。'+
'\n静态筛选：每条轨迹前后外侧三分之一用于三角化与候选选择，中间留出；最多64对候选，2原像素阈值，至少6点、前后各3点、70%拟合覆盖、1度视差；不按语义选择。未获支持的轨迹在全查询统计中保留原基线预测，并显式报告缺失覆盖，不能仅报告容易子集。独立旧集合不参与新集合相机优化。'+
'\n联合束调整：新集合逐重复轨迹簇选最长轨迹，不按误差或语义选择；13,951条候选中73条因初始投影不合法排除，13,878条轨迹、295,841个拟合观测进入求解。两组同样使用2像素soft-L1，SIMPLE_RADIAL内参固定。联合组固定首帧位姿和最远相机一个平移分量以固定规范自由度。两组原生100次迭代上限，均No convergence；返回usable不等于收敛。没有追加预算或改写源相机。'+
'\n初始化代码核查：initialize_static_clothing_cloud.py:18-43仅利用人体范围/尺度、轮廓交集和衣物稀疏点。公开GaussianAvatars的scene/flame_gaussian_model.py:38-41,137-147绑定网格面片；scene/gaussian_model.py:114-158将局部位置/尺度/旋转变为世界参数；train.py:134-146控制过大的位移和尺度。当前只参考机制，尚未采用绑定训练，不能把本次源码核查当作有效实验。官方项目https://github.com/ShenhanQian/GaussianAvatars；用户指定公开分支https://github.com/jmaraval/GaussianAvatars_SMPLX。'+
'\n证据：'+run+'/decision.json ; result.json ; joint-bundle/result.json ; source-route-review.json。图仍为077与079的已完成渲染。当前079模型质量未通过，不部署00。';
const candidatePath=build+'/consensus081-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_初始化结构核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/consensus081-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/consensus081-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/consensus081-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed:[52],canonical_replaced:false}));
