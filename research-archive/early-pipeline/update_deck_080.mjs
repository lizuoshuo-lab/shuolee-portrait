import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/camera-local-inconsistency-audit-080';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='31d17c0752d039938c99b6a41db440aa656d1ec59f3b35088e03b74aa975e517';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json');if(!d.actually_inspected||d.gaussian_optimizer_updates!==0)throw Error('Diagnostic evidence required');
const previous=await read(build+'/surface079-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[51]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records[0];
const edit=(k,value)=>p.resolve(snap.elements[k].id).text.replace(snap.elements[k].text,value);
edit(2,d.ppt_summary);edit(4,d.ppt_caption);edit(5,d.ppt_limit);
p.resolve(snap.id).speakerNotes.text=snap.notes.text+'\n\n080后续诊断（2026-10-24）：\n'+d.summary+'\n'+d.matched_result+'\n'+d.local_result+'\n'+d.cross_bank+'\n'+d.direct_result+'\n'+d.visual_conclusion+'\n'+d.next_step+
'\n观测方法：全部175原始整图以64个半分辨率像素为网格提取角点，每格至多12点；双向相邻KLT和图像块NCC筛选，不使用人物/背景语义遮挡。得到15428轨迹，按三次共同帧内1.5原像素邻近关系合并为13951轨迹簇。每轨迹前后三分之一独立三角化，中间三分之一留出；三个轨迹簇折的相机拟合与查询簇不相交。全局微调采用6自由度、2像素soft-L1稳健代价；行变化诊断增加随预测行线性变化的6自由度，预测行通过6次不动点迭代得到，未把查询纵坐标输入模型。两者均没有碰到约束边界。\n鼻口主集合覆盖76帧，108帧数字是有头部及同帧背景支持的总帧数。第100帧由另一个既有鼻口观测集单独验证。所有数字均为原始图像像素，点/帧可能相关；不是未见实拍视角或独立相机真值。\n源码参考：PixSfM https://github.com/cvg/pixel-perfect-sfm 提供深特征关键点和束调整，WildGaussians https://github.com/jkulhanek/wild-gaussians 提供遮挡与外观变化处理；本轮仅核对官方资料，没有安装训练或本素材收益结论。证据：'+run+'/result.json ; readout-result.json ; confirmation.json ; decision.json。\n图仍为079的已渲染结果，080没有修改相机文件或高斯。';
const candidatePath=build+'/camera080-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_整图观测核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/camera080-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/camera080-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/camera080-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed:[52],canonical_replaced:false}));
