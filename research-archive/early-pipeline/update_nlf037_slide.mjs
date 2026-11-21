import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-nlf-surface-prior-fresh-037';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='179a2e0b269ac82a759fd7bd2e19103f471e08643af282166d6a411c9f5f6b4b';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const hash=async path=>digest(await fs.readFile(path));
if(await hash(source)!==expected)throw Error('Canonical deck changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='nlf_surface_prior_reviewed_no_overall_geometry_gain')throw Error('Actual review required');
const previous=JSON.parse(await fs.readFile(build+'/nlf036-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const sourcePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(sourcePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{
 const item=row.elements.find(x=>x.text===before);if(!item)throw Error('Missing text '+before);
 p.resolve(item.id).text.replace(before,after);
};
replace('身体先验与共享体形','身体表面先验与姿态');
replace('NLF预测三维人体点，为共享体形与逐帧姿态提供起点。','对应表面点与不确定性，约束共享体形和逐帧姿态。');
replace('左组：实拍、原先验拟合、NLF拟合。右组：相同侧面相机下的两种身体。',
 '左组：实拍、关节先验、关节与表面先验。右组：相同侧面相机下的两种身体。');
replace('二维关节残差由 8.71 降至 8.06 像素，整体姿态仍未通过。\n下一步保留表面点与不确定性约束，再检验身体几何。',
 '与先验的表面距离由 31.75 降至 23.66 毫米，不是人体重建精度。\n预测一致性改善，整体几何仍未通过。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two existing result figures');
for(const [i,file] of ['source-000045.jpg','fixed-left90.jpg'].entries()){
 const im=p.resolve(images[i].id);
 im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',
  alt:i?'固定23帧同一世界侧面；左关节先验，右加入表面先验':'45帧实拍、关节先验从头对照、加入完整表面先验从头拟合'});
 im.crop={left:0,top:0,right:0,bottom:0};
}
let notes=row.notes.text;
const nr=(before,after)=>{if(!notes.includes(before))throw Error('Missing note anchor '+before);notes=notes.replace(before,after);};
nr('本页比较既有GVHMR图像/视频先验与NLF独立图像先验提供的身体起点，不能将下载或参数转换成功称为几何正确。',
 '本页在相同NLF初始预测下，隔离比较局部关节先验与增加完整表面点先验的作用。不能将下载或参数转换成功称为几何正确。');
nr('本轮的036后续身体优化并没有直接使用这些表面点或权重，不能称作完整官方目标的复现。',
 '037把这1024个原始非参数化表面点直接用于后续优化，使用官方相对不确定性权重，但局部目标、优化器与阶段为本工程的受控扩展，不是完整官方拟合器复现。');
nr('所有优化变量与Adam状态新建，原023仅在评测时只读。',
 '两组所有优化变量与Adam状态分别新建。037对照在训练完成后与036逐数组比较，全部相同；036没有被读取为初始化。');
nr('本轮1200次身体更新、0次高斯更新。',
 '本轮两组各1200次身体更新，总2400次，0次高斯更新。');
const a=notes.indexOf('六、实际对照结果'),b=notes.indexOf('八、方法来源与证据');
if(a<0||b<a)throw Error('Missing note sections');
notes=notes.slice(0,a)+`六、完整表面点约束
网络原始点为Pᶜₜᵢ，以相机毫米输出。先除以1000，再减去官方预测骨盆，最后逆转官方根朝向，得到局部目标pₜᵢ。SMPL-X顶点同样减去本帧骨盆，得到vₜᵢ。局部点项不直接改变整体世界根朝向和平移；它约束体形与关节，场景配准仍由原相机和真实图像项约束。
L表面=(3TN)⁻¹ΣₜΣᵢwₜᵢΣₖρ((vₜᵢₖ−pₜᵢₖ)/τ)，其中T=59、N=1024、k为三个坐标分量、τ=0.06模型米、ρ(r)=√(1+r²)−1。候选组附加r表面L表面，r表面在300—500步从0线性升到1；对照组附加权重始终为0。其他损失、手部、可见性、相机、参数范围和步数完全相同。
1024点通过模板索引精确对应，使用全部点，没有做面部或手部专项。局部坐标往返与原参数化网格重放通过。表面项对体形、共享姿态、逐帧姿态的梯度均非零且有限；最大体形分量解析梯度−0.00726651，有限差分−0.00726804，预检后参数精确恢复，未产生优化更新。

七、实际作用、未解决问题与下一步
两组从相同原始预测独立初始化，初始化逐数组一致；从头对照的全部最终数组也与036一致，排除了旧断点续训和其他目标改动。与原始NLF预测的表面距离中位数由31.746降至23.656模型毫米，95%分位由78.076降至58.283毫米。这是与网络先验的一致性，不是人体扫描误差，也不是毫米级重建精度。
分部位中，躯干表面距离21.022变为19.921毫米，腿/脚37.565变为23.987毫米，头颈33.325变为17.384毫米。整体降幅不能代表每个部位都同样改善。体形10个系数中，偏离起点超过0.85的数量由7变为6，仍不能证明体形准确。
可见二维关节残差中位数8.059变为8.272像素，95%分位30.067变为31.279像素。未优化鞋/裤脚纹理诊断1.447变为1.444像素，变化很小。固定23帧模型左右膝弯曲10.72/4.97度变为11.50/6.19度；模型角度不是实际关节真值。
已检查全59帧联系表、六张原图叠加、固定时刻的正侧面和上下25度。胸腹形态、手臂位置及侧面体态没有明显整体突破。裸人体不必填满宽松衣物；没有独立人体表面或关节真值，当前不能据此计算真实准确率。候选不作为新高斯训练底模，不替换查看器，不部署00。
表面项确实约束了预测偏移，却仍受原始单目预测偏差限制。下一步核对NLF原始点与根朝向在已标定世界坐标中的跨视角一致性，区分预测随角度变化、真实动作和后续拟合改变。继续使用VPoser、GVHMR、NLF已有先验，SAM3DBody暂缓。停止只扫描表面项权重或增加步数；几何通过后才从20万新点和新优化器训练18000步。

`+notes.slice(b);
const tail=notes.indexOf('本页图像来自实际原图');
if(tail<0)throw Error('Missing figure provenance');
notes=notes.slice(0,tail)+`本页图像来自实际原图与本地网格渲染。三列叠图为原始45帧，固定侧面图为时刻23。无生成图像或人为身体转正。协议与证据见${phase}/control/protocol.json、surface/protocol.json、surface/surface-prior-gradient.json、review/comparison.json、phase-conclusion.json。失败模型在结论核验后清理，保护原始素材、官方预测、源码、许可模板和必要对照。`;
slide.speakerNotes.text=notes;
const after=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']});
await fs.writeFile(build+'/nlf037-edited-layout.json',JSON.stringify(after.records,null,2));
const draft=build+'/nlf037-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_NLF表面先验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nlf037-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/nlf037-final-renders';await fs.mkdir(renderdir,{recursive:true});
const checks=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
 const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
 if(i<54&&!unchanged)throw Error('Unintended visible slide change '+(i+1));
 checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/nlf037-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,
 renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slide:55,unchanged_first54:true,review_required:renderdir+'/55.png'}));
