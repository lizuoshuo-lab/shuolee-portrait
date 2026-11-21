import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-material123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-static-material';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/grayT123-final-control-deck-receipt.json'),review=await read(root+'/visual-review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_visual_review||review.reviewed_rows!==36)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('逐帧改善了脸部，但统一姿态仍未通过','三视角循环匹配仍会连接不同物体');
replace('95个训练视角','视频10：审阅36族对应，6族出现明确物理错接。');
replace('眼眉未参与拟合','7条头部框轨迹中6条在衣领，面部独立留出仍为0。');
replace('第57帧','第0／107／644帧：鞋架被误配到冰箱便签。9帧诊断尚未通过相机准入。');
const figure=root+'/native-context/row_005_family_48.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'第0、107和644帧原生384像素裁片。鞋架和冰箱便签来自不同物体，三视角循环仍然通过。图内没有新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=['2026-10-29 视频10新的原生材料诊断。',review.summary,
 '9个预登记原生帧相隔约107/108帧，约4.5秒。原生SIFT/RootSIFT双向比率匹配形成105个三视角循环、87轨迹、69近重复族。36个预登记代表族及16个384原生上下文实际审阅。6/36不是对全部族的无偏错误率。',
 '明确错接包括鞋架到冰箱、飞镖盘不同扇区、地砖接缝混配、前门襟到后衣领及下摆、眼镜到手腕。循环一致性不是物理身份保证。',
 '21条背景审阅样本中仅3FIT/1HELD可辨认粗材料斑块，非地面2FIT/1HELD，均未认证亚像素位置。预登记相机准入至少20FIT/8HELD，非地面至少6FIT/3HELD，初始帧对12共享FIT，当前最多3。未求新相机。',
 '7条headROI中6条位于衣领，1条嘴角下方胡须斑块仅FIT。没有眼鼻嘴独立HELD；此处不能用头部框计数表示真实面部覆盖。',
 '该结果不能证明全860帧无法重建，也不能将旧灰衣SMPL-X错位唯一归因于新视频10的错误对应。K、外参、静态模型表达、真实人物微动继续分开。',
 '下一步登记缩短匹配间隔的时间桥接诊断，原9帧特征原字节和冻结划分保留，跨FIT/HELD族的桥接需隔离，不转移留出。无新内参、外参、几何或高斯训练。',
 '全部有效独立面部点小于1原生像素的验收尚未达到。本轮GPU任务0。',
 '证据：'+root+'/visual-review.json','图源：'+figure,
 '此前灰衣对照结论保留在报告和以下历史说明：',notes.text].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10材料核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_grayT123_final_control.py','utf8')).replaceAll('grayT123-final-control',tag);
await fs.writeFile(b+'/preserve_deck_video10_material123.py',preserve);
execFileSync(py,[b+'/preserve_deck_video10_material123.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:prior.final_sha256},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:prior.final_sha256,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
