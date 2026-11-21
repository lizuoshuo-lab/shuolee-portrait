import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-semantic123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-temporal-bridge';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-temporal123-deck-receipt.json'),review=await read(root+'/visual-review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_visual_review||review.reviewed_rows!==96)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('人体框内的静止背景','人体框误排背景：像素遮罩对照');
replace('视频10：33帧','33帧遮罩已审阅，恢复216组背景候选。');
replace('8个追加案例','1帧遮罩失败；相机与SMPL-X尚未重新拟合。');
replace('门框同一局部','门框案例（第242／268／295帧）。候选增加不等于物理对齐通过。');
const figure=root+'/bbox-background-evidence.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'第242、268和295帧原生384像素裁片。相同门框局部被包含在头部或身体矩形中，矩形区域不能代表人物像素。图中没有新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=['2026-10-29 像素语义对照完成，物理材料复核待执行。','33个SAM2.1tiny单次CPU遮罩用时118.42秒，全部11页实际审阅。32帧仅准作粗人物排除；617帧出现明显棋盘碎裂，完整拒绝且未重试。新语义不用于训练alpha。','在原点坐标、描述子、99对匹配、1732轨迹、1191族和原FIT/HELD划分全部不变时，背景候选族由464变636：新增216、损失44。全部3个混组隔离和6个已知物理矛盾保持不变。第349帧的候选从2增到29，此为语义资格，不是物理匹配准确率。','首轮标签保存的NumPy整数JSON序列化失败已保留日志，仅修复int转换并重算标签；未重复分割或提取/匹配。','原生材料审阅已预登记113族：96个恢复族（64FIT/32HELD）按时间覆盖选取，17个原始背景族用于失败遮罩617的独立控制。此113族裁片已生成但尚未审阅，不作为相机输入。','没有新K、外参、SMPL-X拟合或高斯训练。眼鼻嘴严格逐点小于1原生像素验收仍未通过。','此前时间桥接证据及边界：',review.summary,
 '在冻结原9帧特征、36对匹配和FIT/HELD划分的基础上补24个原生中间帧。33帧共99对匹配、123个闭合三元组、1732轨迹、1191族。3个连接FIT/HELD的族保持隔离，旧6个物理矛盾记录未移除。实际运行CPU93.24秒，无GPU、内参、外参或几何修改。',
 '96个代表族及8组扩大原生上下文已审阅。背景粗材料候选18FIT/9HELD、非地面14FIT/7HELD，但最大帧对共享FIT仅5，低于12；12个帧无本次已审阅固定FIT，并不证明整段无对应。原采样不是全族无偏错误率。',
 '针对第349帧38条有背景观测轨迹中36条所在族混合ROI角色而排除的现象，预登记8个未审阅、确定性最长混合族。追加8组原生上下文均落在背景（水烟器、门框、地砖），并非人体。先前人体或头部外接框将背景误分，导致可用约束丢失。部分管面反光仍不保证固定物理点；未因此直接晋升相机输入。',
 '本页图仅使用门框案例前3个原生384像素裁片，不变形、不加字。该误分类确证发生在本轮诊断中，尚不能唯一解释旧灰衣SMPL-X全部错位。相机物理标定、材料身份和人物微动依旧分开。',
 '16个头部ROI代表中10个可辨认头面粗局部为5FIT/5HELD，包含耳部，不等于独立眼鼻嘴亚像素对应。认证面部亚像素HELD仍0，目标尚未通过。',
 '下一步一次33帧像素人物遮罩诊断已登记，复用现有SAM2.1 tiny，不下载权重。仅改区域语义，保留点坐标、描述子、匹配、整族划分、排除记录和K；先审阅遮罩再决定标签可用性，不自动放宽阈值。',
 '证据：'+root+'/visual-review.json；'+root+'/mixed-region-cases/visual-review.json',
 '此前结论继续保留：',notes.text].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10语义遮罩核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_video10_semantic123.py',preserve);
execFileSync(py,[b+'/preserve_deck_video10_semantic123.py'],{stdio:'inherit'});
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
