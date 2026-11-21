import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='camera-entry123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/pattern-consumer123-deck-receipt.json'),review=await read(root+'/../../local-clothed-surface-123/camera-observation-entry/result-completed.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(review.total_checks!==26||review.geometric_solver_executed!==false||review.quality_accepted!==false)throw Error('Entry regression evidence missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('审阅结果尚未接入相机入口','错误对应已在相机入口被拒绝');
replace('旧相机脚本会重新提特征','原始像素和FIT／HELD保留，身份未知的点继续保留为未知。');
replace('严格准入已阻止新拟合','26项工程检查通过。尚无实拍相机或人体对齐通过证据。');
const figure=root+'/../../local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/row_009_native.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'同一轨迹的原帧335、375、415：查询分别在脸部、椅架和门把手附近。完整三角闭环仍连接不同物体，图片未加文字或扭曲。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["本轮修复实际local_surface123.prepare调用路径。新增冻结观测读取器和单独的reviewed相机入口，先验证源文件及角色证据再创建运行根。旧相机文件保持原字节，新几何部分与旧seed_pairs之后的计算完全一致。没有重新检测、匹配、优化相机、拟合SMPL-X或训练高斯。", "真实video4数据共99原生帧、42609个原始段、95886个观测。读取时保留原始slot、源帧、图像SHA、特征SHA、keypoint编号、native_xy、原始FIT/HELD与development标志，不从ROI推断物理身份。", "三条已确认物理矛盾的FIT轨迹族关联10段和35个观测。新入口实际记录family_physical_identity_conflict并在求解前拒绝。原8527条HELD段全部留出。未知角色没有改为静态背景，当前可准入静态背景FIT数仍为0。", "初始18项测试覆盖真实错误族/原始像素划分、实际相机入口拒绝、真实prepare缺省/真实输入拒绝、合成prepare到新相机接口的传输、源帧/像素/划分/文件篡改等。合成正例使用几何求解器spy，实际几何求解调用数为0，不构成物理标定或训练验收。", "代码复核随后补齐证据范围检查，另8项合成检查通过。支持文件必须绑定同一视频、轨迹族、分段、全部原始节点、源帧、特征与图像字节、原生像素，并有可见性及测量不确定性声明。检查通过只说明证据引用范围一致，不能由解析器证明声明的物理真伪。", "图沿用已经实际审阅的335、375、415源帧裁片。完整三角闭环也会连接脸、椅架与门把手。图片无图内文字或扭曲。本轮是消费接口修复，不把这条候选错误当作旧SMPL-X全部错位的唯一原因。", "田地人物首次177帧区域核验已完成，耗时87秒，每帧均找到所跟踪人物，零越界仅证明坐标在画面内。实际查看五个固定原生人物、头和脚裁片。第440帧只有后脑勺，检测器仍输出68个面部点，原始响应中位数3.62并非概率。侧脸的远侧五官、背面裤腿下的鞋后跟也不可直接作可见观测。这次只定位区域，尚未核验物理材料对应、求相机或拟合。", "当前各人物仍未通过独立面部逐点严格小于1像素及全身保护。精确输出K、外参、物理对应与人物微动仍需分开诊断。此次没有发布新点云或改变历史PLY/camera，没有00部署或WPS云同步验证。", "证据：local-clothed-surface-123/camera-observation-entry/result.json、result-completed.json、real-source-audit.json、actual-camera-denial.json、code-manifest.json、implementation.json，及field-preflight/coverage-plan.json、coverage/result.json和coverage/visual-review.json。原PPT只改第52页，前51页保留。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123冻结观测入口_含田地审阅.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_camera_entry123.py',preserve);
execFileSync(py,[b+'/preserve_deck_camera_entry123.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:prior.final_sha256},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-field-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:prior.final_sha256,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
