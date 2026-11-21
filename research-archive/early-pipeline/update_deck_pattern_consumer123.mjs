import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='pattern-consumer123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/pattern-retained123-deck-receipt.json'),review=await read(root+'/../../local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/consumer-audit/result.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(review.state!=='stopped_at_registered_consumer_gap'||review.direct_consumer_integration_completed!==false||review.family_conflicts!==3)throw Error('Consumer audit missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('闭环匹配仍可能连接不同物体','审阅结果尚未接入相机入口');
replace('脸部、椅架','旧相机脚本会重新提特征，不读取已经审阅的轨迹。');
replace('发现3条错误FIT段','严格准入已阻止新拟合；错误对应尚未在消费端完成修复。');
const figure=root+'/../../local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/row_009_native.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'同一轨迹的原帧335、375、415：查询分别在脸部、椅架和门把手附近。完整三角闭环仍连接不同物体，图片未加文字或扭曲。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["输入仍是video4花纹上衣人物。上轮实际审阅160头部家族和40背景家族，确认3个FIT段物理身份矛盾。图为源帧335、375、415的192px原生裁片，完整三角闭环仍把脸、椅架、门把手连接。", "本轮按既有consumer-evidence-plan只读检查真实消费接口，不新增检测、匹配、相机、人体先验、SMPL-X或高斯训练。原始审阅、轨迹和特征清单SHA保持不变。", "3个矛盾家族在原始分段表中关联10段、35个观测，逐条保留原source frame、slot、keypoint、图像SHA、特征SHA和浮点native_xy。家族内存在矛盾不等于每个观测都是错点，故仅记录family conflict，点级身份仍保留unknown。", "实际material_observations122接口只允许fit_background、heldout_background和heldout_human。新的原始划分为FIT/HELD，混有人物与场景候选；ROI矩形不是物理语义。对三条真实反例保留原始FIT后调用接口，均因Frozen split/family required而拒绝。这是格式/语义不匹配拒绝，不是接口已经读懂物理矛盾。", "实际local_surface123_camera.py中的camera函数会重新执行SIFT检测和双向匹配，不读取本次已保存segments或physical-review，也不调用N3物理证据准入。因此不能宣称桥接隔离或错误家族已经进入生产相机求解。该旧脚本本轮未运行。", "实际检查pattern运行根仍不存在；调用现有require_camera_admission在CPU上明确拒绝camera_alignment_receipt_missing。没有创建运行根、导入torch或调用SMPL-X。这一保护已存在，本轮没有将它冒充新修复。", "依照预登记停止条件，停止这次接口适配，保存audit-only family-evidence.json和result.json，不改FIT/HELD、不把未知人物点贴成静态背景、不新增训练输入。完成的是接口缺口定位，不是对齐改善或pipeline验收。", "原有123七人物队列还包含未启动的田地深色上衣field，对应内容SHA a028323a751e69ab3236bb3879b636159780906cf34acf19dc2edc2bebfb30a8，现名video7.mp4。继续按已授权队列做一次原生输入核验，固定177帧及5个展示时刻；没有调用相机或人体训练。177帧原生提取完成，实际查看了0、220、440、660、880五帧全图及各自头部与脚部原生裁片。抽检含正面、两侧与背面，脚部入镜但部分鞋被裤脚/远侧腿遮挡。建筑和灌溉接头尚未建立物理对应，树叶、重复立杆和平行管线保留歧义。包和衣物应由自由表面表达，不能把裸SMPL-X轮廓当作穿衣外表面。未测量人物微动或亚像素误差。", "当前强鲁棒性目标仍未完成。精确输出K/残余相机误差、物理身份、人物微动仍需分开；独立有效面部留出逐点严格小于1像素尚无通过证据。右侧仅更新诊断，历史点云与camera保持，未验证00部署或WPS云往返。", "证据：local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/consumer-audit/result.json、family-evidence.json、execution.json；上一轮review.json与contradiction-trace.json。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123消费接口核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_pattern_consumer123.py',preserve);
execFileSync(py,[b+'/preserve_deck_pattern_consumer123.py'],{stdio:'inherit'});
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
