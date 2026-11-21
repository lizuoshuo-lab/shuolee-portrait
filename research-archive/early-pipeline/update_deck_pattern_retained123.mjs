import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='pattern-retained123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/pattern-material123-deck-receipt.json'),review=await read(root+'/../../local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_pixels_reviewed||review.confirmed_contradictions_FIT!==3)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('跨视角匹配把衣服连到了眼睛','闭环匹配仍可能连接不同物体');
replace('原生纹理双向匹配','脸部、椅架、门把手形成完整匹配闭环，物理身份仍错误。');
replace('桥接误连已隔离','发现3条错误FIT段；当前人物尚未进入相机、SMPL-X或高斯拟合。');
replace('第0、120、125帧','第335、375、415帧，同一轨迹的原生192px裁片。查询在中心附近。');
const figure=root+'/../../local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/row_009_native.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'同一轨迹的原帧335、375、415：查询分别在脸部、椅架和门把手附近。完整三角闭环仍连接不同物体，图片未加文字或扭曲。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["本页为video4花纹上衣人物的跨视角材料核验。视频原生2704x1520，491帧中预登记stride5选99帧。没有二次去畸变，也没有重估K。", "先前仅隔离桥接连接，不能保证剩余闭环是正确材料。此次按冻结SHA顺序审阅193个头部家族中的160个，共172个段；另外40个背景家族。余33个头部家族显式未审，不替换或补样本。", "实际审阅212组三视角原生192像素裁片及38个ROI跨界端点，共674个裁片。长轨迹未查看全部节点。头部138个FIT段和34个HELD段；全部样本FIT158行、HELD54行。没有移动原来的FIT/HELD划分。", "本页是row9，源帧335、375、415，三个中心查询分别位于脸部、椅架金属连接处和玻璃门把手附近。三条互相匹配的边构成完整三角闭环，双向描述子比值均低于冻结的0.75。没有使用位移或重投影误差筛点。", "另一个明确反例row12源帧470、475、480，同样为完整三角闭环，前两视角为侧脸区域，第三视角为墙面/门框。row40是20节点段，存在衣服到嘴部的身份矛盾，56条边及33个三角形都不能保证物理正确。实际只审阅该长段登记的端点和区域跨界点。", "3条明确矛盾均属于FIT，但当前pattern尚无相机、SMPL-X或高斯训练。因此它们证明前端候选物理身份不能靠闭环认证，并不证明历史最终相机必然使用了这些错点。正常RANSAC/BA可能排除候选外点；不能宣称已定位旧模型的唯一根因。", "其余相似脸部、头发/轮廓、重复百叶窗和地毯、玻璃反射与遮挡仍保留未知。物体级外观一致不能证明亚像素精确材料身份或静止性。当前没有经认证的独立面部材料留出及定位不确定性，不能宣称小于1px通过。", "旧灰衣SMPL-X已记录身体中位23.85px、面部参与点中位43.99px检测器残差。旧粗门槛只管身体且约50–53px，缺少面部与脚部独立验收。投影/索引核查差<0.0004px只排除已查路径数值整体错配，不证明物理K/E正确。", "本轮预算结束，停止该人物相机准入，不重复提取、补样本或调阈值。后续只适配已有观测证据拒绝明确错误族，并检查消费端实际执行；原始像素/家族/划分不变。历史模型保留，无新增点云或质量通过声明。", "证据：local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/review.json、contradiction-trace.json、decision.json、consumer-evidence-plan.json。右侧查看器仅诊断更新，没有00部署或WPS云往返验证。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123闭环物理核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_pattern_retained123.py',preserve);
execFileSync(py,[b+'/preserve_deck_pattern_retained123.py'],{stdio:'inherit'});
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
