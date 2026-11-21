import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-matcher123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-correspondence-contrast';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-support123-deck-receipt.json'),review=await read(root+'/visual-review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.reviewed_all72||review.rows.length!==72)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('视频10的材料对应','遮挡误配也能通过往返检查');
replace('113族已审阅','冰箱背景点转到人物，往返误差仅0.08px。');
replace('连接计数最多','固定置信度门槛排除此例，物理对齐仍未通过。');
replace('空调局部','第644／698帧。左原点，中RootSIFT，右RoMa。裁片中心为查询位置。');
const figure=root+'/matcher-evidence.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'同一原生查询点的384像素裁片。左原冰箱背景，中冻结RootSIFT对应，右RoMa对应。目标被人物遮挡，不能将小往返误差当真实物理点对齐。图内没有新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29 视频10同一原生查询的匹配方法对照。", "只改变对应算子。原图、SIFT查询位置、FIT/HELD族划分、K、相机、人体均保持不变。固定两个三帧片段484/510/537和644/671/698，以及806/859粗背景控制，共7对。复用本机官方RoMa Outdoor和DINOv2权重，不下载新模型。", "7对仅一次推理，70.48秒，峰值显存5829209600字节。保存所有原始预测与置信度，双向和三视角循环检查只度量自洽。", "72个预登记位置、12页原生裁片和16组扩大上下文已全部实际审阅。筛查固定为双向置信度>=0.9，往返<1原生px。15个通过，其中8个可辨认粗背景、1个粗人物、6个仍有反射或重复纹理歧义。12个原有粗背景控制仅4个通过这一门槛。未调整阈值。", "图为第42行。644帧冰箱篮子/磁贴附近的背景在698帧被人遮住，目标落到脸颊或衣领附近。RoMa往返约0.08px，但置信度0.8375低于0.9，因此本轮固定门槛拒绝。这不证明通过该门槛的点均为正确物理身份。", "头部矩形仍不能代替真实面部身份。其中一些查询实际落到柜门照片、飞镖盘或轮廓旁背景，不能用于人脸亚像素验收。后颈/衣领的粗匹配也不能当眼鼻嘴真值。", "第58行眼镜/眼周粗局部为FIT，已实际审阅671/644/698三帧和返回671。两种方向循环均通过，闭合误差0.174177和0.074986原生px。这是有效的粗对应线索，不是独立物理重投影误差；镜片反光和精确表面点仍未认证。独立有效面部HELD为0。", "全量预测中出现到相反划分旧族的邻近关联，均单独标记，未把HELD转为FIT，也未改旧族。这只是近邻隔离，不认证真实材料身份。", "这一7对探查不能代表整视频相机已注册，也不能唯一解释旧灰衣SMPL-X重影。没有新相机、SMPL-X拟合或高斯训练。", "新盘点发现636个原语义候选族中，已审阅代表208族；18个支持薄弱视角还涉及222个尚未完成全观测审阅的候选族（187FIT/35HELD）。下一步只审阅既有缓存，不重跑特征、RoMa或扫阈值。此前5/33是已审阅粗支持的计数范围，不能推广为所有原候选都不可用。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\video10-correspondence-contrast\\visual-review.json；C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\video10-correspondence-contrast\\decision.json；C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\video10-correspondence-contrast\\run\\result.json"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10匹配对照.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_video10_matcher123.py',preserve);
execFileSync(py,[b+'/preserve_deck_video10_matcher123.py'],{stdio:'inherit'});
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
