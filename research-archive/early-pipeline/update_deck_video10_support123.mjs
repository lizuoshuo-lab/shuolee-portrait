import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-support123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-semantic-role';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-semantic123-deck-receipt.json'),review=await read(root+'/material-visual-review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_visual_review||review.reviewed_rows!==113)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('人体框误排背景','视频10的材料对应与相机连接');
replace('33帧遮罩已审阅','113族已审阅，合并粗背景60FIT／27HELD。');
replace('1帧遮罩失败','连接计数最多覆盖5／33视角，尚未重算相机。');
replace('门框案例','空调局部（第27／54／80帧）。粗对应可辨认，亚像素尚未验证。');
const figure=root+'/material-support-evidence.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'第27、54和80帧同一门上空调的原生384像素裁片，图内没有新增文字。粗局部可辨认不代表亚像素定位。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29 视频10材料对应与相机连接。", "113个冻结族的全部原生观察和16组扩大上下文已实际审阅。96个恢复族含64FIT/32HELD，17个原背景族为失败遮罩617的独立控制，原617遮罩仍被拒绝。原坐标、描述子、99对匹配、整族划分、K均保持不变。", "本轮可辨认粗背景43FIT/18HELD。与先前27个粗背景族按原族去重后共60FIT/27HELD，其中非地面刚性候选45FIT/21HELD。数量是可辨认粗局部，不是认证独立物理点、逐点亚像素真值或无偏匹配准确率。", "反光锅面、重复鞋纹/地砖/飞镖盘等仍保留歧义。数字显示屏内容在20:14与20:15间改变，数字不能当作固定材料。不能因为同一物体被看见，就断言检测中心对应同一表面点。", "最大共享FIT为806/859帧的12个粗族。所有达到原12点初始化条件的起始对均已做只读计数扩展，后续要求8个当前潜在三维局部支持。乐观闭包仅752/779/806/832/859共5视角，另有18个视角单帧不足8个已审阅FIT。该运算没有估计F/E、相机、三维点或人体。成功三角化可能比计数更严格。", "已经确认人体外接框误排部分静止背景，但修复区域语义尚未建立整圈相机约束。不能把本轮新视频证据唯一归因于旧灰衣SMPL-X的全部错位。旧模型的相机物理误差、材料歧义、人物微动、模板表达及训练形变/静态导出仍分别保留。", "下一步对连接断裂只登记一次相同原生查询位置的RootSIFT/RoMa外观对应对照，复用现有模型，不再重跑SIFT、不下载权重，不以置信度或双向一致性当物理真值。新相机和从头训练仍需完整准入。", "图像来源：video10-semantic-role/material-review/contexts/row_003_family_244.png的前3个原生384像素裁片。没有新增文字、缩放或重采样。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\video10-semantic-role\\material-visual-review.json；C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\video10-semantic-role\\material-support.json；C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\robust-portrait-122\\video10-semantic-role\\graph-audit-result.json"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10材料连接核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_video10_support123.py',preserve);
execFileSync(py,[b+'/preserve_deck_video10_support123.py'],{stdio:'inherit'});
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
