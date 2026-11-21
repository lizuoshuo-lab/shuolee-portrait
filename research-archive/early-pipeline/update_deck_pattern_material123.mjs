import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='pattern-material123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/pattern-coverage123-deck-receipt.json'),review=await read(root+'/../../local-clothed-surface-123/pattern-preflight/native-material/visual-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_pixels_reviewed||!review.actual_all10_false_track_pixels_reviewed)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('脚部预测落在桌面和上衣','跨视角匹配把衣服连到了眼睛');
replace('99帧核验','原生纹理双向匹配把衣服与眼部合并为同一轨迹。');
replace('人体mask内','桥接误连已隔离，剩余闭环尚未通过物理对应与亚像素验收。');
replace('第120、245、370帧','第0、120、125帧，同一轨迹的原生192px裁片。查询在中心附近。');
const figure=root+'/../../local-clothed-surface-123/pattern-preflight/native-material/visual-review/row_07_native.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'同一跨视角轨迹的第0、120、125帧原生192px裁片。第一个点在衣服图案，后两个点在眼部。图片未加文字或扭曲。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29。只更新第52页，前51页保留。video4.mp4花纹上衣人物，99个原生2704x1520帧，stride5，从491帧中预先选定。输入已去畸变，不二次去畸变。", "一次SIFT原生区域提取和464对双向L2 ratio0.75匹配，共396237候选与359252匹配边。1.5px去重、3px近重复族划分均在几何之前冻结。15684条未隔离轨迹共95886个观测。", "实际审阅预先选定的30组三视角原生裁片，以及错误轨迹全部10个原生裁片。第7组先在衣服图案，随后在眼部，确认不是同一物理点。不能把其他29组的视觉相似当成物理认证。", "错误桥接为原帧5/key421到85/key9。前向描述子比值0.7238906，反向0.7474796，均通过0.75阈值。二维距离614.466px不是重投影误差，筛选没有使用这个距离。该边没有三角闭环且为桥；整条轨迹其他部分有5个三角闭环，因此仅查整条轨迹有闭环会漏掉这一错误。", "本页显示同一轨迹的原帧0、120、125，三个192x192原生裁片，无图内文字或几何扭曲。查询位于各裁片中心附近。实际错误边5到85的完整逐边证据保存在false-track-trace-reviewed.json。", "同一固定观测做一次拓扑控制：隔离26925条桥接边，保留9661个至少三视角的段、62938个观测。32948个短段观测保留为待核验。桥接边并非全部错误，保留闭环仍可能是重复图案的错误对应。未重新提取或匹配、未修改K或二维位置。", "FIT保留7699段，HELD1962段，其中1946段尚未被当前人工查看。所有原始节点、边和近重复家族及FIT/HELD不变。自写Tarjan和独立NetworkX在全部15684个处理图的桥与连通分段上完全一致。合成控制包含两环经桥相接、完整三角形、链及输入顺序不变。实现正确不等于物理正确。", "固定30组查询中14组仍共同落在有闭环的段内，16组失去完整支持。这是覆盖变化，不是准确率。例7被拆为3个单点及一个7视角眼部段，衣服和眼睛不再合并。", "明确反例是HELD，从未进入新相机或SMPL-X。它说明身份合并弱点，不证明旧拟合唯一根因。30个已审阅家族全部作为开发证据，保持原FIT/HELD，不用其验收新规则的独立精度。排除这些家族后原帧490未查看的HELD头部候选为0，缺口保留。", "上一轮确认脚点误标到桌面、上衣和袖子。关节语义、跨视角物理身份、相机和人物微动应分别核验。当前pattern没有相机、SMPL-X或高斯训练。独立物理可信面部留出0，不声称1px达标。", "下一步在固定段上检查193个头部家族的物理身份，单次预算最多160，剩余显式未审；背景最多40家族。没有新特征/权重/匹配/相机。材料支持不足停止该人物候选，不自动放宽门槛或补点。", "证据：local-clothed-surface-123/pattern-preflight/native-material/decision.json、visual-review/review.json、false-track-trace-reviewed.json、bridge-control/result.json、bridge-control/physical-review-plan.json。右侧仅更新诊断，历史PLY不变，未验证00部署或WPS云往返。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123跨视角身份核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_pattern_material123.py',preserve);
execFileSync(py,[b+'/preserve_deck_pattern_material123.py'],{stdio:'inherit'});
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
