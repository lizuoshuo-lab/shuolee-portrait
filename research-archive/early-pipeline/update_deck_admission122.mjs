import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027';
const figure=evidence+'/native-temporal-material/counterexamples_anchor_and_target.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/visibility122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/admission122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('遮挡引起的','材料对应与输出内参核验');
replace('双向误差小于','双向误差小，仍可能跟错材料；已单独记录6个遮挡反例。');
replace('RoMa拒绝6个','输出K与裁剪取整未核实，尚未进入人物训练。');
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-27 N3/N4：新版1520x2704原生视频的材料证据与输出内参合同。原六列失败裁片继续保留，图片没有任何新文字或几何变形。',
 'N3已完成一次38项CPU合同检查。557帧×307种子=170999槽位，148571数值有效观察全部保留。6项物理反例独立标注；不知道身份和不确定性的点不能由FB或RoMa置信度自动升级为相机拟合/面部独立真值。这不表示全部148571观察都错误。',
 '相机FIT仅允许冻结FIT背景族，要求物理身份、可见性、静止性及测量不确定性证据。留出目标不得加入自己3D点的三角化支持；至少两个其他可信观测和完整覆盖才有资格验收。开发中已查看的材料明确声明，不冒称新盲测。空集不能通过全部面部小于1像素准入。',
 'N4只计算条件K代数，不重估或选择K。第一YAML K：fx1658.0353708015184、fy1656.5394007261227、cx2001.5157183009285、cy1505.8269490738189，标注4000x3000。处理图写4096x3072、3605x2704、1532x2704；用户说数字可能记错。实际新版为1520x2704。',
 '条件H0：假设第一K就是4000x3000去畸变输出K，按高2704等比下采样并理想居中裁1520，则fx1494.4425475491018、fy1493.0941798544786、cx761.3168340952368、cy1357.2026900985354，整数像素中心。尚未把该假设写成正式训练相机。',
 '5项代数控制全部通过。4000到4096时画布和K同步缩放，最终投影完全等价；若4096画布直接使用未缩放的K，两假设在9个面部位置同一射线上最大差50.876px，5x5整图格最大87.298px。这是未确认假设之间的差，绝不是测得视频误差或根因。',
 '中间宽若取3605，中心裁剪起点1042与1043均有可能，与理想中心基线在面部最大约0.5px，彼此cx相差1px。误用1532宽度计算实际1520视频的起点会产生6px差，但这是刻意错误控制，未证实历史使用过。旧1080x1920条件K映射新尺寸在面部约0.030px也只说明代数，不是物理精度。',
 '实际K_rect、K是否随画布缩放、中间整数尺寸及实际裁剪偏移仍缺直接证据；未估畸变、外参或人体运动，没有新增特征/相机优化/高斯训练。合成原生界面R1e验收另见同工作目录native-interactive/result.json；只有该文件passed才算该阶段完成，不能代替实拍准入。',
 '证据：robust-portrait-122/video-update-20261027/observation-admission/{result,review,observation-set}.json及output-intrinsics-audit/{result,review}.json。右侧报告#robust122-n3和#robust122-n4；119历史模型保留。',
 '以下为历史说明，保留溯源：',previous
].join('\n\n');
const candidatePath=build+'/admission122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122材料与内参核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/admission122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_admission122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/admission122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/admission122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/admission122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide'+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/admission122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
