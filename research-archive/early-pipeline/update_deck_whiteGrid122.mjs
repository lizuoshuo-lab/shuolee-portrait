import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../clip03-grid-coverage/array-cache-repair/material_evidence.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/whiteAllocation122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/whiteGrid122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('固定物体角点在筛选中流失','均衡特征分配后的固定物体对应');
replace('三排为消防栓','三排为屋顶拟合族、屋顶留出族与玻璃纹理疑点。');
replace('上方灯具230','屋顶循环轨迹从1条增至49条，需逐组确认材料身份。');
replace('固定表面留出支持','非地面固定物体留出族仅2组，未准入相机求解。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'屋顶拟合族屋顶留出族与玻璃疑点的原生128像素裁片',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height)/2,top:frame.top,width:frame.height,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N12d。相机继续使用第一组YAML K的4000×3000参考尺寸，下采样与居中裁剪到1520×2704。输出尺寸和K的实际导出关联尚未独立实证。本轮没有调整K、求解外参或训练。", "唯一因果变量是背景特征名额分配：仍为三个大横带各2000个背景点，各带细分8列×4行并逐格轮流选最高响应点。头部600点、其他人物1400点保持原编号。总预算8000，原生灰度图、检测参数、RootSIFT、双向ratio0.75、配对步长1/2、三视角循环及8px族规则不变。", "原N12b未保存未选点的angle和octave，因此每帧仅重建一次同参数检测器恢复元数据，并逐项验证所有原始xy/size/response和各8000个旧描述子完全一致。七帧全部通过。未重跑旧基线或替换其证据。", "旧族划分保持。新连接触及旧FIT和heldout时整族隔离，共3族。新族只在计算任何几何误差前采用固定SHA划分。隔离后的全部相机候选族为1002FIT/283heldout，这些数量不是物理认证。", "232帧屋顶搜索区保留点4增至136，循环1增至49。建筑立面搜索区11增至280，循环5增至52。全体循环5004降至4329，轨迹2740降至2519。区域含多尺度和混合材料，数量不可直接解释为独立固定点。", "按执行前规则选择83组，实际审阅全部原生128像素观测及16组256像素上下文。粗静态候选22FIT/11heldout，其中非地面固定物体7FIT/2heldout。未通过6FIT/3heldout条件。玻璃反射、树叶、遮挡和深度交界未为凑数转入有效材料。", "较明确的屋顶6组包含4FIT和2heldout，依据实墙端点、柱序核对同一个结构间隔。它们是粗候选，未认证独立亚像素位置。脸部实际材料只有4FIT/2heldout，其余头部ROI多是衣服与头发，不能当作面部对应。", "图中三排是row14族4060 FIT屋顶、row18族4045 heldout屋顶、row24族4009 heldout玻璃疑点。三列216、248、280帧，128×128原生裁片，选点在中心，无图中文字或二维扭曲。超出图像区域的黑色为裁片填充。", "输入一致性检查首次执行因在循环重复解压同一NPZ数组造成性能故障，保留首次失败和输入后，仅修复为一次加载数组。原84.687秒，修复40.013秒，累计124.700秒在原180秒预算内。未改变任何检测、匹配或筛选参数。", "7帧仅为464帧诊断子集。覆盖改善不能证明物理标定通过或唯一归因重影。后续不扫描网格或阈值，先研究真实材料定位不确定性并保持整族留出。119历史模型保留，无00端、SIBR或独立Spark新验收。", "来源：robust-portrait-122/video-update-20261027/clip03-grid-coverage/array-cache-repair/result.json、visual-review.json、allocation.json、material-evidence-index.json，计划clip03-grid-coverage-plan.json，修复记录clip03-grid-coverage/implementation-repair.json。", "以下保留历史诊断以便溯源："].concat([previous]).join('\n\n');
const candidatePath=build+'/whiteGrid122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122均衡特征分配对照.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/whiteGrid122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_whiteGrid122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/whiteGrid122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/whiteGrid122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/whiteGrid122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/whiteGrid122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
