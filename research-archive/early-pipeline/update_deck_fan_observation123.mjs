import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='fan-observation123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis/fan-observation';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/sequence-diagnosis123-fixed-deck-receipt.json');
const review=await read(root+'/visual-review.json'),metrics=await read(root+'/descriptive-metrics.json');
const result=await read(root+'/result.json'),checks=await read(root+'/synthetic-checks.json'),publication=await read(root+'/publication.json');
if(await filehash(source)!=='3d89e84d953a8f7157c490f2631b40b7b957ffe5d12a93f0117caaecf36a4952'||prior.final_sha256!==await filehash(source)||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_native_figures_reviewed||review.quality_accepted||result.network_forwards_completed!==6||result.optimizer_steps!==0||result.gaussian_steps!==0||checks.unit_tests.tests!==12||checks.unit_tests.failures!==0||checks.unit_tests.errors!==0)throw Error('Evidence state mismatch');
const figureEntry=publication.record_images.find(x=>x.source_frame===155);
const figure=figureEntry.path.replaceAll('\\\\','/').replaceAll('\\','/');
if(await filehash(figure)!==figureEntry.sha256)throw Error('Image mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-layout.json',JSON.stringify(await p.inspect({kind:'layout'}),null,2));
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('共享体型拟合的求解与观测核查','专用面部观测对照');
replace('求解检查已修复，实拍改善尚待验证','检测位置仍有像素级分歧，尚未证明精度或重影改善');
replace('8项CPU回归通过。初始化只剩两肩，支持不足。下一步核验专用面部观测。','12项CPU测试与读取映射通过。两种检测中位差约2像素。下一步验证根节点初始化。');
replace('历史结果，原视频155帧：左原图，右共享体型SMPL-X叠加。手部保留先验。','视频第155帧，左起：原图、RTMW、FAN原始、翻转还原、热图合并。五栏同裁剪、同比例。');
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'原视频第155帧，同一原生192像素裁剪。左起原图、RTMW、FAN原始、翻转还原、热图合并。无烧字。'});
image.frame={left:160,top:270,width:960,height:192};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const rows=metrics.rows.map(r=>({source_frame:r.source_frame,fit:r.groups.observed_fit,held:r.groups.observed_held}));
p.resolve(last.id).speakerNotes.text=[
 '2026-10-29 专用 FAN 三帧面部观测对照。该页只报告观测和工程适配证据，不代表新人物拟合或高斯结果。',
 '固定原视频帧0、155、310，冻结相机、可见性和拟合/留出划分。共6次CPU网络forward，耗时16.221918376秒，GPU任务0，新拟合0，高斯步数0。',
 '每帧可见FIT点分别为1、15、11个。FAN热图合并与RTMW的逐点欧氏距离中位数分别为2.251358、2.030475、2.158373个原生像素，最大值分别为2.251358、3.704110、3.217868像素。这些是检测器间分歧，不是对物理真值的误差。',
 'FAN原始与翻转还原在可见FIT点上的最大差分别为0、0.980415、1.826826像素。帧155的4个语义留出点最大翻转差为1.386517像素。翻转一致性不等于绝对精度，不能当作标定后的定位不确定性。',
 '12项CPU单元测试和安装版解码器合成控制通过。实际三种分支均通过原生IBUG68静态51点的读取映射核验，17至67号值准确保留，非单位crop元数据未重复应用。所有新FAN观测仍不参与拟合，因为缺少独立物理身份确认和标定定位不确定性。',
 '冻结153条静态面部语义记录，其中34条可见、27条FIT、7条HELD、119条UNKNOWN。没有按残差或分数更改选择，没有把隐藏灰点或留出点加入拟合。原帧0仅一个可见鼻点且没有面部留出。',
 'RTMW改为FAN同时改变检测模型和裁剪上下文，不能把组合变化归因于单一组件。连续坐标仅移除官方解码器的最终整数截断，不消除裁剪/重采样误差，不证明物理亚像素精度。原生脸宽约42至62像素，放大至256像素没有产生新细节。',
 '尚无独立物理真值，因此不能判断FAN比RTMW更准确，不能称重影改善或质量通过。三帧不代表全部94帧或跨人物泛化。下一步单独验证根节点初始化策略，本页没有运行该测试。',
 '图像：'+figure+' SHA256 '+figureEntry.sha256+'。五栏统一原生192×192裁剪，组合960×192，无文字烧入图片。文字说明保持为可编辑PPT文字。',
 '来源：'+root+'/result.json；'+root+'/descriptive-metrics.json；'+root+'/visual-review.json；'+root+'/generic-checks.json；'+root+'/synthetic-checks.json。',
 '逐帧原始汇总：'+JSON.stringify(rows)
].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123专用面部观测对照.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_fan_observation123.py',preserve);execFileSync(py,[b+'/preserve_deck_fan_observation123.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:prior.final_sha256},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
await fs.mkdir(b+'/'+tag+'-renders',{recursive:true});
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 await fs.writeFile(b+'/'+tag+'-renders/slide-'+String(i+1).padStart(2,'0')+'.png',bytes);if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
if(await filehash(source)!==prior.final_sha256)throw Error('Canonical changed during authoring');
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:prior.final_sha256,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,first51_render_hashes_identical:true,canonical_replaced:false,figure,figure_sha256:figureEntry.sha256,actual_changed_slide_review_pending:true},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
