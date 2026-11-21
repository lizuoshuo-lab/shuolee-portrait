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
const prior=await read(build+'/material122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/visibility122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('人像对齐：','遮挡引起的错误材料对应');
replace('历史鼻区材料对照','新版原生裁片：上排为278帧参照，下排为遮挡后的错误落点。');
replace('旧版脸框含','双向误差小于0.023像素，眼部和靴子轨迹仍会跟到帽子、裙子。');
replace('新版戴帽人物已','RoMa拒绝6个反例，但远视角支持不足。本轮尚未训练。');
const el=slide.elements.find(e=>e.kind==='image');const im=p.resolve(el.id),frame=im.frame;
im.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'Six native occlusion identity counterexamples. Top anchor frame278, bottom incorrect temporal correspondence. No annotations or geometric warp.',fit:'contain'});
im.frame=frame;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-27 N1/N2，新版冬帽人物1520x2704、557帧，sourceSHA32be30636adabfc4c25a1c8387cf19302d3ca39b4f040072081646d36523317c。',
 'N1预登记307种子：面部9、衣物38、鞋20、地面240。全557帧相邻双向光流，固定FB<0.5px，首次失败停止单方向。不重新去畸变，不估K、外参或三维结构，不读取旧相机/旧几何。27行登记原生96px裁片全部浏览。',
 '图中六列依次为track2/frame70、3/70、11/0、287/0、287/556、291/556。上排是frame278原像素参照，下排为N1对应落点。眼部2/3跟到帽子，地面11与鞋部287/291跟到裙子。相邻FB分别0.007422、0.022905、0.003310、0.003663、0.001606、0.001125px，均被原算法标有效。这是可见性/物理身份失败证据，不能据此唯一归因所有历史重影。',
 'N1每12帧直接路径与连续路径存在分歧，面部282次检查中16次>=1px、最大4.4955px。两路径分歧本身不说明哪条错误，不把所有大差异称累计漂移。原始观察与失败全留，没有按几何误差删除或更改fit/heldout。',
 'N2在运行前登记6个负例与4个图像对278对0/70/266/556，使用已有官方RoMa Outdoor和DINOv2权重，无下载。固定双向最小confidence>=0.9及原生FB<1.0px。4对全部307种子预测均保存，没有调阈值。6负例全部拒绝；近邻266保留7/9面部、36/38衣物、6/20鞋、0/240地面；其余三个远视角全部保留0。',
 'N2五张原生审阅页已实际浏览，含四视角全部9面部候选及六个负例。近邻可见面部纹理局部接近，远视角含遮挡、重复与背景误匹配。低置信度拒绝这些反例不等于普适识别遮挡或亚像素真值。6负例来自N1已知失败，是有限诊断集。学习置信度与双向误差都不是独立物理对应或测量不确定性。',
 '全帧相机支持仍不够；实际K_rect与裁剪参数未独立实证，背景负对照与人物微动须分开验证。没有新高斯、PLY或相机发布。当前119历史预览质量仍未通过。目标模式继续。',
 '证据路径robust-portrait-122/video-update-20261027/native-temporal-material/{result,visual-review,counterexamples_index}.json及material-visibility/{result,visual-review,visual-index}.json。源图为同目录counterexamples_anchor_and_target.png。所有坐标为原生MP4整数像素中心。',
 '以下为历史说明，保留溯源：',previous
].join('\n\n');
const candidatePath=build+'/visibility122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122遮挡材料对应.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/visibility122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_visibility122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/visibility122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/visibility122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/visibility122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide'+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/visibility122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
