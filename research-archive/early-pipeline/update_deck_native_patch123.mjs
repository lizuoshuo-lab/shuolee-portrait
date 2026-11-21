import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-native-patch123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-convergence123-deck-receipt.json'),review=await read(root+'/../video10-correspondence-contrast/head-roi-contrast/native-affine-contrast/visual-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_native_crop_review||review.rows.length!==24)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('相机误差降低，人体对齐仍未通过','局部细化没有修复人像对应');
replace('留出中位3.36降到1.29px','24组实拍查询：原匹配4组通过，两种细化均只有3组。');
replace('33个观测变差','鼻嘴查询虽能闭环，反向定位仍错2.40px。停止这条局部细化路线。');
replace('第484／510／537帧','同一鼻嘴查询：源图、原匹配、平移细化。未改相机，未启动训练。');
const figure=root+'/../video10-correspondence-contrast/head-roi-contrast/native-photometric-contrast/visual-review/row19_source_baseline_refined.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'原始第19组鼻嘴胡须查询，源图、原第二视角匹配、原生平移细化。3个384px原生裁片，未扭曲，未在图内写字。粗材料相似不能证明亚像素身份。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29。SMPL-X对齐仍未通过。此次只完成局部光度匹配分支的合成控制和实拍对照，及另一人物video4的原生输入核验。没有改K、外参、SMPL-X或高斯。", "24个已冻结原生查询，31px局部窗口，2D平移加亮度增益偏置，已知平移0.35/-0.40px、增益1.05、偏置2/255、噪声0.5/255、固定种子12313。24/24合成控制通过，中心误差中位0.008462px、最大0.039676px。只验证理想复制图小变换的实现。", "实拍复用整图RoMa结果，不做新推理。相同24组的A-B-C-A和各边反向定位共144拟合，统一30评估预算。保留原置信度0.9、双向与循环小于1px和参数边界限制。原4/24通过变成3/24，通过集为行12、16、20，无新增通过，行19失败。118/144成功，37次触及边界。", "第19行是FIT族163鼻嘴/胡须附近。原循环0.174px，平移细化循环0.585px但第一段FB误差2.403px且有边界/求解失败。循环自洽不是物理对应精度，未删观察或转移HELD。图像为三列384px原生裁片，中心查询，图内无文字。", "冻结局部解的只读导数投影显示，原4组通过中的3组满足预登记的缺失仿射几何支持条件。因此只追加一个不同几何模型的有界对照，无窗口/阈值扫描。该24组已经用于方法开发，不能当独立验证集。", "仿射几何控制已知矩阵[[1.03,.02],[-.015,.98]]、同平移与亮度、噪声和30评估预算，8列导数通过差分核验。24/24通过，中心最大0.019947px、角点最大0.068512px。", "真实仿射结果仍3/24通过，较平移无新增通过。95/144求解成功，52次触及边界。实际审阅两方案各24组120个原生192px裁片，另看鼻嘴384px上下文。耳/镜腿/胡须粗对应可辨，颈侧、头顶和背景存在遮挡与材料歧义。两方案均拒绝，局部细化分支结束。", "失败不能唯一归因于内参、外参、残余畸变或人物微动。相机先前修复的HELD仍93/161不小于1px、最大42.07px。物理认证的面部HELD为0，33诊断视角未覆盖860训练帧。", "旧灰衣SMPL-X面部残差中位43.99px，旧全身门槛容许约50至53px，没有面部解剖可见性保护。这是已证实的放行缺陷。独立投影算术误差小于0.0004px，不等于物理相机准确。", "所有7个历史源视频经完整SHA256匹配已确认只改名。花纹人物pattern是video4.mp4，初次旧路径不存在时在解码前停止并保留记录，哈希确认后只修正输入路径，未更换视频。已提取99/491原生2704x1520帧并保存原帧号/时戳。", "实际审阅video4的0、120、245、370、490帧，桌子遮住下身与脚，侧脸有头发遮挡。未见背面只描述这5个样本，不推广全部99帧。下一步只核验区域与可见性，保持相机、SMPL-X和训练准入独立。", "证据：robust-portrait-122/video10-correspondence-contrast/head-roi-contrast下native-patch-control、native-photometric-contrast、native-affine-contrast；local-clothed-surface-123/pattern-preflight及source-name-reconciliation.json。前51页保持原字节与渲染。没有新模型、00部署或云同步验证。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123原生匹配细化核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_native_patch123.py',preserve);
execFileSync(py,[b+'/preserve_deck_native_patch123.py'],{stdio:'inherit'});
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
