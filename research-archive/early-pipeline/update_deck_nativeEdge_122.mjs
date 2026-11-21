import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-edge-profile';
const figure=root+'/edge-profile-comparison.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/edgeAssociation-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.actual_review_complete!==true||review.uniform_rows!==11)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeEdgeProfile-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('残余畸变测量中的取边错误','原生像素定位与模型偏差');
replace('窗框与横向建筑边的原生等间隔裁片。','两条建筑轮廓的原生等间隔裁片。');
replace('最强梯度会跳邻边，干扰补偿参数的估计。','部分轮廓抖动减小，真实定位精度仍未确认。');
replace('连续关联仍缺可靠覆盖，尚未应用补偿或新训练。','明暗响应与邻边混合会产生系统偏差，尚未训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'上行为C1，下行为D1，原生96px裁片，保持比例，图内无文字',fit:'contain'});
image.frame={left:75,top:190,width,height:width*192/672};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 D4原生整数像素局部边缘成像剖面定位。", "D4在相同11条源边、3帧和每边120个位置上直接拟合整数像素，实际完成1320个局部拟合，保持旧观测和全部原始像素不变。没有重新求畸变或相机。", "旧方法在只有一条已知边缘的控制图中检出3个局部峰，证实插值/取峰可把一个边缘拆成多个响应。D3的多峰不应直接解释为多个物理边。", "新模型使用像素面积积分的模糊阶跃加局部颜色平面。36个已知单边缘实现控制均通过0.05px门槛。这些生成条件与拟合模型接近，不能据此声称真实人脸达到亚像素。", "同一冻结D1有效点集合下，C1直线垂距RMS 0.268至0.180px，D1为0.241至0.145px，D2为0.228至0.135px。未按新误差删除任何点。但A5和B1反而变差；更直也可能是模型偏差，不代表真实坐标更准。", "1320次拟合中1306次数值收敛，943次触及至少一项预登记参数界限。大量sigma触下界、部分局部方向达界，单一模糊边模型不能可靠描述全部视频边。保留全部结果，不放宽界限重新跑。", "已实际审阅全部11条等间隔原生裁片与22组两方法最大分歧裁片。A3/A4仍有窗框相邻边与开窗干扰，B2仍有多条建筑横线，B3有背景楼体交叠，不能自动当成一条孤立模糊边。", "控制中，仅在模糊后加入明暗非线性，定位就偏0.447px，而形式标准差只有0.00166px。两个邻近物理边会让单边模型偏1.546px，形式标准差仅0.00872px。拟合置信度严重小于模型偏差，不能作为物理亚像素保证。", "当前已验证采样取峰和模型响应两类测量误差，仍未证明它们是人像重影的唯一根因，也未证明这段视频的真实响应等于控制中的gamma。", "没有应用残余补偿或新启人物训练。旧13条面部留出最大13.04875764px、原K、旧相机和119模型均保留。", "本次是测量模型控制，不是新镜头补偿。3个局部几何参数为边偏移、局部方向和模糊宽度，各颜色通道使用仿射亮度平面和阶跃幅值，最小二乘消去线性光度参数。积分覆盖原生像素面积，未将原图二次去畸变或插值检测。", "1320处全部结果保留。统计直线度时沿用D1被冻结的旧有效点，不按D4残差或边界命中筛掉样本。1306数值收敛与943触界不能代表材料真实性。", "36个独立面积积分已知单边控制的最大定位误差约1.02e-8px，属于与模型接近的合成实现检查，不是实拍精度。更关键的模型失配控制：gamma-after-blur产生0.446784px偏差、形式标准差0.001663px；双边误拟合成单边产生1.545932px偏差、形式标准差0.008724px。真实视频未确定采用该gamma，不能将控制数值当成真实相机的偏差。", "图片上C1、下D1，每行7个96px原生裁片，取索引0/20/40/60/80/100/119。外部裁片展示不改变训练原图，图内无文字。", "本组建筑边D1至D4到此有界结束，未使用任何新补偿相机或提取器用于人物训练。后续审计完整项目证据与剩余可辨识性，不扫阈值、扩大sigma边界或重复旧试验。", "证据：robust-portrait-122/native-edge-profile/{plan,controls,result,summary,review}.json及原生裁片。其他51页保持不变，以下保留历史记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/nativeEdgeProfile-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生边缘定位.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeEdgeProfile-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeEdge_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeEdgeProfile-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeEdgeProfile-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeEdgeProfile-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeEdgeProfile-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
