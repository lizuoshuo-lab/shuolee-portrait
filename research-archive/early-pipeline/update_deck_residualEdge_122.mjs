import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/residual-edge-association';
const figure=root+'/edge-comparison.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/residualLinesVerified-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.actual_review_complete!==true||review.rows_reviewed!==11)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/edgeAssociation-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('残余畸变补偿的独立检验','残余畸变测量中的取边错误');
replace('三帧建筑原生裁片：第0、232、463帧。','窗框与横向建筑边的原生等间隔裁片。');
replace('6条拟合、5条留出；留出误差未下降。','最强梯度会跳邻边，干扰补偿参数的估计。');
replace('候选未采用。对应错配仍在，尚未启动新训练。','连续关联仍缺可靠覆盖，尚未应用补偿或新训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'上行为A3窗框，下行为B2建筑横边，原生96px裁片，保持比例，图内无文字',fit:'contain'});
image.frame={left:75,top:190,width,height:width*192/672};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 D2原始取边审阅与D3只改变关联的测量试验。", "继续沿残余畸变补偿方向，先实测其观测入口。D2确认逐截面最强梯度会受相邻边、交叉线和局部纹理影响，不能自动代表同一物理直边。", "D3只改变同一批原生梯度剖面的关联方式。保留全部局部峰，从一个已登记种子出发，仅在前后都唯一对应时继续，遇分叉或缺失即停。没有平滑原图、拉直边缘或拟合新的畸变参数。", "11条边中5条起点即数值歧义，6条形成局部候选，总计170个位置。旧算法有1191个可用标记，但两套有效性规则不同，不能将170/1191写成准确率或留存率。", "B1得到110个连续候选，C1/C3/B2/B3/D2分别为12/2/4/9/33。这些只证明局部数值关联，不代表物理身份或亚像素定位通过。", "实际审阅两页全部11条边的种子、停止点和候选端点。窗框与建筑缝附近确有交叉干扰，同时部分视觉上清晰的轮廓也被拒绝。多个亚像素局部峰可能属于同一模糊边，不能一律称为不同物理边。", "已知弯曲边的控制中，曲线3.599686px起伏逐值保留，旁边更强的平行边没有吸走关联，歧义和缺失能停止。但该合成控制不能替代真实视频的误差验证。", "本轮没有新的补偿系数或改进人像。旧13条面部留出最大13.04875764px保持原值，主查看器继续显示119历史模型。", "D2实际核对全部13条等间隔原生裁片及26组相邻跳变。A3最大法向偏移突变8.947px，B3为5.979px。解析区间覆盖D1全系数范围，A3/A4/B3部分三点偏离邻点弦线下界3.019/3.676/4.188px。此结论仅针对这些已观测三点和该光滑逆映射族，不是面部误差或任意镜头模型结论。", "D3沿相同D1灰度梯度剖面枚举全部原门槛局部峰，种子及固定2px邻接范围预登记。前向与反向都唯一才继续，歧义/缺失即停止。未填空、重播种、拟合直线、平滑曲线或扫参数，原始最强峰偏移逐值一致。", "图上行为A3、下行为B2，每行7个原生96px裁片，采样索引0/20/40/60/80/100/119。图中无标注，显示为整图等比例放大，不是新补偿图。", "旧119预览/原K/外参/输入及划分全部保留，未训练新人物。下一步解决物理边两侧及成像剖面歧义后，才可登记新的独立补偿验证。", "证据：robust-portrait-122/residual-edge-identity/{plan,result,interval-bound,review}.json 和 residual-edge-association/{plan,controls,result,review}.json。以下保留原阶段记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/edgeAssociation-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122取边关联核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/edgeAssociation-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_residualEdge_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/edgeAssociation-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/edgeAssociation-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/edgeAssociation-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/edgeAssociation-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
