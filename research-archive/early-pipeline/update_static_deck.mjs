import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='be51ddd7e6d47d49f1fee886383bffe9e4ef0209018e02b40a78a20f9dd0cb65';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Report changed; preserve external edits');
const result=JSON.parse(await fs.readFile(base+'/current/static-surface-comparison-067/review-decision.json','utf8'));
if(!result.actually_inspected)throw Error('Actual images must be reviewed before changing claims');
const alpha=JSON.parse(await fs.readFile(base+'/current/static-white-edge-audit-068/review-decision.json','utf8'));
if(!alpha.actually_inspected)throw Error('Boundary review required');
const previous=JSON.parse(await fs.readFile(build+'/foot054-deck-receipt.json','utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']})).records;
await fs.writeFile(build+'/static-preserved-prior-notes.json',JSON.stringify(snap.filter((x,i)=>[1,19,44,50,54].includes(i)).map(x=>({id:x.id,notes:x.notes})),null,2));
const edit=(i,before,after)=>{const e=snap[i].elements.find(e=>e.text===before);if(!e)throw Error('Missing anchor '+before);p.resolve(e.id).text.replace(before,after);};
const notes=(i,text)=>{p.resolve(snap[i].id).speakerNotes.text=text;};
const imageReplace=async(i,k,path,pos)=>{const e=snap[i].elements.filter(x=>x.kind==='image')[k],im=p.resolve(e.id);im.replace({blob:await fs.readFile(path),contentType:path.endsWith('.png')?'image/png':'image/jpeg',fit:'contain',alt:'实际同相机静态高斯渲染对照'});im.crop={left:0,top:0,right:0,bottom:0};if(pos)im.position=pos;};
edit(1,'动态人体与真实背景的重建目标','当前目标：整个穿衣人物的静态重建');
edit(1,snap[1].elements[2].text,'全部实拍方向约束同一组三维高斯。位置、尺度、旋转和透明度共享，\n图像差异由相机投影、遮挡和视角相关颜色解释。');
edit(1,snap[1].elements[3].text,'175 个有效帧，包含 59 个标定相机和 116 个插值相机。\n每轮重新建立 20 万初始点，训练 18,000 步。');
edit(1,snap[1].elements[4].text,'固定几何，改变俯仰、方位与距离，检查衣服轮廓、人体外观和连续性。\n当前采用共享静态表示，背景随后在同一坐标系中单独验证。');
notes(1,'当前用户要求整体静态，优先于历史动态规划。当前175个真实图像均参与训练；59个相机由SfM标定，其余116个由时间插值获得，不称全部相机重新标定。单一静态场仍可能无法解释真实微动。每一轮从原始观测和官方预训练输出重新拟合粗范围，重新生成点和优化器；没有继承上轮个体几何。稀疏观测限制及不足汇总于末页。数据与状态：shuolee/long-term-plan.md，current/person-static-pgsr-fresh-064，person-static-no-scale-fresh-065，person-static-no-multiview-fresh-066。');
edit(19,snap[19].elements[0].text,'静态人物的 PGSR 训练目标');
await imageReplace(19,0,build+'/static-formula.png',{left:66,top:173,width:1148,height:63});
edit(19,snap[19].elements[3].text,'颜色项采用 L₁ 与结构相似性，轮廓项监督不透明度。最小尺度压缩从起点启用；\n法线、多视图往返投影与图像块一致性从 7,000 步后启用。');
edit(19,snap[19].elements[4].text,'20 万点重新初始化，18,000 步；增密截止 15,000 步，点数上限 50 万，\n不透明度裁剪阈值 0.005，绝对梯度分裂阈值 0.0008。');
edit(19,snap[19].elements[5].text,'图像高度 1,536，保持完整相机内参；邻居方向相差 4°—24°。\n使用真实人物内部的 7×7 图像块，每次最多采样 8,192 个。');
notes(19,`这是PGSR官方代码的本地人物适配，不是论文30k完整协议复现。公式各项：RGB=0.8 L1+0.2(1-SSIM)，alpha为整图不透明度与人物遮罩的L1；Ls为可见高斯最小标准差的平均值；LN为渲染法线与平面深度法线差，使用真实图像梯度权重和前景内部掩码；LG为真实相邻视图深度重投影往返像素距离，使用exp(-距离)的停止梯度权重；LP为局部平面单应性变换后的归一化相关损失。Ln、LG、LP在第7001步开始。
颜色损失在完整相机渲染后的前景包围ROI计算，ROI每边增加人物高度10%的边距；没有改变裁切相机。完整图像按高度1536等比例缩放，像素内参同步缩放。相邻方向4到24度，优先接近12度，最多16个邻居。几何筛选往返像素误差小于1、源alpha大于0.9、正深度及双侧内部遮罩；NCC的整个映射图像块必须仍在前景。
初始化依据175帧遮罩投票形成体积边界，间隔0.012模型米，投票阈值0.97；59标定相机权重1，116插值相机权重0.5。20万点来自新表面采样及可观测服装特征，不是裸人体表面；新人体拟合只提供粗范围与尺度。064、065、066重新生成的初始数组完全一致。
每1000步升一级球谐，最高3阶；增密500步后每100步，至15000步前，常规梯度阈值0.0002，绝对梯度阈值0.0008，每次新增绝对分裂最多25000点；每1000步移除少于2帧实际合成观测的点。每3000步和白背景500步重置不透明度。两个对照只分别关闭最小尺度项或几何与NCC项，单视图法线保持。
来源：https://github.com/zju3dv/PGSR ，https://zju3dv.github.io/pgsr/paper/pgsr.pdf 。代码de24f1a及本地shuolee_static.py，完整参数保存在各gaussian-run/config.json；本地研究许可，不代表已取得PGSR商业许可。`);
edit(44,snap[44].elements[0].text,'静态表面约束的侧面效果');
await imageReplace(44,0,base+'/current/static-surface-comparison-067/right90-064.png',{left:190,top:128,width:280,height:426});
await imageReplace(44,1,base+'/current/static-surface-comparison-067/right90-'+result.display_candidate+'.png',{left:790,top:128,width:280,height:426});
edit(44,snap[44].elements[4].text,'同一侧面相机：左为全部表面约束，右为'+result.display_candidate_label+'。');
edit(44,snap[44].elements[5].text,result.ppt_effect);
edit(44,snap[44].elements[6].text,'两组均从头训练 18,000 步，固定相机、初始点与增密规则。整体结论与不足见末页。');
for(const idx of [4,5,6])p.resolve(snap[44].elements[idx].id).position={...snap[44].elements[idx].position,top:[565,605,646][idx-4],height:36};
notes(44,JSON.stringify(result,null,2)+'\n照片不是独立新视角真值。图为真实高斯渲染，来自static-surface-comparison-067。064、065和066的控制变量核验分别见controlled-protocol.json。只依据当前图像作局部结论；整体验收未通过。');
edit(31,snap[31].elements[0].text,'人物局部容量与边界覆盖的验证顺序');
edit(31,snap[31].elements[2].text,'症状定位：固定真实相机与问题区域，分别观察颜色、累计不透明度与点中心。\n区分输入、投影、几何支持和渲染阶段的误差。');
edit(31,snap[31].elements[3].text,'轮廓学习：逐视角核验人物遮罩、渲染覆盖与边界梯度。\n先区分缺少高斯覆盖、轮廓外多余覆盖与真实图像观测冲突。');
edit(31,snap[31].elements[4].text,'对照设计：先用诊断筛选原因，再固定一个主变量、数据与训练预算。\n预先定义支持和否定预测、主体保护标准与停止条件。');
notes(31,'当前动态路线已经废弃。固定共享静态高斯，采用实际图像与轮廓监督。模型的非零梯度只证明可以优化，不证明监督一致或几何正确。');
edit(50,snap[50].elements[0].text,'轮廓覆盖与前景颜色共同优化');
await imageReplace(50,0,build+'/alpha-formula.png',{left:66,top:170,width:690,height:395});
edit(50,snap[50].elements[3].text,'前景颜色与覆盖度\n接受同一轮实拍监督\n保留原人物遮罩');
edit(50,snap[50].elements[4].text,'轮廓区域单独归一化，使细边界在优化中获得明确权重。');
edit(50,snap[50].elements[5].text,'黑白合成共同约束颜色与覆盖度。输入先预乘遮罩，再进行缩放。');
edit(50,snap[50].elements[6].text,'本地静态训练适配。原始相机、遮罩与几何损失保持，全部参数从头训练。');
notes(50,'本页对应069。F为预乘前景颜色，A为累计不透明度，b为合成背景。R为双线性重采样。Omega是人物包围框及每边10%人物高度边距，Gamma是11像素膨胀与侵蚀差大于0.01的轮廓带。Lpair由两个背景分别0.8L1+0.2(1-SSIM)取平均，Lalpha由ROI与边界平均绝对误差等权平均，外部系数0.1。几何项与064一致。黑白是解耦颜色与覆盖的实验手段，不能被称为解决观测缺失。原生064边界有非零位置/尺度/旋转/不透明度梯度，但整图均值弱化轮廓残差。实拍分割不是精确alpha真值。证据：static-white-edge-audit-068/boundary-before/boundary-audit.json，person-static-alpha-fresh-069/controlled-protocol.json。');
edit(51,snap[51].elements[0].text,'边界学习的实际对照');
edit(51,snap[51].elements[1].text,'同一观察相机与裁片：实拍原图、原静态模型、边界组合对照。');
edit(51,snap[51].elements[2].text,'');edit(51,snap[51].elements[3].text,'');
await imageReplace(51,0,base+'/current/static-white-edge-audit-068/head-boundary-comparison.jpg',{left:66,top:205,width:1148,height:368});
edit(51,snap[51].elements[5].text,'后脑亮壳减轻；覆盖位置由独立不透明度图继续检查，整体结论见末页。');
edit(51,snap[51].elements[6].text,'真实训练观测的同相机渲染。两轮均从20万初始点训练18,000步。');
notes(51,JSON.stringify(alpha,null,2)+'\n左侧实拍原图，中间基线064，右侧069。三者采用相同像素裁片。没有按颜色删除浅色点，没有更换相机或仅切换白背景遮盖白边。每个侧面和未知视角仍独立检查。');
edit(54,snap[54].elements[0].text,'静态整体重建：作用、不足与下一步');
edit(54,snap[54].elements[2].text,alpha.ppt_summary);
await imageReplace(54,0,base+'/current/static-surface-comparison-067/below25-064.png',{left:190,top:178,width:250,height:380});
await imageReplace(54,1,base+'/current/static-surface-comparison-067/below25-'+result.display_candidate+'.png',{left:800,top:178,width:250,height:380});
edit(54,snap[54].elements[4].text,'同一仰视方向：全部约束与'+result.display_candidate_label+'，不以训练图像拟合代替新视角验收。');
edit(54,snap[54].elements[5].text,alpha.ppt_limit);
notes(54,JSON.stringify(result,null,2)+'\n'+JSON.stringify(alpha,null,2)+'\n069尚未通过边界验收：六个已见视图内侧覆盖不足平均从2.80%升至5.01%，外侧过量覆盖从13.24%降至5.66%。平均alpha误差下降不能掩盖局部退化。三项同时改变，只能作为组合效果。固定064末态的只读梯度核验发现透明度项相对RGB的位置梯度范数约3.3%至4.2%，11像素软轮廓带只有约9%进入法线项；没有Adam预条件，不能据此认定训练全过程根因。下一轮先按实际合成贡献分析投影、遮挡和跨相机轮廓支持，再预先固定单变量、支持和否定预测及停止条件。070未启动，不自动扫描权重。\n当前不足集中汇总：侧面和仰视无独立真值，真实微动与水平拍摄范围可能限制静态共享模型；灰色图和点中心只能说明模型自身支持，不能证明三维准确。标准PLY导出需要完整位置、协方差、球谐旋转和原生重载核验。实际本地独立导入并不等于00实机验收。PGSR商业许可未确认。本轮未部署00。动态路线已废弃，不作为后续扩展。');
// Remove abandoned dynamic-only proposals while preserving original source evidence.
const removed=[33,52,53],kept=Array.from({length:55},(_,i)=>i).filter(i=>!removed.includes(i));
const changed=[2,20,32,45,51,52,55];
for(const i of kept)if(!changed.includes(i+1)){
 const bytes=new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer());
 if(digest(bytes)!==previous.renders[i].render_sha256)throw Error('Unexpected pre-footer change '+(i+1));
}
for(let j=0;j<kept.length;j++)for(const e of snap[kept[j]].elements){if(/^\d+ \/ 55$/.test(e.text||''))p.resolve(e.id).text.replace(e.text,`${j+1} / ${kept.length}`);}
p.slides.keep(kept);
const draft=build+'/static-constraints-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_静态表面对照.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:draft,finalPath,
pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
explicitTotalSlideCount:kept.length,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
verifyArtifactToolImport:true,receiptPath:build+'/static-constraints-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));const renders=[];
for(let i=0;i<kept.length;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(changed.includes(kept[i]+1))await fs.writeFile(build+'/static-final-'+(kept[i]+1)+'.png',bytes);
 renders.push({slide:i+1,previous_slide:kept[i]+1,render_sha256:h,changed:changed.includes(kept[i]+1)});
}
await fs.writeFile(build+'/static-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:changed,renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:changed,canonical_replaced:false}));
