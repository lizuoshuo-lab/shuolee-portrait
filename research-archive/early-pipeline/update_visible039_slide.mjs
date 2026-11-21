import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx',phase=base+'/current/smplx-nlf-visible-projection-fresh-039';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='39508842bee6cadabe220c63b19b701a0f8c020e45798e2d08fe7541de53ace5';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='visible_projection_no_clear_geometry_gain')throw Error('Review required');
const previous=JSON.parse(await fs.readFile(build+'/template038-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const beforePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(beforePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{const e=row.elements.find(x=>x.text===before);if(!e)throw Error('Missing text '+before);p.resolve(e.id).text.replace(before,after);};
replace('人体模板与共享体形','身体先验与可见表面约束');
replace('模板定义形状空间，真实相机与表面观测约束人物对齐。','比较三维表面位置与可见二维对应对身体配准的作用。');
replace('左组：实拍、中性模板、男性模板。右组：同一世界侧面相机。',
 '左组：实拍、三维约束、可见二维约束。右组：同一世界侧面相机。');
replace('胸腹轮廓随模板明显变化，肢体对齐仍未通过。\n两组从零体形拟合；二维残差为 8.18 与 9.01 像素。',
 '可见二维约束没有带来明显的整体对齐改善。\n两组从头拟合，关节残差均约 9 像素。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two images');
for(const [i,file] of ['source-000045.jpg','fixed-left90.jpg'].entries()){
 const im=p.resolve(images[i].id);
 im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',
  alt:i?'固定23帧同一侧面相机，三维表面与可见二维约束':'45帧原图与两组从头拟合'});
 im.crop={left:0,top:0,right:0,bottom:0};
}
slide.speakerNotes.text=`一、问题与先验的职责
图像网络提供姿态、表面对应和不确定性，SMPL-X模板决定平均人体、形状空间、关节回归及蒙皮。此前模板对照已确认中性与男性模板会明显改变胸腹轮廓，但没有同时解决肢体配准。继续只靠增加点数或拉近单目三维预测，可能把预测深度偏差也带入人体。此次对照检查图像对应和三维表面位置的不同作用，继续使用现有NLF/GVHMR/VPoser资源，不新增SAM3DBody下载。

二、输入与可见性
59张标定顺序视频帧，非59台同步相机。NLF原始输出有10475个参数化顶点和1024个非参数化对应点，模型使用NeurIPS2024官方v0.3.2 patch6。原始预测散列7e4d3a519d35c4afd9229f0a7fe9402cfa041e009426fd29fc62f8eba19ee799。所有新实验从原图、原始预测、原相机、原遮罩和许可模板开始，不加载旧个体拟合、高斯、形变或优化器。
筛选基于原始参数网格：面积加权外法线与朝向相机方向的余弦大于0.15，四邻近像素深度缓冲中最小差小于0.010模型米，非参数点的二维目标位于人物遮罩内且相机深度为正。固定可见性，不随拟合更新以免模型自行隐藏误差。总计18637次观测，每帧250—413点，中位291。已查看0、23、45帧点覆盖；它是网络预测，不是人工皮肤标注。
相机投影保持原始内参及半像素约定。角点式主点投影转换到像素中心时减0.5。世界坐标、相机坐标及像素重放最大差0.000324像素，这只验证代数接口，不证明相机或预测的物理准确性。

三、跨视角一致性诊断
把每个表面索引暂时假定为静止世界点，先做加权归一化线性三角化，再做10步稳健GaussNewton，像素残差尺度3。以连续10帧为一组留出，训练至少10次可见观测、分布于至少两个时间块，留出至少2次。488个表面索引共1800组拟合，得到15454次留出观测。
训练残差中位4.996像素，留出8.853像素、95%分位21.832像素。手臂13.047、躯干10.471、头颈6.419、腿脚7.158像素。留出观测56.39%在10像素内、92.60%在20像素内。人物微动、学习对应、遮挡与标定误差均可能参与；不能用该诊断断定唯一误差来源。诊断三角化不用于拟合初始化，也不按其残差删点。

四、两组从头拟合
两组均使用已有SMPLX_MALE.npz，模板散列ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5。10维体形β=0，新Adam优化器，各1200步，共2400次身体更新，0次高斯更新。共同初始化和原始观测逐数组一致，旧模型仅作为历史证据。
1—300步只拟合整体旋转、平移和共享尺度，301—700步开放共享关节和体形，701—1200步开放逐帧有界修正。共同关节每轴0.30弧度、逐帧0.10弧度，旋转在SO(3)上组合。β在±1内；原始HaMeR只提供局部手指，不搬用裁切相机根旋转。真实相机固定。
身体、脚、手的二维稳健尺度20、18、10像素，脚权重0.2、手权重0.5。保留单向遮罩包含、局部姿态和关节先验、体形正则、脚底相对运动与估计地面上方1—7厘米软区间、时序平滑。其余实现与预算保持一致。

五、表面目标的区别
三维对照：1024个原始非参数对应点减去骨盆并逆转整体朝向，在模型局部空间比较同索引顶点。L3D为逐帧逐点不确定性加权的三个坐标稳健残差均值，尺度τ=0.06模型米，ρ(r)=√(1+r²)−1。
二维候选：L2D=(1/T)Σt(1/(3Wt))Σi wti Σk∈{u,v}ρ((πtk(Vti)−utik)/stik)，Wt=Σi wti，stik=ftk τ/Zti_pred。焦距以像素、深度以模型米计，因此s以像素计，固定在45.31—78.56像素范围，中位64.59。分母3沿用原三维平均因子，是本地匹配设计而非概率推导。w为可见性乘预测不确定性负1.5次幂，并逐帧归一化；不声称概率已校准。
单帧二维项不直接规定沿相机射线的深度，多视角、共享体形和其他约束仍提供空间限制。两组表面权重300—500步从0增到1。候选同时改变可见点选择、残差空间及对整体根变量的作用，不是纯去掉一个深度坐标的消融。有限非零梯度和体形有限差分通过，检查后参数原样恢复。

六、实测效果
左图为45帧原图和两种网格叠加，右图为固定23帧同一世界侧面相机。查看全部59帧联系表及固定正面、左右90度、上下25度，轮廓与肢体位置接近，没有明显整体改善。裸人体不需要填满宽松外套和裤子；不能把更大轮廓或更直的膝盖自动判为正确。
三维/二维两组可见关节残差中位9.007/8.999像素，95%分位34.200/34.105。未优化鞋纹诊断1.393/1.410像素。表面投影中位13.443/13.052像素，但95%分位28.045/29.858像素，加权二维目标0.010337/0.010583。不能只挑中位数下降。与NLF局部三维预测的距离23.730/35.786模型毫米是预测一致性，不是人体误差。原生模板最终网格重放差均为0，只确认接口。

七、未解决问题与下一步
在最终状态固定姿态、根和手部，只读计算每项对β的导数。形状相关各损失复现差小于1.8×10⁻⁸，两组均5个系数绝对值超过0.85，其中5/4个总下降方向仍向边界外。tanh链式导数最小0.080/0.086，说明当前参数化抑制了边界附近更新，但不证明放宽可改善真实身体。
图像相关项与地面接触项的β梯度余弦−0.911/−0.932，属于局部方向冲突，不能直接判定地面错误。下一步依据原始鞋底、站姿投影和尺度核查接触假设，再选择一次从头对照；必要时保留相对鞋纹运动、只移除绝对脚底高度软区间。避免同时放松多种约束或继续扫描表面权重。当前没有替换查看器或部署00，完整高斯训练仍等几何通过后按20万新点、18000步执行。

八、来源与证据
NLF官方代码 https://github.com/isarandi/nlf ，SMPL-X官方代码 https://github.com/vchoutas/smplx 。现有模板由用户提供，NLF官方权重许可限制于非商业研究，本轮仅本地研究。这里是受控本地适配，不是缺失入口的GaussianAvatars_SMPLX官方完整复现。
实际证据：${phase}/visible-input-audit.json、cross-view-consistency.json、surface3d/protocol.json、visible2d/protocol.json、review/comparison.json、shape-pressure.json和phase-conclusion.json。照片与网格均来自真实素材和计算结果，无生成式补全。`;
await fs.writeFile(build+'/visible039-edited-layout.json',JSON.stringify((await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']})).records,null,2));
const draft=build+'/visible039-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_可见表面_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/visible039-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/visible039-final-renders';await fs.mkdir(renderdir,{recursive:true});const checks=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
 const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
 if(i<54&&!unchanged)throw Error('Unintended change '+(i+1));checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/visible039-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),
 changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,unchanged_first54:true,review_required:renderdir+'/55.png'}));
