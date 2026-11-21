import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-absolute-surface-fresh-032';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='e89bc9db00e516a54be682337f510750a238c759fe3c6164e5ef10086f0e55ba';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Source changed');
const result=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(result.state!=='depth_order_component_improved_body_geometry_not_accepted')throw Error('Unreviewed result');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count changed');
const oldRenders=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await p.export({slide:p.slides.items[i],format:'png',scale:1})).arrayBuffer());
 oldRenders.push(digest(bytes));
 if(i===54)await fs.writeFile(build+'/surface032-source-slide55.png',bytes);
}
const inspection=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','notes.text']});
const row=inspection.records[0],slide=p.resolve(row.id);
const replace=(before,after)=>{
 const el=row.elements.find(x=>x.text===before);if(!el)throw Error('Missing text anchor '+before);
 p.resolve(el.id).text.replace(before,after);
};
replace('整体配准与人体姿态','外表面深度与人体几何');
replace('固定体形与关节，只调整整体旋转和平移，检查摆放误差。',
 '原始衣物纹理提供外表面深度，约束身体沿观察射线的前后次序。');
replace('左组：原始预测。右组：整体配准后。每组依次为实拍、GVHMR、Multi-HMR。',
 '左组：实拍、从头对照、深度约束。右组：同一世界相机下的两种身体。');
replace('整体配准减轻脚部悬空，局部屈膝与手臂偏差仍在。\n新的身体先验尚未通过，几何核验通过后再训练高斯。',
 '共同诊断射线的深度冲突由 477 降至 282，侧面姿态未明显改善。\n身体体积减小约 1.95%，二维残差增加，本轮未通过几何验收。');
const im=row.elements.find(x=>x.kind==='image');if(!im)throw Error('Missing image');
const image=p.resolve(im.id);
image.replace({blob:await fs.readFile(phase+'/review/geometry-comparison.jpg'),contentType:'image/jpeg',fit:'contain',
 alt:'原图、从头对照与外表面深度约束候选，以及同一固定世界相机的侧面网格比较'});
// Actual full image is1857x886. No panel is cropped.
const width=1857/886*382;
image.frame={left:(1280-width)/2,top:174,width,height:382};
let notes=row.notes.text;
if(!notes.includes('十二、目前解决的问题与不足'))throw Error('Notes anchor missing');
const section=`十二、原始外表面深度与裸人体的单向约束
031从原始COLMAP图像特征重建三维外表面支持，032只增加绝对深度顺序损失做两组从头对照。输入仍是59张依次拍摄的原始标定图像，不能称为59台同步相机。没有加载023的个体拟合来初始化。023仅在两组结束后用于对照重放验证，控制臂所有保存数组与023完全相等。
深度证据沿用014按原始图像内部纹理筛选的312条稳定轨迹，重新读取原始images.bin的特征坐标。相同点同一帧的重复特征，范围超过1像素则剔除，否则取均值。三角化以DLT初始化，使用soft-L1、2像素尺度和最多60次函数求值。评价第t帧时，删除包含t的连续10帧区块后再三角化，避免用当前像素直接决定自己的深度。相机仍来自使用这些图像的SfM，不能把该区块检查称为独立相机真值。
每个三维点计算固定相机条件下的重投影雅可比J。像素噪声尺度取残差MAD与1像素中的较大值，线性条件协方差为σ²(JᵀJ)⁺。相机Z方向上的1.96倍标准差与各删除区块重建的深度最大偏差取较大值，构成操作性误差带。这没有计入未知的相机系统偏差和真实身体微动，不是经过标定的95%真实精度区间。
筛选至少3个可用区块、留区块投影误差不超过3像素、深度误差带不超过0.02个名义模型米，得到182条轨迹、3385次观测。SfM的米制比例未知，长度统一除以原始GVHMR骨盆近似静止求出的比例3.180052世界单位/模型米。这里的厘米、米均是模型归一化单位，不能解释为实际测量距离。
参与约束的躯干、大腿、小腿来自图像中人物包围框高度分段，不是准确的人体分割标签。按每组稳定点ID排序交替分配，62条轨迹、1098次观测用于目标，另外60条、1062次观测用于已见诊断。头部和脚部均不接受本次新增损失。诊断图像和点已被反复检查，因此也不是独立留出测试。
对于图像中的一个可见外表面点，定义Z外为删除所在区块后三角化的相机Z深度，Z体为该条原始观察射线与裸人体网格的第一个正深度交点，ε为上述误差带和0.01个名义模型米的较大值。若Z体在Z外前方超过ε，就存在模型穿过已观测外表面的冲突。残差r=max(0,Z外−Z体−ε)/(0.02s₀)，ρ(r)=√(1+r²)−1。先对每条轨迹的全部已选观测取平均，再对每个高度区域平均，最后对三个区域等权平均。总目标仅新增0.3倍该项，100至300步线性启用。
该约束没有把衣物点当成皮肤或关节目标。裸人体应在衣物之后，但衣物后方可能存在许多不同体形和姿态，因此它不能独立确定皮肤深度。没有网格交点时新增项为零，同时必须记录丢失交点和网格缩小，防止仅靠缩小身体获得更低损失。
每10步在当前网格上用CPU float64选取精确双面最近三角形。屏幕重心通过倒数深度还原透视交点。每一步用该三角形的可微射线-平面交点计算深度，并验证三维重心仍在三角形内。初始化CPU与GPU深度最大差小于2.4×10⁻⁶世界单位，绝对参考几何的梯度有限且非零。结束后重新遍历当前所有最近交点评测，不使用训练过程中缓存的三角形。
两组均从官方GVHMR原始逐帧预测、既有原图检测、相机、许可SMPL-X和原始HaMeR手部观测重新生成初始化。原来的关节、单向衣物轮廓、姿态与体形先验、地面/足部稳定、鞋面相对运动和时间平滑目标保持一致。300步前只配准整体，300到700步优化共享姿态与身份，700到1200步放开受限逐帧姿态。两组各1200步，优化器全新，无高斯更新。
未经本项拟合的1062次躯干腿部观测，深度冲突由478降至282，网格交点由911减至900。两组共同命中的897次观测上，冲突477降至282；14次原交点消失，3次出现新交点，所以改善不能完全由隐藏掉评价射线解释。23帧网格体积下降1.946%，所有图像的可见关节残差中位数8.713增至9.908像素。左右膝模型弯曲由12.38/6.89度变为13.04/8.91度。这些是模型和观测拟合诊断，没有三维关节真值。
已逐张检查0/11/23/33/45/58原图网格叠加，以及同一23帧、固定世界相机下的正面与两个侧面。外表面冲突减少，但侧面屈膝、上臂摆放和躯干形状没有清晰突破，不能升级为高斯训练底模。相机方向依据原始地面点估计，未声称独立IMU重力。深度约束只是部分可观测几何信息，继续增加权重无法凭空提供衣物下的真实解。
下一步停止此单向权重扫描，检查独立先验与真实标定相机的三维配准及不可辨识参数。SAM3DBody仍待合法授权权重，授权后优先执行已准备的六张原图推理。未通过人体姿态和几何检查前，不开始20万点、18000步新高斯训练，也不替换查看器或部署00。
方法属于本项目的受控诊断扩展，不冒充任何官方论文完整复现。代码：GVHMR-shuolee/shuolee/absolute_surface_constraint.py、refine_absolute_surface_fresh.py。原始证据：${phase}/review/comparison.json、phase-conclusion.json，以及相邻smplx-absolute-surface-depth-031/depth-audit.json。图为本地真实源图及网格渲染，无生成补图。

十三、目前解决的问题与不足`;
notes=notes.replace('十二、目前解决的问题与不足',section);
slide.speakerNotes.text=notes;
const draft=build+'/surface032-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_外表面深度_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface032-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renders=build+'/surface032-final-renders';await fs.mkdir(renders,{recursive:true});
const rows=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(renders+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
 rows.push({slide:i+1,source_render_sha256:oldRenders[i],final_render_sha256:digest(bytes),unchanged:oldRenders[i]===digest(bytes)});
}
if(rows.slice(0,54).some(x=>!x.unchanged))throw Error('Unrelated slide changed');
await fs.writeFile(build+'/surface032-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[55],all_slides_rendered:55,unchanged_first54_render_hashes:54,
 renders:rows,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slide:55,unchanged_first54:54}));
