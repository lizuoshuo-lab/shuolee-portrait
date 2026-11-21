import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/native-pixel-center-audit-091',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='5cf848064552c4b66ecc646f57a7d643429d3d00956c2f2f66bd07c9ca0cf842';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),r=await read(run+'/remedy-result.json'),prior=await read(build+'/human090-deck-receipt.json');
if(!r.passed)throw Error('Correction checks failed');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51],old=slide.elements.find(e=>e.text?.startsWith('观测替换与相机扩展均未消除鼻口错位。'));
if(!old)throw Error('Expected conclusion missing');
p.resolve(old.id).text.replace(old.text,'已定位半像素不一致，修正验证通过。\n鼻口重影仍未解决，需继续查明主因。');
p.resolve(slide.id).speakerNotes.text=[
 '最新结论：'+d.summary,d.native,d.defect,d.correction,d.subject,d.limits,
 '本页图像继续使用083原生高斯的084冻结贡献分析。左侧原视频100实拍，右侧相同相机下的渲染及实际高斯贡献。颜色只作归因，不表示新的训练结果。没有修改照片，也没有新增理论示意图。',
 '坐标定义：COLMAP采用像素角点坐标，第一像素中心为(0.5,0.5)；图像数组索引采用整数像素中心。当前原始K在缩放后按GPU RGB输出为u=fₓX/Z+cₓ−1/2，v=fᵧY/Z+cᵧ−1/2。因此RGB样本(u,v)对应的射线是r=((u+1/2−cₓ)/fₓ,(v+1/2−cᵧ)/fᵧ,1)ᵀ。旧几何分支少了两个1/2，实际CUDA测量而非仅解析式对照确认了偏差。',
 '平面深度：PGSR累积法向图n和距离图d，沿样本射线得到Z=d/[−nᵀr]。本轮保留原生RGB、透明度、法向和距离图，仅在可选适配器内换用一致的射线重新计算深度。几何回投影、NCC单应性矩阵和深度法向继续使用同一几何K。没有更换高斯、相机外参或物体尺度。当前仅验证NCC scale=1，不能外推其他下采样约定。',
 '为什么需要实际GPU检验：若两个CPU函数复用相同错误约定，它们也可能彼此吻合。091使用独立世界空间测试高斯，经过正在训练环境使用的已编译PGSR CUDA扩展后测量图像质心；同时比较倾斜平面的解析深度与真实深度输出，再用原生回投影函数送回RGB。175个相机的9个固定位置全部可见，没有排除失败视角。',
 '渐变验证：在一个固定倾斜平面测试点上，对世界位置沿相机深度方向求自动梯度，并以0.001世界单位的中心差分核验。自动值0.97317，差分值0.97346，相对差约0.030%，梯度有限且非零。这只核验修正的可微路径，不代表实际人物训练已经改善。',
 '冻结人像影响量：采用6个预定源相机及全部16个现有训练邻居，像素步长8，以原始人物mask与固定模型透明度筛选实际源支持，对两个单应性投影均保留相同有限、正深度、图内查询。统计250800个有效采样，上方四分之一区域46944个。区域按mask外接框定义，不是解剖人脸标注。采样位移中位0.0887训练像素，p90=0.2585，最大2.749；深度相对变化中位0.0359%。没有测量人物各项损失梯度及整个训练过程的累积变化，不能归因为主因。',
 '此前竞争假设：088不依赖相机的原图匹配仍显示语义检测位置与图像纹理的分歧。官方FAN在175帧中152帧有效，唇缘共同对应变化约2.49至2.44原图像素，覆盖与改善均未过门槛。089增加逐帧焦距也未改善411个固定下脸查询。090在SIMPLE_RADIAL和OPENCV模型下重新三角化并联合优化共享内参、外参与结构，原生BA分别31/42次收敛；独立下脸中位误差均约3.67原图像素，仍未采用。不同试验的原图像素和本轮1536高训练像素不可直接混比。',
 '静态人像当前仍为083：重新拟合人体与中性FLAME、20万点新初始化、18000步后300220点。中央脸部1772点仅在初始化阶段投到单层表面，训练中全部点自由，未采用完整GaussianAvatars网格绑定。位置、完整协方差与全部SH已按+Y标准PLY导出；格式通过而鼻口几何未通过。本轮不重新导出同一文件冒充新结果，不部署00。',
 '后续有界工作：'+d.next_bounded_work.scope,d.next_bounded_work.gate,d.next_bounded_work.stop,
 '参考源码：PGSR https://github.com/zju3dv/PGSR 的cuda_rasterizer/forward.cu、scene/cameras.py、scene/gaussian_model.py；本地适配器shuolee_static.py及独立修正shuolee_pixel_center.py。原生扩展SHA、源码快照、协议与所有数值证据保存在'+run+'。',
 '当前默认训练入口、相机与083高斯未被修改。只有可选适配器的内部一致性已通过，人物根因与质量仍未通过。'
].join('\n\n');
const candidatePath=build+'/human091-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_像素中心核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human091-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human091-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human091-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
