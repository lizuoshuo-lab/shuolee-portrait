import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/barycentric-arap-face-086',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='3a440f657d3500b9a9298c0771c0647b01525ce7297e31292099b3085a2e85ff';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),visual=await read(run+'/visual-review.json'),previous=await read(build+'/human085-deck-receipt.json');
if(!visual.actually_inspected)throw Error('Actual image review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51],old=slide.elements[5];
if(!old.text.startsWith('逐顶点校正未保持'))throw Error('Expected conclusion missing');
p.resolve(old.id).text.replace(old.text,'ARAP保持了网格有效性，但鼻口语义投影仍偏。\n纹理误差下降未能通过对齐验收，当前高斯仍有重影。');
const sub=slide.elements.find(e=>e.text?.startsWith('整图相机与人体初始化已接通'));
if(!sub)throw Error('Expected subtitle missing');
p.resolve(sub.id).text.replace(sub.text,'整图相机与人体初始化已接通，当前高斯仍有鼻口重影。');
p.resolve(slide.id).speakerNotes.text=[
 '086最新结论：'+d.summary,d.method,d.algebra,d.measured,d.anatomical,d.source_photos,d.limitation,
 'ARAP技术路线：用官方FLAME中性模型从头拟合共享Sim3，再固定175个整图相机。17条训练纹理轨迹只用非留出帧与非中间三分之一观测进行soft-L1三角化。三维目标投到新中性网格的最近三角面后保持重心系数，图像点不会吸附到最近顶点。926个下半脸顶点可变，边界、头皮、耳、颈与其他顶点固定。',
 '局部全局求解：参考Open3D0.18.0 Spokes ARAP，以每个顶点邻边的局部刚体旋转保持结构。余切权重取相邻对顶角平均并截到非负。局部步骤用SVD求正规旋转，全局步骤用稀疏KKT同时求网格位置与精确线性重心约束。50次固定迭代，未扫描权重或步数。源码：https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/geometry/TriangleMeshDeformation.cpp 。',
 '保留的验证：9条完整留出纹理轨迹、35个完整留出帧、每条轨迹中间三分之一。所有118次留出轨迹射线查询都保留。上脸、边界和颈部保护不变。局部面法向与初始面法向点积必须为正，面积比至少1e-4，最大偏移不超过0.03模板米。0个反向面与面积比0.68264只证明通过这组局部检查，不证明全局无自相交或真实人体几何。',
 '这轮以精确纹理约束替换了085的直接顶点Adam与语义关键点目标。因此是求解器及约束形式的组合对照，不能把变化单独归因于ARAP正则化。语义鼻口验证退化，候选拒绝，没有初始化或训练新的高斯。',
 '官方嵌入核验：FLAME重心系数以float32存储，行和最大误差2.98e-8；原值未重新归一化。与官方前向的关键点位置最大差1.39e-17模板米。这一数值一致性不是人物对齐。图像语义检测也不是3D真值，66号内唇点没有足够可见观测，未生成假目标。',
 d.next_bounded_work.scope,d.next_bounded_work.distinction,
 '本页图片仍为083原生渲染的084冻结贡献分析，明确不作086训练效果图。左为第100帧实拍，右为同相机高斯重影贡献点。红色由中性模板中央面区域的高斯贡献，青色为其他区域；属于空间贡献划分，不是完整生成历史或根因证明。',
 '083初始20万点中仅1772个中央面点替换为重新配准中性表面的位置，头发、耳、颈、衣物保持原初始化。18,000步后300220点，原生重载和独立PLY格式检查通过，但鼻口重影与新视角破碎未通过质量验收。',
 '本轮证据：'+run,'相机、旧高斯和查看器PLY未修改；最新诊断086，最新高斯083，未部署电脑00。'
].join('\n\n');
const candidatePath=build+'/human086-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_对应与对齐核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human086-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human086-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human086-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
