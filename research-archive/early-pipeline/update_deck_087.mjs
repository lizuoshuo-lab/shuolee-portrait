import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/head-rigid-correspondence-diagnostic-087',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='20a893ec720cd6655c4bd38f8f824eaff2172a76df59c09a83b990f864368453';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),visual=await read(run+'/visual-review.json'),previous=await read(build+'/human086-deck-receipt.json');
if(!visual.actually_inspected)throw Error('Actual image review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51],old=slide.elements.find(e=>e.text?.startsWith('ARAP保持了网格有效性'));
if(!old)throw Error('Expected conclusion missing');
p.resolve(old.id).text.replace(old.text,'鼻口位姿补偿未通过独立纹理与背景检查。\n继续核验图像对应，当前高斯仍有重影。');
p.resolve(slide.id).speakerNotes.text=[
 '当前结论：'+d.summary,d.method,d.coordinate,d.named,d.texture,d.same_frame,d.limits,d.source_photos,
 '本页图像保留083原生高斯的084冻结贡献分析。左侧是第100帧实拍，右侧是同相机渲染及重影贡献。红色表示初始中央面区域相关点的空间贡献，青色为其他区域。图像不表示087的训练效果，也不能证明点的完整生成历史。',
 '087坐标约定：原始SfM点与人体训练世界之间只有一个全局相似变换。若训练世界位置等于尺度乘旋转后的SfM位置再加平移，则逆变换先减平移，再乘旋转转置并除尺度。逐帧外参同步满足该关系。原图采用SIMPLE_RADIAL标定，矫正图采用针孔模型。原生光栅化的半像素约定与OpenCV像素中心通过显式0.5换算。代数闭合不能代替标定精度验证。',
 '087支持与查询：原始175帧按每25帧中的连续5帧留出，共35帧。所有三维语义点和纹理点都不使用这35帧的观测。鼻口点编号30至35以及48至67按相邻两个点分成三组，每次只用另外两组拟合，保留一组验证。上脸只作查询。原始纹理轨迹使用图像空间聚类，同簇同帧只计一次。未按相机残差筛选好点。',
 '087求解：以共享头部三维中心为旋转中心，在每帧相机坐标中估计旋转和平移。初值为零，不继承旧拟合。旋转每分量限制在正负2度，平移每分量限制在头部深度的正负2%。Soft-L1尺度为2个原图像素，检测翻转分歧用于相对加权，参数有0.02的弱先验。每次最多80次函数评估。这个局部模型只是诊断，未写入正式相机或静态人体训练。',
 '087同帧反证：将头部中心周围的旋转与平移写成等价的相机坐标刚体变换，再作用于同帧独立背景和躯干三维点。使用严格相同帧与交叉验证分组的查询统计，避免比较不同帧的区域中位数。背景与躯干显著退化，因此拒绝把本轮鼻口补偿作为整图外参。该反证不排除其他标定误差。',
 '087可观测性：20帧可进行三组拟合，另15帧支持不足。原561个可见鼻口查询中389个进入本项检验，172个明确记录为未检验，不能宣称全35帧通过。60次拟合中52次触及参数边界，不能据此确认真实头部微动。独立下脸纹理中位数改善未达20%，较大误差增多。',
 '此前086方法：926个下半脸顶点使用ARAP，17条训练纹理轨迹提供三角面重心精确约束，边界与其他顶点固定。50次局部全局求解保持局部面有效，9条留出纹理轨迹误差10.20至6.16像素，但鼻口语义误差7.14至8.49像素，因此候选拒绝。Open3D实现参考：https://github.com/isl-org/Open3D/blob/v0.18.0/cpp/open3d/geometry/TriangleMeshDeformation.cpp 。',
 '083全身静态训练：初始20万点中1772个中央面点替换为从头配准的中性模板表面位置，其余头发、耳、颈与衣物保持该轮新初始化。18000步后300220点，导出格式验证通过，但鼻口重影和新视角破碎未通过质量验收。',
 '下一项核验：'+d.next_bounded_work.scope,d.next_bounded_work.acceptance,
 '本轮证据：'+run,'当前静态路线不引入动态网络。最新诊断087，最新高斯083。'
].join('\n\n');
const candidatePath=build+'/human087-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_局部位姿核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human087-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human087-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human087-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
