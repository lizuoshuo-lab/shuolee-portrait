import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/observed-static-face-surface-085',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='012a6f0f6fafc72f2e15d20982ec5ec9574d597306ee0b42922f8c2df5e8f75c';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),visual=await read(run+'/visual-review.json'),previous=await read(build+'/human084-deck-receipt.json');
if(!visual.actually_inspected)throw Error('Actual image review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51],old=slide.elements[5];
if(!old.text.startsWith('中性模板跨视图误差'))throw Error('Expected conclusion missing');
p.resolve(old.id).text.replace(old.text,'逐顶点校正未保持鼻口细小三角面的有效性。\n先检验局部结构约束，再进入训练；当前质量仍未通过。');
const note=p.resolve(slide.id).speakerNotes;
note.text=['085最新结论：'+d.summary,d.observations,d.measured,d.geometry,d.limitation,
 '本轮从许可中性模板和原始观测重新拟合统一Sim3。926个顶点的共享静态位移从零开始，相机、表情和逐帧姿态不参与更新。17条纹理轨迹用于拟合，9条完全留出。另有35个完整帧与各轨迹中间三分之一留出。固定模板中的三角面及重心坐标将纹理轨迹映射到表面，拟合时不重新关联。',
 '目标由等轨迹权重的2像素soft-L1纹理项、0.25倍可见鼻口关键点项、0.2倍位移拉普拉斯项、0.01倍位移幅度项和0.02倍相对边长项组成。最多1200步，但第4步已有4个面当前法向与初始法向点积非正，按预定规则立即停止。没有放松门槛、删除问题面或补跑。',
 '留出数字是第4步被拒绝候选的诊断，不是收敛后的方法成绩。全部118次留出轨迹转移均保留，按每轨迹中位数等权汇总为10.204929至10.094452像素。35留出帧中561个可见鼻口关键点为7.137711至6.637777像素。几何与20%改善门槛均未通过，不产生新高斯PLY。',
 '下一步参考Open3D的ARAP局部边向量保真。它优化各邻接边相对局部旋转后的变化，可指定约束顶点与目标位置。先核验约束映射和无位移恢复，再冻结新的从零拟合协议。ARAP仍不保证无翻折或正确几何，必须保留相同门槛。https://www.open3d.org/docs/release/tutorial/geometry/mesh_deformation.html',
 '开源实现核查：GaussianAvatars的scene/flame_gaussian_model.py中compute_laplacian_loss当前使用dynamic_offset，static_offset的优化器注册被注释。085是本地受控适配，不是官方完整训练复现。仅共享静态人物，衣服保持独立表面，眼镜专项未启用。',
 '本页照片仍为083原生渲染的084冻结归因，不能称为085效果图。最新高斯仍是083，质量未通过；相机估计误差、真实微动与观察误差仍未分离。证据目录：'+run,
 '\n084此前冻结归因记录（保留机制证据，执行状态以上述085为准）：\n'+(slide.notes?.text||'')].join('\n\n');
const candidatePath=build+'/human085-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_静态表面预检.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human085-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);if(i===51)await fs.writeFile(build+'/human085-final-52.png',bytes);else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));renders.push({slide:i+1,render_sha256:h,changed:i===51});}
await fs.writeFile(build+'/human085-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
