import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-identity-prior-fresh-033';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='f30ff3be0fafbf346885cddc332264f13e7e7b88a87f966a9b2ee27d866e3d44';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical deck changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='identity_controls_reviewed_geometry_not_accepted')throw Error('Missing reviewed evidence');
const prior=JSON.parse(await fs.readFile(build+'/surface032-deck-receipt.json','utf8'));
if(prior.final_sha256!==expected)throw Error('Missing source render provenance');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Unexpected slide count');
const snap=await p.inspect({select:'$.slides[54]',include:['id','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const sourcePreview=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(sourcePreview)!==prior.renders[54].final_render_sha256)throw Error('Source preview drift');
const anchor='十三、目前解决的问题与不足';
if(row.notes.text.split(anchor).length!==2)throw Error('Missing notes insertion anchor');
const added=`十三、体形约束与局部姿态的受控对照
体形参数β和局部关节姿态θ同时影响投影。固定相机的二维目标可以通过改变体形、改变姿态和调整身体位置共同减小残差，较低图像误差不能单独证明体形或骨架正确。
原023体形项为L增量=Σ(βk−β0,k)²/10，权重0.04，其中β0是官方GVHMR的序列共享预测，系数没有单位。它约束对预测的修改幅度。VHAP_smplx参考实现另有L体形=Σβk²/10的零中心先验，代码中标准差取1。这里仅参考这一损失定义，不把适配诊断称为完整官方VHAP流程。
033重新生成两组初始化。固定体形组保持β=β0，不优化身份参数，其余姿态和根变换仍开放。零中心组将原体形项替换为L体形，保留0.04权重和β0±1的原有逐系数边界。两组均使用相同原始59帧、标定相机、GVHMR原始姿态、HaMeR局部手部观测和许可中性模板。每组1200次更新，优化器全新，没有加载旧个体拟合状态作起点。
其余身体关键点、足部关键点、手部关键点、单向衣物轮廓、姿态增量、局部关节先验、脚底接触、对应脚底点稳定、鞋面纹理相对运动和时间平滑项沿用023。033没有加入032的深度损失，以免混淆干预。原023仅作只读对照，已在032精确重放；本轮训练源文件散列不变，两个新组的初始化数组也与原对照完全相同。
原拟合、固定预测体形、零中心体形先验的可见二维残差中位数分别为8.7126、10.0578、8.7604像素。体形系数均方根分别1.0574、0.4632、1.0411。23帧左膝模型弯曲分别12.38、13.38、12.41度，右膝分别6.89、8.64、6.91度。这些数值是拟合诊断，没有独立关节真值。原始衣物深度只用于事后检查，不参与033优化。
实际六视图叠加及同一23帧固定世界正面、左右侧面显示，固定体形改变了躯干比例，但没有消除侧面姿态和上臂位置问题。零中心组与原拟合接近。两组都不升级为高斯训练底模。这个结果只排除了本次两种受控改动带来整体改善，不能证明初始体形准确，也不能证明所有体形先验无效。裸身网格本来也不必填满宽松衣物。
下一次独立图像预测将核验NLF-L。该方法先预测有对应关系的三维人体点，再拟合参数化人体，提供真实相机内参接口和SMPL-X输出。它的坐标单位和输出映射仍需本地核验，下载或模型加载不等于人物几何通过。此处不报告未经运行的本素材效果。NLF官方权重只用于非商业研究，本轮不部署公司电脑。SAM3DBody仍待合法授权权重。
参考代码：https://github.com/jmaraval/VHAP_smplx/blob/8980d4948d91b05ff96ae91bae29e3f15a238f5e/vhap/model/tracker_smplx.py 。NLF官方源码与发布：https://github.com/isarandi/nlf 及 https://github.com/isarandi/nlf/releases/tag/v0.3.2 。本轮证据：${phase}/phase-conclusion.json 和 review/comparison.json。实际图像：review/six-source-comparison.jpg、review/fixed-world-comparison.jpg。

十四、目前解决的问题与不足`;
slide.speakerNotes.text=row.notes.text.replace(anchor,added);
const draft=build+'/identity033-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_体形先验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/identity033-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renders=build+'/identity033-final-renders';await fs.mkdir(renders,{recursive:true});
const verification=[];
for(let i=0;i<55;i++){
 const b=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(renders+'/'+String(i+1).padStart(2,'0')+'.png',b);
 const h=digest(b);if(h!==prior.renders[i].final_render_sha256)throw Error('Unexpected visible change '+(i+1));
 verification.push({slide:i+1,render_sha256:h,unchanged:true});
}
const after=await q.inspect({select:'$.slides[54]',include:['notes.text']});
if(!after.records[0].notes.text.includes('十三、体形约束与局部姿态的受控对照'))throw Error('Notes were not preserved');
await fs.writeFile(build+'/identity033-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_notes_slides:[55],visible_slides_changed:[],all_slides_rendered:55,
 renders:verification,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,notes_updated_slide:55,all55_visible_renders_unchanged:true}));
