import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-convergence123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-joint-tracks123-deck-receipt.json'),review=await read(root+'/joint-track-candidate/convergence-repair/visual-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_native_crop_review||review.rows.length!==4)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('共享三维点后，相机仍未对齐','相机误差降低，人体对齐仍未通过');
replace('留出观测128/161未低于1px','留出中位3.36降到1.29px，93/161个观测仍不小于1px。');
replace('拟合中位0.78px','33个观测变差。头部原生裁片未改善匹配，尚未训练。');
replace('第617／644／671帧','第484／510／537帧：保留误差增大的材料对应。');
const figure=root+'/joint-track-candidate/convergence-repair/visual-review/heldout_834_part_1.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'三个384px原生裁片，中心是盆栽及灯箱／柜体附近的冻结查询。第537帧误差由2.83增至42.07px，精确身份尚未认证，保留反例和图边界填充。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29，相同K、初值、563FIT及161HELD观测的唯一数值修复。相机改善但SMPL-X对齐、训练准入和人像质量均未通过。", "首先以既有33相机162点可见图执行完整合成控制：原生观测噪声0.2px，旋转扰动1度，位置和点扰动0.02。解析Jacobian12列独立有限差分通过，175评估收敛，最终重投影均值0.1485px，最大0.4649px。合成通过仅验证实现。", "真实修复从同一原始初始化重启，不使用之前80次的优化相机。初始残差差异小于4.55e-13，12列实拍解析Jacobian核验通过。K、观察、权重、第一相机和尺度固定。解析导数和数值预算一起变化，不能冒充单因素物理实验。", "一次登记的2000评估／120秒预算，实际41.03秒。终止status0、successfalse、nfev2000、njev1951，cost73.51、optimality0.56445，仍未收敛。FIT中位0.3350px、最大4.7676px，全部正深度。不继续增加该轮预算。", "候选冻结后再评测同45条HELD的161次观察。用其余全部视角三角化预测留一视角。中位3.3604降到1.2913px，P90为3.5772px，最大414.5369降到42.0674px。68个严格小于1px，93个不通过。物理面部HELD仍0，33视角不能替代860帧注册。", "128个观测改善，33个变差。8个原通过点变为不通过，43个原失败点通过。全部反例保留，不按残差删点，不用HELD选参数、根或观察。", "相对旋转698至752从错误分支附近移到此前FIT三视图更支持的根附近，角差0.1267度。事后比较，未使用HELD选根，不等于物理真值或唯一解。", "本页为误差反增族834在484、510、537帧的无缩放384px裁片。第537帧2.83增至42.07px。盆栽和灯箱／柜体边缘可辨认，黑条是图外填充，另查192px精确中心仍不足以证明错接。另实际审阅77镜框、978冰箱和1126袋状物，共17个观察、7张图。", "额外完成一次冻结查询的原生头部裁片对照。原整图2704x1520压到798x448粗特征及1344x756细特征，头部粗特征约97至112乘130至159。新768整数裁片以448粗、768细匹配，局部上下文和有效分辨率共同改变。未检测新点或修改训练图。", "三对图、每帧600相同查询、六方向。六个有向三视图门槛计数都下降，示例644至671至698从62降23。实际查看预登记24组、120张192px原生裁片，有可辨认耳部／镜腿／胡须，也有背景和遮挡误配。少数新通过只代表自洽，仍未认证亚像素物理精度。该ROI方案不采用，无重跑或门槛扫描。", "旧灰衣SMPL-X参与面部点残差中位43.99px，粗身体门槛容许约50至53px，缺解剖可见性。帧、编号、独立投影数值差小于0.0004px。当前新视频10诊断不能直接证明旧灰衣唯一根因。", "后续仅登记原生纹理二维可测性及已知亚像素变换合成控制。实拍局部光度细化须单独登记，仍不拟合新相机或训练人体。证据保存在video10-global-initializer/joint-track-candidate/convergence-repair和video10-correspondence-contrast/head-roi-contrast。前51页不变。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10数值修复与局部匹配.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_video10_joint_tracks123.py',preserve);
execFileSync(py,[b+'/preserve_deck_video10_joint_tracks123.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:prior.final_sha256},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:prior.final_sha256,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
