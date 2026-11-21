import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-native-align-fresh-113',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='8edc9c36c9688d9dad5e9b8978b9f2004e1f305fdbbf876d2f915b621bf418c5';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const prior=await read(build+'/human113-deck-receipt.json'),d=await read(run+'/decision.json');
if(!d.actual_visual_review_completed||!d.slide_summary)throw Error('Actual review decision missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','layout','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slidecount');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/human113final-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('训练期对齐：同一特征共享空间位置',d.slide_title);
replace('共享三维特征点已接入训练，同步优化颜色、轮廓与位置',d.slide_summary);
replace('先逐层计算位置误差再聚合，避免前后高斯靠平均位置抵消误差。',d.slide_observation);
replace('18,000步训练进行中；最终以全身边界和连续新视角评估。',d.slide_conclusion);
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one evidenceimage');
const image=p.resolve(images[0].id);image.replace({blob:new Uint8Array(await fs.readFile(run+'/evaluation/face-000100.jpg')),contentType:'image/jpeg',fit:'contain',alt:'固定第100帧：左原照片、中098、右113。两组20万点从头训练18,000步，同相机同裁剪。'});
image.frame={left:65,top:177,width:1150,height:384};
p.resolve(slide.id).speakerNotes.text=[d.summary,d.quantitative_summary,d.visual_summary,d.limits,
 '关键方法：固定175个真实074相机、共享静态穿衣人物；每轮重新拟合先验与初始化20万点，不使用旧高斯或旧优化器。2,714个共享目标由原始二维对应重新估计，含50个命名面部特征。',
 '每个有限3D高斯的代表位置为校准视线上的条件密度最大点：z=d^T Sigma^-1(mu-o)/(d^T Sigma^-1 d)，Y=o+zd。原生RGB渲染贡献作为辅助几何权重，先逐贡献计算平方误差再聚合，避免前后重复层靠均值抵消。只为该几何辅助固定贡献与尺度；正常颜色、轮廓仍更新全部参数。',
 '初始化法向来自本轮175mask形成的穿衣表面及新拟合头部。初始尺度的法向标准差为切向0.25倍，最终点不被吸回裸人体模板。表面初始化和几何辅助共同变化，因此113只支持组合方案评价，不能拆解单因素因果。',
 '投影核验：175相机的实际观测及视域内测试，公式投影与源码矩阵误差均小于0.0001像素；三组旋转平移相机对同一世界点的检验通过。此前未限制视域的检查在图外点触发0.0032像素浮点差异，原失败记录保留。该测试不能证明估计的真实相机准确。面部逐轨迹重投影误差中位数整体约5个原图像素。',
 '固定相机与二维观测时，||Y-X||²分为目标点到视线的垂直距离平方以及沿视线的深度误差。当前辅助可以改变后者，不能使不一致的观测视线自动相交；相关分解单独留档。',
 '图源：本轮evaluation/face-000100.jpg。原照片、098、113共用相机和裁剪；是实拍训练视角，不能称独立新视角真值。另有固定侧视和150帧同路径连续比较，以及肩袖、躯干、腿脚检查。',
 '下一项：'+d.next_bounded_work.scope,
 '数值门槛：初始无限切平面方案的同批1,400观测只有89.6%有效，未达到预先90%要求。有限协方差方案在同批观测达98.6%，梯度与原RGB一致性检查通过；未放宽覆盖门槛。训练中每500步检查观测有效比例。',
 '交付：完整世界坐标标准3DGS以及+Y独立导入版本，同步转换完整协方差及所有球谐阶。格式兼容与外观质量分别记录；本轮没有在00部署。',
 '源码基线：https://github.com/zju3dv/PGSR 。113对齐是本地研究扩展，不声称上游官方实现。证据目录：'+run].join('\n\n');
const candidatePath=build+'/human113final-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_双人像路线.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human113pair-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human113final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slidechange '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human113final-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
