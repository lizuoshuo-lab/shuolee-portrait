import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-global-preflight123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-remaining-material-review';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-remaining123-deck-receipt.json'),review=await read(root+'/visual-review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(review.reviewed_rows!==222||review.rows.length!==222)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('相机初始化与重复纹理错配','相机初始化与重复纹理错配');
replace('新审阅222组材料','新审阅222组材料；同一物体仍可能匹配到错误部位。');
replace('相同轨迹的合成连接结构可解','旧脚本仍先逐帧注册；全局初始化待补齐，实拍精度未过关。');
replace('第725／752／779帧','第725／752／779帧：飞镖盘裁片中心落在不同扇区。');
const figure=root+'/review/contexts/row_196.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'三个原生384像素裁片，每格中心为保存的查询位置，落到同一飞镖盘的不同扇区。图内无新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29 视频10材料与相机初始化核验。现有SMPL-X对齐问题尚未解决。本页不表示产生了合格新相机、人体或高斯模型。", "剩余222个代表族的696个原生观测、19页裁片和16组扩大上下文全部实际审阅。120个为可辨认粗固定局部，94个仍有歧义，6个非刚性材料静止性未认证，2个物理身份矛盾。原像素、FIT/HELD和旧轨迹不变。", "合并此前审阅共162个FIT、45个HELD粗背景族。其中非地面刚性候选91个FIT和29个HELD。每个33诊断视角均有至少8个粗FIT，但这不等于共享三维相机路径或亚像素真值。原视频860帧尚未完整注册。", "图为第196行、FIT族1338、第725/752/779帧；每格中心为保存的查询位置，分别落到同一飞镖盘的不同扇区。另一矛盾为第172行、HELD族1275。两者从后续使用中隔离，未按重投影误差删点，未把HELD转FIT。不能把同一物体当成同一物理点。图内没有新增文字和形变。", "沿用初始共享FIT>=12、扩展已有三维FIT>=8的顺序计数，已审阅轨迹最多覆盖6/33视角。即使把所有可用未认证FIT和同族分支理想化合并，共527族，上限也为14/33。分支并集不是合法物理轨迹，禁止直接用于真实相机求解。", "首版计数错误漏掉8个已人工从失败遮罩中恢复的背景族。修正为原候选和已审阅材料的并集，162/527均包含已审阅基线；6和14的计数未变。初始文件和纠错计划保留。", "进一步采用原162个FIT代表的确切可见性，33相机、563观测，生成一次已知非退化合成几何，不使用真实二维位置或旧优化相机。移除首相机6自由度及全局尺度后，Jacobian为1126x677、秩677、无额外局部连续歧义。12列中心有限差分和已知尺度规范通过，最小列归一化奇异值0.00573194。", "合成满秩不证明真实内参、视差、物理身份、动态场景、全局唯一解或面部精度。它说明不能把顺序注册失败等同于轨迹结构必然不可解，应单独核验全局共享轨迹初始化。旧顺序计数保护不被悄悄降低。", "独立且物理可信的面部HELD仍为0。实际相机、输出K、人物微动和材料对应不能唯一归因。本轮没有真实相机估计、SMPL-X重拟合或高斯训练，没有替换历史查看器模型。", "前置核验已经完成：207组粗材料、724次原生浮点观测与原始关键点逐项一致，162FIT/45HELD不变，无同一特征ID跨划分。第一K按4000x3000到2704x2028再裁剪y=254和半像素约定，精确复现既有横屏条件K。该代数一致性不证明物理导出标定。", "证据目录：C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-remaining-material-review。主要文件：decision.json、visual-review.json、ceiling-corrected-result.json、generic-observability/result.json。", "源码核验：旧global_fixedk121.py采用双视图起步、逐帧PnP，再对已注册视角联合BA，并非全局相机初始化。现有PoseLib2.0.5、SciPy1.14.1可用于构建新原型；PyCOLMAP3.10只暴露增量映射，已查PATH没有GLOMAP。未下载或安装。", "下一步：一次4项已登记合成控制，测试已知相对旋转的全局平均、共享轨迹的全局位置与结构、纯旋转退化拒绝、断图拒绝。已知相对旋转仅是组件测试，不是实拍姿态估计。只用固定K，120秒CPU预算，不运行实拍相机、人体或高斯训练。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10全局初始化核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_video10_global_preflight123.py',preserve);
execFileSync(py,[b+'/preserve_deck_video10_global_preflight123.py'],{stdio:'inherit'});
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
