import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-joint-tracks123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-global-ambiguity123-deck-receipt.json'),review=await read(root+'/joint-track-candidate/visual-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_native_crop_review||review.rows.length!==4)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('相同5个点，算出3组不同相机','共享三维点后，相机仍未对齐');
replace('三组解都使拟合误差接近零','留出观测128/161未低于1px，当前相机不能用于人体拟合。');
replace('三视角约束使该点极线误差','拟合中位0.78px，留出中位3.36px。优化仍未收敛。');
replace('第698／725／752帧','第617／644／671帧：桌面材料仍需确认精确对应。');
const figure=root+'/joint-track-candidate/visual-review/heldout_1126_frames_617_644_671.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'三个384px原生裁片，中心是桌面袋状物附近的冻结材料查询位置；无缩放、扭曲或图内文字。袋状材料物理固定性与精确对应未认证。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29，本轮在固定条件K上，检查视频10的多视角共享XYZ联合求解。相机、SMPL-X和人像质量均未通过，不替换历史点云。", "材料图有33视角、162FIT轨迹、563FIT观测。保存的57条相对姿态边形成32个三角闭环，但107、510、537、564、590、617六帧没有这样的闭环。另有159组三视角共享材料组合未进入两视图姿态图，各1至4条共同轨迹。原每对12点保护未放宽。", "单次初始化复用本轮新估的57条相对旋转，不改变五点根。连接Laplacian求全局旋转，再用全部FIT世界射线共同求中心和共享XYZ。没有旧人物几何或旧优化相机初值，没有K优化、重新匹配或留出反馈。", "首先在已有合成几何和已知旋转上加0.2px噪声，seed12311，正深度位置求解最大残差0.6145px、均值0.1764px，控制通过。它仅验证已知旋转时的位置实现，不等于完整联合优化或实拍通过。", "实拍单次稀疏BA固定第一相机和一个尺度分量，soft_l1尺度1px，80次函数评估预算。全部563FIT保留，拟合中位5.6136降至0.7813px，最大仍107.63px，全部正深度。优化返回status0、successfalse、nfev80、optimality2613.52，未收敛。", "候选冻结后对45HELD轨迹的161次观测逐次排除一个视角，用其余全部观察三角化并预测排除视角。161个预测均正深度，33个误差严格小于1px、128个不小于1px，中位3.3604px、P90为32.7319px、最大414.5369px。这是原生欧氏重投影，非此前Sampson距离。没有按误差删点或挑视角。", "事后只读实际Jacobian为1126×677，固定7维相似变换自由度。12列中心有限差分通过，列归一化SVD秩677，条件数1498.21，局部切空间鲁棒梯度最大2872.03。实际模型未达驻点，满秩不证明真实材料、K、唯一解、精度或无静态解。", "本页图是误差最大材料族1126在617、644、671帧的原生384px裁片。中心在棕黄色袋状物附近，桌面固定不代表袋状材料一定静止或精确对应。另实际查看244通风口、696地砖地毯边缘、397木架金属器具共12个原生裁片。保留反光、模糊、软物体和重复纹理的不确定性，没有据残差重分类。", "灰衣SMPL-X的已存实拍诊断显示面部检测残差中位43.99px，部分鼻嘴投影在脸颊或耳前。帧/编号和独立投影数值差小于0.0004px；粗身体门槛容许约50至53px，缺解剖可见性。新视频10的相机失败不是旧灰衣唯一原因的证明。", "物理可信面部HELD仍0，860帧未完整注册。所有候选仅诊断，不准入身体拟合或高斯训练。下一步只验证解析Jacobian与完整合成联合恢复，不自动延长本轮80评估预算，也不把失败认定为数据无解。", "证据：robust-portrait-122/video10-global-initializer/joint-track-candidate/result.json、stationarity-audit/result.json、visual-review/review.json、decision.json。上一轮三根证据和全部原图仍保留在同一工作日志。前51页保持原样。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10共享点联合核验.pptx';
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
