import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='video10-global-ambiguity123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-global-preflight123-deck-receipt.json'),review=await read(root+'/visual-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_image_review||review.families.length!==7)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('相机初始化与重复纹理错配','相同5个点，算出3组不同相机');
replace('新审阅222组材料','三组解都使拟合误差接近零，旋转却相差约42°。');
replace('旧脚本仍先逐帧注册','三视角约束使该点极线误差由86.08降至1.22px，仍未过关。');
replace('第725／752／779帧','第698／725／752帧：同组地面留出点的原生裁片。');
const figure=root+'/visual-review/heldout_floor_1297_frames_698_725_752.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'三个原生384像素裁片，每格中心为冻结的地面HELD查询坐标。只作材料检查，模糊及重复纹理未达到亚像素认证。图内无新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29 相机多解与SMPL-X人物对齐。新视频10的局部相机诊断，不等同于旧灰T/全部人物唯一根因，不表示产生了合格新相机、人体或高斯。", "在冻结162个FIT、45个HELD粗材料代表，33个原生诊断视角和同一条件K上，一次58对相对姿态诊断。没有新提特征、没有优化K、没有使用旧相机或几何初始化。HELD从未进入估计或根的排序。", "57对返回可用相对姿态，其中16条边达到预登记局部支持/视差保护，可靠图分成21组。41对少于12FIT，另有一对返回0内点/0平移失败值，3对有平面替代解释。拒绝原因可能重叠。没有通过降低12点门槛发布全局相机。", "32个三视角旋转闭环，中位0.689度，P90为31.406度，最大67.396度；18个超过预登记0.5度筛查标记。该闭环不是独立真值，不能代替材料身份和面部像素误差。", "第698/752帧只有5FIT，RANSAC给出几乎零Sampson距离的解，但HELD244/1297分别1.358/86.079px。这个事后案例只用于解释，不作为独立性能测试，不按误差删点。", "对完全相同5点和K执行一次确定性PoseLib五点根枚举，得到3个全正深度解，最大FIT距离均低于1.3e-8px。相对原选解的旋转差约38.54度、41.60度、0度。说明两视图最小拟合和正深度都不能唯一确定相机。", "只按既有FIT的698→725与725→752旋转乘积比较，三根闭环差为8.465度、0.270度、41.630度，较一致根的两HELD误差为0.237/1.223px。排序完全不看HELD。仍有10<12的支持边，局部案例且未完全<1px，未获得全局或训练准入。", "本页照片为HELD1297在698、725、752帧的384px原生整数裁片，未缩放和二维扭曲。三帧均实际审阅，中心仍在地面局部，第二第三张左侧有人体。原图片无新增文字。所有5FIT和2HELD的原生两视角裁片均实际查看，玻璃高光、帽子软材料、模糊和重复瓷砖保持不确定，没有按残差改分类。", "4项合成全局初始化控制通过：旋转平均、全局位置/点恢复、纯旋转拒绝、断图拒绝。最终状态序列化pid重复出错，已从保存结果恢复，不重复求解。真实两视图接口的只读平移/零平移处理错误也已修正，保留原失败记录；数据/阈值/seed未改变。", "固定K=fx1120.8319106618264,fy1119.820634890859,cx1352.8626255714278,cy763.7770175739016。来源第一YAML4000×3000，经0.676缩放、y254裁剪、半像素约定。已去畸变MP4不二次去畸变，D=0仅是当前条件基线；实际导出K未独立认证。", "下一步检查多视图共享轨迹对分支与尺度的约束，保持冻结材料、固定K和独立保护。当前0个完整新相机、0个新SMPL-X拟合、0步新高斯训练。物理可信面部HELD仍为0，860帧尚未完整注册。不能把极线距离当三维或面部重投影。", "证据：C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer/decision.json；relative-graph/repair-invalid-pose/result.json；minimal-root-diagnosis/result.json和cycle-result.json。历史第52页的全部材料分析仍保存在工作日志，其他51页不变。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123视频10相机多解核验.pptx';
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
