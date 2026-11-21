import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-supervision';
const figure=root+'/review-inputs/report-comparison.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/freshPrep-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.all_silhouettes_coarsely_reviewed!==464||review.actual_review_complete!==true)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeSupervision-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('遮挡观测与衣物表面初始化','白衣视频的原生图与分割草稿');
replace('合成轮廓（上）与20万点中心投影（下）。','第0、232、463帧，每组左为原图、右为分割。');
replace('6个前景、2个未知视角，支持率由75%改为100%。','464帧图像与时间已固定，完整相机仍待注册。');
replace('26项控制通过。实拍面部仍未对齐，训练未启动。','边界与遮挡仍待验收。面部误差未改善，尚未训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第0、232、463帧的原图与分割草稿；无图中文字，保留比例',fit:'contain'});
image.frame={left:75,top:190,width,height:width*541/1824};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 T5白衣视频完整原生帧和分割草稿。执行前登记全部464帧，不复用旧人物几何、点云或优化器。来源视频SHA256 0a4d31cfcf31036ef2244b2709daaf29b425c1921c5516183302670728d030de，显示原生1520×2704。", "逐帧ffprobe实读实际PTS，首帧0、末帧19.267744秒，帧间隔最小41.601、中位41.616、最大41.626毫秒。保持全段时序，不用24fps标签重建时间轴，不插入或删除帧。", "FFmpeg RGB48 PNG提取不旋转、不缩放、不再次去畸变。464张原生RGB逐一记录尺寸和SHA，7张与既有诊断PNG逐字节一致。PIL/模型读取使用8位RGB视图，PNG保留16位RGB存储；这属于解码颜色表示说明，不是几何重采样。", "人物分割复用本机固定SHA的YOLOv8n和SAM2.1tiny，无新增下载。跟踪主人物检测框身份，以框与中心提示分割，每帧保留所有检测框、选中身份、alpha、文件SHA和原始时间。检测/SAM内部缩放不等于亚像素边界定位，输出alpha为1520×2704草稿。", "实际审阅全部15页共464个轮廓缩略图，以及第0、116、232、348、463帧原图/分割和原生头脸、手部衣物、脚部裁片。粗轮廓检查不能替代全464帧逐像素边界验收。", "464个轮廓缩略图中未见整个人物丢失或切换为背景行人的粗错误。全部mask未触碰画框，5帧原生细节中脸、手、鞋的主体保留。此结论限于粗轮廓与已审阅裁片。", "第0帧和第348帧，手旁白T恤内部出现误删小洞。原图是连续布料，二值alpha却给出0。具体原生像素位置和颜色已保存，未自动填洞或改原图。", "第348帧发束与颈部之间仍残留绿色背景。头发细丝、衣袖/鞋边的混色也未完成逐像素验收；高SAM分数不能代表边界小于1像素。", "48帧检测到多个候选人物。连续框身份检查没有中断，5张完整原图中背景人物未并入主人物，但这不等于全段外部遮挡已经验证。", "相邻mask最低IoU为0.9264，发生在第429到430帧。这个数混合了相机运动、人物变化和分割变化，不能据此宣称真实人物静止或相机正确。", "整段视角从一侧经过正面到另一侧，没有完整背面观测。5张原图中的面部/肩部关系需要结合稳定场景与固定材料轨迹进一步检查；仅凭这些不同视角不能量化人物微动。", "未自动创建所有像素可观察的mask。外部遮挡、边缘混色、颜色有效区域和物理对应需要分开核验，当前alpha未作为通过验收的训练监督。", "完整464帧相机仍未实际注册。当前7个诊断相机不能插值补齐。K、外参、材料轨迹与留出划分均未改变，13条面部留出最大仍13.04875764原生像素，没有新增真实人物拟合或高斯训练。", "第一组YAML的4000×3000经下采样、居中裁剪仍为主要条件输出假设。尺寸字段本身不独立证实去畸变的K_rect，旧相机已使用4000基准；本轮完整解码不构成新的标定证明。", "图为实际第0、232、463帧，每组左原图、右分割，展示缩小但未拉伸，图内无文字。原生RGB/alpha存D:/GS-SLAM-Data/robust122-whiteT-native-supervision，D:/00仅同步审阅图和证据，不同步大体积中间帧。", "证据：robust-portrait-122/native-supervision/plan.json、frame-plan.json、frame-clock.json、decode-receipt.json、result.json、review.json、review-inputs/index.json。后续需处理审阅中具体监督缺陷，并在新的可辨识证据下登记相机诊断，不重复失败的七帧BA。以下保留同一报告此前阶段记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/nativeSupervision-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生监督.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeSupervision-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeSupervision_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeSupervision-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeSupervision-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeSupervision-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeSupervision-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
