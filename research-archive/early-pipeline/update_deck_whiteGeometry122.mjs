import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../clip03-fresh-geometry/face_feet_evidence.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/whiteGrid122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/whiteGeometry122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('均衡特征分配后的固定物体对应','固定背景相机下的面部错位');
replace('三排为屋顶拟合族','三排为面部原图、面部投影与鞋面投影；三列为200、232、280帧。');
replace('屋顶循环轨迹从','背景留出中位数0.33px；面部留出中位数3.89px。');
replace('非地面固定物体留出族','面部最大13.05px，未通过逐点1px要求，暂不训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'原生面部及鞋面留出投影，蓝圈观测，粉色十字预测',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height)/2,top:frame.top,width:frame.height,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N12e/f/g。继续固定第一组YAML4000×3000参考K，按原N4半像素缩放和理想居中裁剪到1520×2704。没有换4096、重估K或二次去畸变。YAML尺寸证据与真实输出K的独立确认保持区分。", "N12e只测定位：原83行的前三个观测，41×41原生局部窗口，双向仿射配准及三视角闭环，共498方向，477测量、10边界不足、8不收敛、3超出预定范围。面部SIFT和局部仿射位置最多相差3.254px，这是估计器差异而非已知真值误差。玻璃疑点也有小闭环，不能因此升级为固定物理点。结果未用于替换本轮几何观测。", "N12f对既有全部31个屋顶族、全部观测的原生128裁片和256上下文完成审阅。3组轮廓或玻璃支持仍不明确，保留证据但不作为固定点。合并此前其他物体/地面后40FIT与15heldout，其中非地面25/6。没有重新提特征、换分组或按残差删点。此前2个非地面留出仅指有限已审阅样本，不是整份数据上限。", "N12g第一条白T固定K相机候选：以满足预登记支持的最早邻对216/232从头PoseLib相对位姿，依次PnP注册200、184、248、264、280，原始SIFT观测与40背景FIT族进入固定K共享三维优化，未加载旧优化相机或几何。40族均有正深度和预定视差，无结构排除。7帧为464帧的诊断子集。", "独立目标排除三角化统计：背景留出86个目标，中位0.331884px，最大1.544858px，81/86严格小于1。非地面24目标中位0.247845，最大1.046321。面部heldout2族13目标，中位3.891467px，最大13.051620px，2/13小于1。全部6面部族41目标中位4.609864，最大13.325810，2/41小于1。鞋袜7族47目标中位0.395835，最大2.827080，36/47小于1。", "背景优化前后中位0.337299降到0.331884，但面部heldout中位3.584912升到3.891467。没有相机候选择优或更改内参。背景低中位数不能说明全部像素或人脸已对齐。", "实际审阅全部41面部、47鞋袜目标的原图与投影。眉眼存在开合，晚期面部投影出现共同竖直偏移，鞋面多数贴合。当前没有唯一证据区分人物微动、测量偏差或相机在人体深度的弱约束。背景早期184/200无非地面留出，屋顶主要集中一栋远楼。206人物诊断目标最大168.43px异常全部保留，未宣称整体正确。", "图中三列200、232、280帧；第一排face row59/family800 heldout原图，第二排同族预测叠加，第三排feet row77/family1454 heldout预测叠加。蓝圈是原始SIFT像素，粉十字是未使用目标观测三角化的3D预测。每格128×128原生像素，无二维形变或图中文字。", "下一步只读分析背景FIT相机弱约束是否足以解释脸部共同残差，再判断低自由度头部微动诊断可辨识性。冻结本轮，不扫K/参数，不再次提特征或重跑相机。面部逐点1px未通过，未训练或交付新人物点云。119历史模型保持。", "来源：robust-portrait-122/video-update-20261027/clip03-localization/result.json与visual-review.json；clip03-roof-review/visual-review.json；clip03-fresh-geometry/result.json、validation.json、visual-review.json、figure-index.json；各自执行前plan。以下保留历史诊断以便溯源："].concat([previous]).join('\n\n');
const candidatePath=build+'/whiteGeometry122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122固定背景与面部误差.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/whiteGeometry122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_whiteGeometry122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/whiteGeometry122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/whiteGeometry122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/whiteGeometry122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/whiteGeometry122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
