import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/residual-distortion-direction';
const figure=root+'/structure-comparison.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/nativeSupervision-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.actual_review_complete!==true||review.native_line_crops_reviewed!==13)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/residualLinesVerified-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('白衣视频的原生图与分割草稿','残余畸变补偿的独立检验');
replace('第0、232、463帧，每组左为原图、右为分割。','三帧建筑原生裁片：第0、232、463帧。');
replace('464帧图像与时间已固定，完整相机仍待注册。','6条拟合、5条留出；留出误差未下降。');
replace('边界与遮挡仍待验收。面部误差未改善，尚未训练。','候选未采用。对应错配仍在，尚未启动新训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第0、232、463帧原生建筑裁片，保持比例，图内无文字',fit:'contain'});
image.frame={left:75,top:190,width,height:width*1100/4560};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 D1残余畸变补偿与T6逐帧对应独立诊断。用户明确要求继续补偿方向；已实际执行新试验，不等同N6冬帽单径向参数结果。", "已按用户要求继续残余畸变方向，完成一次新增独立补偿试验。不是重复N6的冬帽视频单径向参数联合几何求解。", "本次固定白衣视频K，用静止建筑直边检验输出像素到理想射线的4参数残余映射，不使用人物点或外参求这4个参数，不改变原图。", "13条原生边缘先作材料审阅；A1/A2含管道和多个近邻边界，实际选中的硬直边不明确，求解前排除。剩余6条拟合、5条整建筑留出；所有原始测量保留。", "拟合组RMS由1.139降至1.114px，留出组由1.773升至1.783px。B1留出屋顶由0.493升至0.787px，超过预登记保护限；不因训练组变小就采用。", "k1达到预设0.05上限；全画面最大预测修正95.350px，超过32px保护限。只测到画面上方约y12至818，不能将向脚部的强外推当成实际测出的畸变。", "C1、D1、D2这些清晰楼体边缘，补偿前RMS约0.27、0.24、0.23px。带窗框、横向接缝或相邻边的测量可达1至3px；混合了边缘定位、物体本身直线假设与真实残余扭曲，不能全部归因于镜头。", "数值Jacobian秩4、条件数40.47仅说明当前数学目标有局部响应；独立留出、边缘物理真实性和跨画面覆盖未通过，不能据此宣布物理标定正确。", "T6还找到第264帧同一短裤family974落到不同印花的对应矛盾。即便镜头补偿正确，也不能把同一张图上两个不同材料点变成同一个点。", "未应用本次补偿参数、未重新拟合人物或训练高斯。13条原有面部留出最大13.04875764px保持原值，主查看器仍为119历史模型。", "模型为processed_output_inverse_brown4：观测原生像素经固定K归一化后，xu=x(1+k1r²+k2r⁴)+2p1xy+p2(r²+2x²)，yu=y(1+k1r²+k2r⁴)+p1(r²+2y²)+2p2xy。此为输出像素到理想射线的逆模型，不能当成原始FISHEYE_8，也不能当成OpenCV前向系数。未修改训练原图、旧K、外参或原生渲染器。", "系数[k1,k2,p1,p2]=[0.05,-0.0329409091,0.0162457713,0.0010419650]。拟合6条、留出5条。拟合组等线权RMS1.138690→1.113886px，留出1.772676→1.783267px。预登记边界、留出和全域位移保护均失败，因此保留诊断而不发布为补偿相机。", "解析系数Jacobian与独立差分最大2.03e-7，空间Jacobian差1.72e-10，已知参数前后投影回环误差0。这些是实现数值控制，不是实拍标定。首次测量JSON时间戳函数漏调用导致写盘失败，已保存原脚本和失败证据，窄修复后13张原生审阅图逐字节一致；没有更改观测配置或重新搜索参数。", "T6使用单种子相邻原生帧双向LK，全464帧/87材料族，第一次数值拒绝即停，不重检测或吸附旧SIFT锚点。保留27221次尝试及27123个数值通过槽位；每帧至少10场景候选只说明局部数值连续性。", "T6非种子原SIFT与逐帧跟踪的最大差：场景1.53665、面部2.31566、衣物168.79320、鞋袜6.29493px。这些是两方法差，不是已知真实点误差。实际13页87行、6面部锚点条、8最大分歧条均已审阅。264帧family974的中/右原生裁片处于不同短裤印花，不能由共同镜头扭曲解释为同一材料点。", "当前图只展示第0/232/463帧原生上部建筑裁片，未重采样训练图或伪造补偿后人像。画面高度2704，已测线的y范围约12至818，不能宣称脚部畸变已约束。", "下一步继续分离残余映射、边缘/材料身份、相机外参和人物微动。在新的跨画面独立支持建立前，不加阶数硬压误差、不重跑同一候选、不启新高斯训练。真实面部13条留出最大13.04875764px仍未达标。原119预览保留。", "证据：robust-portrait-122/residual-distortion-direction/{plan,measurements,line-review,result,review}.json；native-continuity/{result,review}.json及原生裁片。其他51页保持不变。以下保留此前阶段记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/residualLinesVerified-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122残余补偿_核验完成.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/residualLinesVerified-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_residualLines_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/residualLinesVerified-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/residualLinesVerified-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/residualLinesVerified-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/residualLinesVerified-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
