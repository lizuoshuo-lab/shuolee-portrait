import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/surface123-head-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-temple-validation-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-observation-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-observation-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-observation-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('局部姿态能减错，相机仍不能替换','面部关键点精度与可见性不足');
replace('灰衣：皮肤留出中位18.83→6.93像素，33/35仍超1像素。','网络中人脸约30像素高，鼻点约每1.41原图像素一档。');
replace('局部调整若套给场景相机，路面中位误差0.69→326.90像素。','第二皮肤标记9/41帧定位失败，停止新几何拟合。相机误差仍未排除。');
replace('青圈：皮肤观测；橙：静态；绿：头部逐帧。仍未达到亚像素验收。','74/76帧原图与鼻点。橙点在mask外，青点在mask内。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'原帧74、76的鼻部原生160px裁片，每组左原图右保存的检测点，图内无文字，无形变',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["SMPL-X仍未对齐，本轮确认拟合观测本身存在精度与可见性缺口。实际RTMW把全身输入缩为288×384，41帧脸部ROI在网络中位只有29.75像素高。argmax解码加翻转平均后，369个鼻部点全部位于约1.41原生像素间隔的解析格点上，不能当作连续的亚像素材料测量。另有12个鼻部观测位于原人物mask外0.786–3.660像素，仍因4像素余量获得正权重。原生图审阅也发现侧面点包含轮廓或遮挡推断，mask不等于解剖真值。量化和错误约束能引入跳动，但不能单独解释此前18.83像素的静态皮肤残差。新增第二处颞部色素斑的光流往返最大仅0.200像素，但40–48帧光度定位碰到预登记边界，早段材料中心也不够明确，因此停止在测量阶段，没有运行新的三维拟合。相机残差、模板形状、真实运动与对应误差尚未唯一分离。历史模型保留，灰衣高斯未启动。", "只读检查41个已存帧和全部369鼻部观测，未重跑检测或修改点权重。RTMW输入288×384，人物bbox加1.25padding并校正长宽比。模型SimCC argmax除2，原向和左右翻转坐标取平均。由真实bbox和解码解析得到的原生格点间隔中位1.412193726px，最大1.438488865px；369点距各自解析格点最大3.22e-13px。没有按观测拟合格点原点。", "单次解码的约2.82px间隔在翻转平均后变成约1.41px。浮点坐标不等于物理亚像素准确度。这能解释跳动的一部分，但不能证明静态皮肤18.83px误差全部来自量化。也不能将1.41px格点直接当作误差下界。", "脸部ROI在全身检测输入中位29.75px高。全部鼻点权重均为正，其中12点位于原SAM2 mask外0.786–3.660px，仍被4px边界余量保留。权重范围0.748–0.979。mask本身不是解剖真值，不能凭mask或残差直接删除远侧点。", "实际审阅全部41帧鼻部原图/点对照和41帧头部原生裁片。保守人工可见区域标签252可见鼻区、27远侧不确定、90边界或推断。这不是材料身份或解剖真值，不用于重加权。", "一次预登记第二材料M1测量：frame54颞部色素斑种子[792.5,965.5]，41原生帧，LK往返阈值0.5px，实际最大0.20004px。固定13×13负高斯+平面背景局部模型，40–48帧共9帧触及参数边界。实际复看原图后早段色素中心与相邻眉纹分离仍不清楚。不能用往返误差认证材料位置。", "严格执行预登记停止条件：没有把失败帧删掉，没有扩宽中心/宽度参数，没有缩短序列，没有运行新三维点或相机/姿态拟合。全部41测量保留。此前基于41头图的材料库存只表示可测候选，不能解读为全部41帧身份/中心已确认。", "此前头部代理逐帧调整在独立皮肤验证中仍33/35>=1px；不可用它替换整场景外参。当前独立面部与全身保护未通过，不启动灰衣高斯。历史白T/戴帽研究模型保持quality=false。", "先核查现有RTMW的原始SimCC响应是否可从当前模型独立取得，并核对官方连续峰位解码及坐标变换；将解码量化和遮挡权重分成两个修复变量。先登记并验证已知峰位/像素变换的有界工程控制，再决定少量原生帧实测。不把连续坐标当物理真值，不修改原观测或相机，不复跑已失败M1、逐帧姿态或训练。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\observation-quality-conclusion.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-observation-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123观测精度核查.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-observation-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_observation.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-observation-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-observation-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-observation-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-observation-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
