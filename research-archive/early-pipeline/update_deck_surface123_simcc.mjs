import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/surface123-observation-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-simcc-decoder-review/actual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/surface123-simcc-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/surface123-simcc-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/surface123-simcc-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('面部关键点精度与可见性不足','连续解码未解决SMPL-X错位');
replace('网络中人脸约30像素高，鼻点约每1.41原图像素一档。','6帧鼻点位置变化中位0.84像素，最大1.82像素。');
replace('第二皮肤标记9/41帧定位失败，停止新几何拟合。相机误差仍未排除。','逆仿射误差小于0.0001像素，遮挡与材料对应仍需验证。');
replace('74/76帧原图与鼻点。橙点在mask外，青点在mask内。','第80帧：原图、原解码青点、连续解码橙点。原拟合保持。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第80帧三个原生160像素鼻部裁片，左原图、中原argmax青点、右连续解码橙点。没有图内文字，没有形变。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["SMPL-X仍未对齐。现有导出包明确将RTMW的连续SimCC细化关闭，已在保持原图、bbox、RGB归一化、模型、置信度、翻转和相机完全不变的条件下做6帧12次CPU推理。原argmax坐标与旧观测精确一致，独立逆仿射最大差0.0000953像素，未发现这一步坐标映射错误。连续细化改变54个鼻点位置的中位幅度0.843像素、最大1.816像素。实际原生图比较仍有侧脸轮廓/遮挡推断点，不能把连续浮点坐标当成亚像素材料真值。这仅确认解码量化的一项可控影响，没有证明真实定位误差降低，也没有解决此前数十像素的SMPL-X错位。相机、模板形状、跨帧运动和语义对应仍未唯一分离。未修改观测、相机或历史模型，未启动新拟合/训练。", "预登记6帧40、48、56、64、72、80，每帧原向与水平翻转各推理一次，共12次CPU推理，保存全部原始133点SimCC响应。没有改变RGB归一化、bbox、模型、相机、得分合并或原观测。仅对已有响应作连续峰位解码。", "旧argmax结果与已存观测逐点完全一致，置信度差0。独立cv2逆仿射回原图最大差9.52747e-5px。这个数是实现一致性，不是真实相机或材料定位精度。", "54个鼻点坐标变化中位0.842562px、P90为1.239838px、最大1.815771px。这是两种解码的差值，不是定位误差降低。全部鼻点的数值稳定检查通过，仍不能据此确定遮挡或物理身份。", "模型导出包pipeline.json解码参数sigma6/6.93、split2、use_dark=false；可选细化按官方MMPose的SimCC DARK公式，使用固定37/43高斯核。没有把官方默认关闭解释成已证实的配置缺陷。", "独立已知连续高斯峰控制最大误差2.24e-5bin，零、平坦、边界、相隔多峰和非有限响应均按预登记规则标无效。只是解码计算验证，不代表自然图像亚像素准确度。", "导出包旧detail/preprocess残留192x256，而实际ONNX与SimCC解码均384x288。此次以真实形状与精确观测复放核验，不能据旧文本断言模型输入错配。", "12次推理和result.json均已完成，图片目录index.json因NumPy整型裁剪坐标写入失败；只从保存响应重算核对并修复图片目录，没有新增推理，原result.json保持。", "实际审阅全部6帧的原图/旧解码/新解码原生裁片。主鼻形布局几乎不变，后段侧脸轮廓和远侧推断仍存在。没有独立的物理精度改善证据，因此保持原拟合不变，不启动新高斯。", "解码变量已经有界核验，保留原始SimCC响应，不再重复推理或开启DARK后直接训练。下一步读取现有SMPL-X面部语义点的表面支持、前后可见性和损失调用，确认遮挡点进入损失的具体路径，对观测可见性单独制定并验证保守的支持合同；不能用当前错位网格或SAM mask冒充解剖真值。若缺少能区分可见材料和遮挡推断的独立证据，保留未知并停止相应精细约束，不靠换阈值获得通过。", "公式参考：https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/mmpose/codecs/utils/refinement.py", "公式参考：https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/mmpose/codecs/utils/post_processing.py", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\simcc-decoder-conclusion.json"].concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\n'),'此前研究记录：',previous]).join('\n\n');
const candidatePath=build+'/surface123-simcc-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123连续解码核查.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/surface123-simcc-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_surface123_simcc.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/surface123-simcc-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/surface123-simcc-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/surface123-simcc-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/surface123-simcc-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
