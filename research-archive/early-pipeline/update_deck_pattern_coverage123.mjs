import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='pattern-coverage123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-global-initializer';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/video10-native-patch123-deck-receipt.json'),review=await read(root+'/../../local-clothed-surface-123/pattern-preflight/coverage/visual-review/review.json');
if(await filehash(source)!==prior.final_sha256||!prior.canonical_replaced)throw Error('Source changed');
if(!review.actual_saved_native_crop_review||!review.all99_context_and_face_tiles_reviewed)throw Error('Review missing');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51];
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('局部细化没有修复人像对应','脚部预测落在桌面和上衣');
replace('24组实拍查询','99帧核验：54帧出现出画脚点，图内点仍有错误部位对应。');
replace('鼻嘴查询虽能闭环','人体mask内的错误脚点仍可能拉偏拟合，置信度不能代替可见性。');
replace('同一鼻嘴查询','第120、245、370帧，预测点居中的原生192px裁片。图像未扭曲。');
const figure=root+'/../../local-clothed-surface-123/pattern-preflight/coverage/visual-review/invalid_foot_locations_native.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'video4第120、245、370帧的原生192像素裁片，查询中心为预测脚点，实际可见部位分别为桌面、上衣和袖子。无图内文字或像素形变。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
const notes=p.resolve(last.id).speakerNotes;
notes.text=["2026-10-29。新人物video4.mp4（pattern花纹上衣）区域核验完成。SHA256 d9555dc04d313f4615ad4de08640f0b44efb097b4654151bc27bb946a71455d5。491原帧按既定stride5解码99帧，2704x1520，不二次去畸变。", "仅一次现有官方YOLOv8n与RTMW-L推理。YOLO640/conf0.45/IoU0.7；RTMW288x384，RGB官方归一化、1.25padding、0.04bodymargin、现有argmax解码，无翻转或阈值扫描。99帧人物区域连续，但检测身份与精确解剖可见性不同。48.36秒完成。", "原生逆仿射与独立算术差值0px，这是坐标映射一致性，不能当真实关键点精度。所有133点及原始SimCC响应保留。594个脚部预测有105个超出原图，涉及54帧。其余489个在图内的预测并未因此通过可见性认证。", "实际审阅99个上下文与99个人脸原生裁片、5个512px预测脚部上下文及4个192px点位裁片。接触表在工具显示时缩小，只用于覆盖审阅，不用于亚像素判断。192px点位裁片按原生尺寸单独查看。", "第120帧slot24/key17 left_big_toe=(1412.86955,1370.47018)，原图对应桌面。第245帧slot49/key21 right_small_toe=(1530.58454,974.92645)，原图对应上衣/头发。第370帧slot74/key21=(1218.85655,939.44572)，原图对应袖子。第490帧slot98/key17=(1390.82499,1429.59519)，原图对应桌边设备/线缆附近。四例均不是该脚部物理点。", "现有image_support已经严格拒绝出画点，并计算离personmask距离，保留4px容差。它不验证解剖部位，落在上衣内的脚点仍可能通过该过滤。可选mesh_surface_zbuffer是模型自遮挡近似，源码明确external_occluders_modeled=False。尚未为pattern生成mask或拟合，不能声称上述105个出画点已进入训练或实际权重不为零。", "99帧只覆盖左侧、正面和右侧，未显示背面；腿脚经常被桌子遮挡或截断，个别视角有部分粉色鞋。头发遮挡侧脸，slot36/source180与slot79/source395可见闭眼例。人体微动没有量化，重复百叶窗/反射不能自动作为固定背景真值。", "旧灰衣SMPL-X鼻嘴真实错位，原面部中位43.99px，旧全身门槛约50至53px不足以保证面部精度。当前新训练入口已有独立精细对齐准入。此次新人物的点位错误不能唯一解释旧人物的内外参、材料身份和微动。", "下一步预登记一次99帧原生纹理候选：全局2500、人物1500、头部1000预算，独立近重复族与整族留出，在任何几何残差前冻结。只做材料支持核验，不把RTMW编号当跨视角物理点。新相机、SMPL-X和Gaussian训练均未开始。独立可信面部HELD仍0。", "证据：local-clothed-surface-123/pattern-preflight/coverage/result.json、support-audit.json、visual-review/review.json、decision.json及native-material-plan.json。前51页不变，原历史PLY不变，未验证00部署或WPS云往返。"].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123遮挡关键点核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,figure,figure_sha256:await filehash(figure)},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_video10_material123.py','utf8')).replaceAll('video10-material123',tag);
await fs.writeFile(b+'/preserve_deck_pattern_coverage123.py',preserve);
execFileSync(py,[b+'/preserve_deck_pattern_coverage123.py'],{stdio:'inherit'});
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
