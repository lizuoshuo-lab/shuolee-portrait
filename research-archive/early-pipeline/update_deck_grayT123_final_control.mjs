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
const prior=await read(build+'/r7-pipeline122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/grayT-final-control-review/visual-review.json');
if(!review.actual_visual_review||review.quality_accepted)throw Error('Review inconsistent');
const figureSpec=await read(build+'/grayT123-final-control-figure.json'),figure=figureSpec.path;
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
await fs.writeFile(build+'/grayT123-final-control-native-layout.json',JSON.stringify((await p.inspect({select:'$.slides[51]',include:['id','layout','elements.id','elements.kind','elements.text','elements.position','elements.style']})).records,null,2));
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/grayT123-final-control-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('静态对齐未通过；训练入口检查完成','逐帧改善了脸部，但统一姿态仍未通过');
replace('同一皮肤点35留出','95个训练视角：头部颜色误差下降19.35%，全身仅1.57%。');
replace('36项工程检查通过','眼眉未参与拟合；新视角仍有拉丝，尚未通过静态验收。');
replace('原图／RTMW有效点','第57帧：原图（闭眼）／静态对照／逐帧形变对照（仍张眼）。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),width=1148,height=width*figureSpec.height/figureSpec.width;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'第57帧：原图、静态对照、FAN逐帧头颈表情对照。原图闭眼而渲染仍张眼。原始工作分辨率裁片，保持比例、无新增图内文字。',fit:'contain'});
image.frame={left:(1280-width)/2,top:180,width,height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["灰衣人物SMPL-X仍未对齐。已审阅00完成的95帧静态/逐帧头颈表情对照及12个统一姿态新视角。头部前景颜色误差均值下降19.35%，84帧改善、11帧退步；全身仅下降1.57%，24帧退步。逐帧版本确实减轻训练视角鼻嘴重影，但第57帧原图闭眼而渲染仍张眼；眼眉在本轮脸部拟合中的有效权重为0。统一姿态点云在新视角的眼周、嘴边、头发、颈肩、裤腿与脚部仍有破碎或拉丝，静态质量不通过。", "有效脸部拟合观测只有鼻852、嘴1900；眉眼0。第57帧眼睑状态不一致可见，但不能由单帧观感量化亚像素误差。", "逐帧形变渲染与统一姿态标准PLY是两个不同表示；PLY不包含逐帧头颈/下颌/表情变量。", "同一组采样头部中心，统一姿态与逐帧状态的95帧中位投影差的中位数为19.932至19.942原生像素；这是模型状态差，不是真实材料的重投影误差，也不是物理运动幅度。", "899×1600训练坐标转1520×2704原生坐标的K行缩放逐帧精确一致。初始审阅报告单位标注错误已修正并保留原件；未改任何训练相机。", "点中心已可自由离开SMPL模板；最近模板顶点距离不等于真实衣物表面精度，不能靠自由点数量声称贴合已通过。", "相机剩余误差、检测/物理对应歧义和真实头部运动各自的份额尚不能唯一分解。", "全部有效独立面部留出逐点<1原生像素未达标；本次95训练视角颜色分数不能替代该验收。", "12个新视角无实拍真值；背面与遮挡表面不能当可靠恢复。", "本次95帧对照审阅已完成，不重复本轮拟合/18k训练。先只读核验刚收到的新视频9/10的实际人物、相机处理信息、静止表情及近景固定纹理覆盖，判断是否带来独立证据；不据此自动准入或启动新候选。现有灰衣相机、特征身份与逐帧表情补偿仍未完全分离，先补能区分它们的证据，再登记单变量修正。", "只读核验完成的00对照：95个训练相机、18000步、461598点。静态与逐帧对照使用同一初始化，不是新人物独立泛化测试。", "头部前景L1 0.0277799→0.0224055，84帧改善11帧退步；头部裁片mask L1 0.0822014→0.0531489，95帧改善；全身前景L1 0.0478256→0.0470740，71帧改善24帧退步。", "实际审阅95个头部三联图、6个全身三联图和12头部/12全身统一姿态新视角。连续角度图已查看，但未在本次原生交互界面验收，也没有新视角实拍真值。", "首次结果把899×1600的像素误写为native，已保留原报告并在unit-correction.json记录纠正。1520/899和2704/1600轴向比例给出严格上下界；相机参数和训练源未修改。", "此前原生训练入口R7的36项合成工程控制通过，不等于本人物物理对齐；本次没有新拟合、相机优化或训练。", "证据：C:\\Users\\Administrator\\Desktop\\GS-SLAM-Lab\\outputs\\mini-splatting2-mip-depth-001\\shuolee\\current\\local-clothed-surface-123\\grayT-final-control-review\\conclusion.json", "图源：D:\\Projects\\Shuolee-analysis-20261028\\dynamic-face-fan-all-views\\head-crops\\frame_000057.png"].join('\n\n');
const candidatePath=build+'/grayT123-final-control-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123灰衣全帧与统一姿态核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/grayT123-final-control-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_grayT123_final_control.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/grayT123-final-control-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/grayT123-final-control-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/grayT123-final-control-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/grayT123-final-control-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
