import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/server-delivery-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source rendering changed');
await fs.writeFile(build+'/c0a122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('人像对齐：','人像对齐：背景约束与相机耦合');
replace('原38、50、63帧','历史鼻区材料对照（38 / 50 / 63帧）；不是本轮训练效果。');
replace('新工程裁剪映射','122局部诊断：主点敏感度仅剩0.05%–0.32%，背景残差不能证明人脸对齐。');
replace('裁剪修正后23帧','内参与位姿高度耦合，非唯一根因；未改相机、未重训，逐点<1 px尚未通过。');
const previousNotes=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-27当前结论：122 C0a在121保存的失败相机/点结构处，只读分析共享输出内参四方向与外参/三维点的Jacobian耦合。没有新标定或重训。旧人物图像仅保留作材料对应问题的历史对照，不是122训练结果。',
 '冻结FIT：沙发4/24诊断帧、71条三维轨迹、217个观测；竖屏24/24诊断帧、715条轨迹、2696个观测。第一相机与原尺度基准固定。原生整图像素、输入/图像哈希与轨迹身份核验通过，所有注册相机内观测保留，未用留出点、未按残差删点。',
 '独立方向为logfx、logfy各0.001，cx、cy各1像素。逐点投影掉三维点Jacobian后，再投影掉固定规范后的外参Jacobian。剩余L2导数幅度比例：沙发fx0.1521%、fy0.1148%、cx0.1528%、cy0.0518%；竖屏fx1.3183%、fy0.0209%、cx0.3190%、cy0.0693%。主点方向剩余范围约0.05%到0.32%。这些是局部导数敏感度，不是实拍误差降低率、概率或校准不确定度。',
 '固定合成控制的解析Jacobian与中央有限差分最大绝对导数误差4.87994694e-8。两组四方向系统在相对1e-10阈值下仍满秩，不能声称数学上精确不可辨识。竖屏原BA未收敛，且材料身份未逐点物理认证；弱约束与地面近共面支持相关，尚无对照证明平面性是唯一原因。',
 '因果边界：背景低残差不足以证明K准确或人脸逐点<1原生像素。尚无法唯一区分K错误、外参、材料对应歧义、镜头残余畸变和人物微动。已经去畸变/缩放/裁剪且无额外防抖的MP4不再二次去畸变。草地与室内无有效相机图，未补造分析结果。',
 '工程下一步：R1b十项合成完整协方差/SH/PLY原生回读控制已完成，预览实际拒绝ray3d包并保留119。继续R1c按SHA/完整K/单一位姿/世界坐标/显式后端求值合同加载原生包。原生交互UI、公司00和实拍人脸准入仍未验证。',
 '证据：current/robust-portrait-122/camera-observability/result.json、review.json、jacobian-fixture.json；脚本shuolee-slide-update/diagnose_camera_observability122.py；已有工作日志01_内参文件解析与待核实项.html#robust122-c0a。',
 '以下为之前材料与裁剪诊断的历史记录，保留溯源：', previousNotes
].join('\n\n');
const candidatePath=build+'/c0a122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122局部耦合核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/c0a122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath},null,2));
execFileSync(python,[build+'/preserve_deck_c0a122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/c0a122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/c0a122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/c0a122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected render change on slide'+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/c0a122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
