import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'), hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/human117-deck-receipt.json'), expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source last slide render changed');
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('重影诊断：','人像对齐：材料对应与裁剪映射');
replace('固定相机检查','原38、50、63帧鼻区。下半脸15/17条局部匹配有歧义。');
replace('下半脸15/17','新工程裁剪映射已修正。23帧坐标最大改变量1.69 px，重影根因仍未定。');
replace('相机与微动','裁剪修正后23帧流程通过。相机、对应与微动仍待分离，暂不重训。');
p.resolve(slide.id).speakerNotes.text=[
 '当前诊断：117轮固定原074外参，只优化共享f/cx/cy，背景留出中位误差0.49256到0.50544 px，下半脸3.51206到3.53005 px，未通过原验收门槛。114双路线高斯的质量验收也未通过。',
 '材料对应：复用231条三视角轨迹作为次级验证。下半脸15/17条局部相关峰差小于0.05。镜片反光与侧视遮挡让同一语义标签不一定落在同一真实材料点。当前证据不能唯一区分相机误差、错配与人物微动。',
 '图源：117/D2-material-correspondence/semantic_track032_000038_000050_000063.png。三列分别为38、50、63帧鼻尖标签周围128×128原像素的等比例放大，没有几何扭曲。这是原始材料核查图，不是下述裁剪修正后的训练效果。',
 '工程修正：服务器适配的旧映射用了256/size，固定版本GVHMR的三点仿射crop_and_resize把图像端点映射到0..255，因此正确尺度是255/size。新工程直接调用上游实现，并保存每帧source_to_crop和crop_to_source矩阵。255为256个像素中心的端点跨度，size为输入正方形裁剪边长。',
 '数值验证：三个含边界填充的测试中，栅格像素与逆映射最大差0.0161839653源像素。对008已有23帧进行只读核查，旧新映射最大差1.6876866007去畸变源像素。这是坐标改变量，不是重建误差改善，不能据此声称历史重影已定位或消失。',
 '修正后实际回归009：从同一参考原视频全新执行23帧整图SfM、分割、关键点、GVHMR与1200步身体拟合。逐帧正逆映射与原图哈希一致，裁剪端点误差最大0.0000208935裁剪像素。粗关节误差中位0.840%人物框高、P90为2.115%，均为拟合观测内误差。六组实拍/模板覆盖图实际检查后，侧面头部、胸前和手脚轮廓仍有差异。没有高斯训练或新PLY。两轮SfM均重新计算，不能将008/009数字差异归因于裁剪修正。',
 '安装与集成：在本机既有WSL系统内，从私有Git克隆并建立独立虚拟环境，一次安装和实际CUDA算子检查通过。009只验证既有参考视频的工程链路，按步长8采样改变了时序间隔，不是全帧、新人物、异地服务器或稳定版的质量验收。',
 '训练条件：相机、真实材料对应、人物静止性必须先取得独立支持，再各自重新拟合和从头初始化20万高斯、训练18000步。不能以SfM拟合内误差或身体拟合内误差代替这些独立证据。现阶段未发布稳定版。',
 '源码：https://github.com/lizuoshuo-lab/shuolee-portrait 。裁剪修正提交bbca2bc，操作说明提交e7c8be57bf4df50d87c05a1dbbc6f4cdd4c13744。上游：https://github.com/zju3dv/GVHMR ，实际版本固定于工程依赖清单。',
 '本地证据：D:/00/工作/人体重建/provenance/prior-crop-contract.json；D:/GS-SLAM-Data/shuolee-server-install-check-002/prior-crop-audit/result.json；D:/00/工作/人体重建/provenance/installation-check-002.json；D:/00/工作/人体重建/provenance/server-interface-check-009.json。'
].join('\n\n');
const candidatePath=build+'/server-delivery-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_009回归核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/server-delivery-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath},null,2));
execFileSync(python,[build+'/preserve_deck_server_delivery.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/server-delivery-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/server-delivery009-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/server-delivery-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected render change on slide'+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/server-delivery-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
