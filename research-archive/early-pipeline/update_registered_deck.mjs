import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-registered-camera-fresh-072';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='89ec6c5b364f89bf707f4d0067100e8d2691b3737d2aca9fe93ee5b2b98e1e8b';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical report changed; preserve it');
const decision=JSON.parse(await fs.readFile(run+'/review/visual-decision.json','utf8'));
const numeric=JSON.parse(await fs.readFile(run+'/review/same-camera-result.json','utf8'));
const result=JSON.parse(await fs.readFile(run+'/gaussian-run/result.json','utf8'));
const protocol=JSON.parse(await fs.readFile(run+'/controlled-protocol.json','utf8'));
if(!decision.actually_inspected||result.steps!==18000||!protocol.final_primary_view_sampling_order_checked)throw Error('Wait for actual completed review');
const previous=JSON.parse(await fs.readFile(build+'/mask-deck-receipt.json','utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const text=(i,k,value)=>{const e=snap[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
const note=(i,value)=>{p.resolve(snap[i].id).speakerNotes.text=value;};
const img=async(i,k,path)=>{const e=snap[i].elements.filter(e=>e.kind==='image')[k],v=p.resolve(e.id);v.replace({blob:await fs.readFile(path),contentType:path.endsWith('.png')?'image/png':'image/jpeg',fit:'contain',alt:'真实同相机人物高斯对照'});v.crop={left:0,top:0,right:0,bottom:0};};
text(19,3,'175帧相机均由实拍特征配准，再约束共享静态高斯。\n人物掩码内学习颜色；轮廓、法线与跨视图一致性约束几何。');
note(19,(snap[19].notes?.text||'')+'\n072已完成相机输入对照：原59个SfM位姿不变，额外116帧由背景对应EPnP-RANSAC和LM估计，不使用插值初值。人物粗范围、20万点与优化器从头生成，训练18k。旧070只读比较。两阶段原掩码支持不改：身体粗范围拟合59帧有mask，高斯训练175帧有mask。新增116帧初始化投票权重仍为0.5，不同时改权重。相机改变的下游初始化/邻居变化属于输入流程变化，不能称为固定初始点的单项位姿消融。'+JSON.stringify(protocol));
text(50,0,'逐帧相机配准后的训练对照');
text(50,1,'实拍原图、原相机训练、逐帧配准训练：两模型用同一套相机重渲染。');
await img(50,0,run+'/review/head-023.jpg');
text(50,5,decision.ppt_effect);
text(50,6,'两轮均从20万新初始点训练18,000步；完整轮廓与未知视角结论见末页。');
note(50,JSON.stringify({protocol,result,numeric,decision},null,2)+'\n图为预先固定的视图23，完全相同的071相机、像素裁片与显示比例。不能把它看作未知视角真值。轮廓指标按全部175帧计算；实际观察与分割mask并非独立三维真值。');
text(51,2,decision.ppt_summary);
await img(51,0,run+'/review/below25-070-gray.png');
await img(51,1,run+'/review/below25-072-gray.png');
text(51,4,'同一仰视相机：左为原相机训练结果，右为逐帧配准后的从头训练结果。');
text(51,5,decision.ppt_limit);
note(51,JSON.stringify({decision,numeric,protocol},null,2)+'\n两幅仰视图无对应实拍GT，只用于几何与连续外观诊断。相机配准通过不等于白边修复；质量未通过不部署00、不追加步数或自动扫权重。当前为共享静态人物，动态和眼镜专项没有重启。');
const changed=[20,51,52],draft=build+'/registered-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_逐帧相机训练.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/registered-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(changed.includes(i+1))await fs.writeFile(build+'/registered-final-'+(i+1)+'.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unintended rendering change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:changed.includes(i+1)});
}
await fs.writeFile(build+'/registered-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:changed,renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed,canonical_replaced:false}));
