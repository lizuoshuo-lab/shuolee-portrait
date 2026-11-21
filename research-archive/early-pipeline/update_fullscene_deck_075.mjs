import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-fullscene-camera-fresh-075';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='9974038724b0ba58e8f59a837019a5e6d422620ca1cbb5c89acd7a1b64cb187a';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical report changed; preserve it');
const decision=JSON.parse(await fs.readFile(run+'/review/visual-decision.json','utf8'));
const numeric=JSON.parse(await fs.readFile(run+'/review/same-camera-result.json','utf8'));
const result=JSON.parse(await fs.readFile(run+'/gaussian-run/result.json','utf8'));
const protocol=JSON.parse(await fs.readFile(run+'/analysis-before-training.json','utf8'));
if(!decision.actually_inspected||result.steps!==18000||!protocol.no_old_fit_gaussian_or_optimizer)throw Error('Wait for actual completed review');
const previous=JSON.parse(await fs.readFile(build+'/registered-deck-receipt.json','utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const text=(i,k,value)=>{const e=snap[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
const note=(i,value)=>{p.resolve(snap[i].id).speakerNotes.text=value;};
const img=async(i,k,path)=>{const e=snap[i].elements.filter(e=>e.kind==='image')[k],v=p.resolve(e.id);v.replace({blob:await fs.readFile(path),contentType:path.endsWith('.png')?'image/png':'image/jpeg',fit:'contain',alt:'真实同相机人物高斯对照'});v.crop={left:0,top:0,right:0,bottom:0};};
text(19,3,'整幅图SfM与联合束调整估计175帧相机，统一去畸变。\n人体粗拟合确定范围，人物轮廓和衣物纹理初始化点云。');
note(19,(snap[19].notes?.text||'')+'\n075当前流程：全部175张原始完整画面重新提取SIFT、全对匹配、几何检验、SfM和全局BA，内外参及径向畸变共同优化。依据新相机统一映射RGB、mask和特征。1200步新的SMPL-X粗拟合限定初始化包围范围；175帧轮廓与765条衣物纹理支撑初始化20万高斯，不把点绑定到裸人体网格。继承既定静态PGSR损失，人物mask内部学习颜色。模板、原始视频和检测器输出可以复用，之前个人化拟合、高斯、优化器都未加载。参考072仅用作对照；单个全局相似变换统一坐标，未参与074相机求解。新旧初始化不同，所以是完整输入流程比较，不是只改变外参的固定点消融。证据：'+run+'/analysis-before-training.json ; '+run+'/preparation.json ; '+run+'/clothing-support/audit.json\n'+JSON.stringify(protocol));
text(50,0,'整图相机与人体初始化的训练对照');
text(50,1,'实拍原图、072、075：两份高斯使用同一075相机与像素裁片。');
await img(50,0,run+'/review/head-033.jpg');
text(50,5,decision.ppt_effect);
text(50,6,'两轮均从20万新初始点训练18,000步；完整轮廓与未知视角结论见末页。');
note(50,JSON.stringify({protocol,result,summary:numeric.summary,gates:numeric.gates,decision},null,2)+'\n图为预先固定的原视频34帧，完全相同的075相机、像素裁片与显示比例。图源：'+run+'/review/head-033.jpg。轮廓指标按全部175帧计算，评测图像高1536像素；掩码并非独立三维真值。');
text(51,2,decision.ppt_summary);
await img(51,0,run+'/review/below25-072-gray.png');
await img(51,1,run+'/review/below25-075-gray.png');
text(51,4,'同一下方25度相机：左为072，右为整图定位与人体初始化后的075。');
text(51,5,decision.ppt_limit);
note(51,JSON.stringify({decision,summary:numeric.summary,gates:numeric.gates,protocol},null,2)+'\n图源：'+run+'/review/below25-072-gray.png 和 below25-075-gray.png。两幅仰视图无对应实拍GT，只用于几何与连续外观诊断。074独立面部检查失败的证据仍保留，075依照用户要求完成有界端到端对照。当前为共享静态人物，动态和眼镜专项没有重启。');
const changed=[20,51,52],draft=build+'/fullscene075-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_整图定位人体训练.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/fullscene075-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(changed.includes(i+1))await fs.writeFile(build+'/fullscene075-final-'+(i+1)+'.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unintended rendering change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:changed.includes(i+1)});
}
await fs.writeFile(build+'/fullscene075-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:changed,renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed,canonical_replaced:false}));
