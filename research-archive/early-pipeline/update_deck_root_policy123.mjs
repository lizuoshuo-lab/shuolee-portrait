import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',b=work+'/shuolee-slide-update',tag='root-policy123-reviewed';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/shared-three-frame/final-state-diagnosis/root-policy-control';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'),filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/fan-observation123-deck-receipt.json'),result=await read(root+'/result.json');
if(await filehash(source)!=='1f56ad01357f643ac1ffd5d6d415d1a9e5184e6c12e898a79d7598c68dd9cff8'||prior.final_sha256!==await filehash(source)||!prior.canonical_replaced)throw Error('Source changed');
const data={};
for(const arm of ['A','B']){
 const final=await read(root+'/'+arm+'/final-metrics.json'),initial=await read(root+'/'+arm+'/initial-metrics.json'),optimizer=await read(root+'/'+arm+'/optimizer-result.json'),r=await read(root+'/'+arm+'/result.json');
 if(!r.schedule_complete||r.numerical_converged||r.final_stationary||r.accepted_for_registered_fixture||r.cuda_initialized||r.real_fits||r.gaussian_steps||!r.all_K_E_unchanged||final.frames.length!==3)throw Error('Result state mismatch '+arm);
 data[arm]={final,initial,optimizer,result:r};
}
const maxdiff=key=>Math.max(...data.A.final.frames.map((f,i)=>Math.abs(f[key]-data.B.final.frames[i][key])));
const bodyDifference=maxdiff('body_rms_px'),faceDifference=maxdiff('held_face_rms_px');
const seq=(arm,key,scale,decimals)=>data[arm].final.frames.map(f=>(f[key]*scale).toFixed(decimals)).join(' / ');
const values=[
 ['指标，按合成帧0 / 1 / 2顺序','A 保留给定根位姿','B 先用身体点调整根位姿'],
 ['身体 RMS（像素）',seq('A','body_rms_px',1,3),seq('B','body_rms_px',1,3)],
 ['留出脸最大误差（像素）',seq('A','held_face_max_px',1,3),seq('B','held_face_max_px',1,3)],
 ['三维顶点 RMSE（毫米）',seq('A','vertices_world_rmse_m',1000,2),seq('B','vertices_world_rmse_m',1000,2)],
 ['缩放梯度（门槛 1e-9）',data.A.optimizer.final_diagnostic.optimizer_gradient_max.toFixed(6),data.B.optimizer.final_diagnostic.optimizer_gradient_max.toFixed(6)]
];
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');const last=snap[51],slide=p.resolve(last.id);
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(b+'/'+tag+'-before-layout.json',JSON.stringify(await p.inspect({kind:'layout'}),null,2));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('专用面部观测对照','根节点初始化的合成对照');
replace('检测位置仍有像素级分歧，尚未证明精度或重影改善','三帧已知几何中，两种策略结果几乎一致');
replace('12项CPU测试与读取映射通过。两种检测中位差约2像素。下一步验证根节点初始化。','身体与留出脸RMS组间最大差仅6.94e-6、6.53e-5像素，未显示B优于A。');
replace('视频第155帧，左起：原图、RTMW、FAN原始、翻转还原、热图合并。五栏同裁剪、同比例。','小投影误差仍伴随毫米级几何误差。两组均未达收敛门槛，未进行实拍拟合或高斯训练。');
for(const e of last.elements.filter(e=>e.kind==='image'))p.delete(e.id);
const table=slide.tables.add({rows:values.length,columns:3,left:66,top:204,width:1148,height:310,columnWidths:[340,382,426],values});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#D6E0E5',width:1});
table.cells.block({row:0,column:0,rowCount:5,columnCount:3}).assign({textStyle:{fontFamily:'微软雅黑',fontSize:22,color:'#193544'},margins:{left:12,right:12,top:10,bottom:10},anchor:'center'});
table.cells.block({row:0,column:0,rowCount:1,columnCount:3}).assign({fill:'#EAF0F3',textStyle:{fontFamily:'微软雅黑',fontSize:22,bold:true,color:'#193544'}});
for(let r=1;r<5;r++)for(let c=0;c<3;c++)table.getCell(r,c).fill=r%2?'#FFFFFF':'#F6F8F9';
slide.speakerNotes.text=[
 '2026-10-29 根节点初始化的唯一合成A/B控制。本页是数学已知几何观测结果，没有实拍效果图，不是实拍相机或人脸精度验收。',
 'A保留给定的camera-domain根位姿，显式跳过支持不足的辅助root conditioning，然后在共同正式目标中联合优化。B仅使用已有合法COCO25身体观测和原权重先调整root，不重新启用髋、手或留出脸点，不更改深度系数。两组同初值、同固定完整K/E、同共享beta、同五阶段目标和停止规则。',
 '已知中性SMPL-X与VPoser生成3帧已知合成投影。全部51个静态脸点作为HELD，不能进入拟合。数学真值无需实拍可见性判定，但不能推广成真实材料对应或物理定位精度。',
 '最终身体RMS A='+seq('A','body_rms_px',1,9)+'px；B='+seq('B','body_rms_px',1,9)+'px。组间最大差='+bodyDifference+'px。',
 '最终HELD脸RMS A='+seq('A','held_face_rms_px',1,9)+'px；B='+seq('B','held_face_rms_px',1,9)+'px。组间最大差='+faceDifference+'px。',
 '最终HELD脸逐帧最大误差A='+seq('A','held_face_max_px',1,9)+'px；B='+seq('B','held_face_max_px',1,9)+'px。完整3D顶点世界坐标RMSE A='+seq('A','vertices_world_rmse_m',1000,9)+'mm；B='+seq('B','vertices_world_rmse_m',1000,9)+'mm。小投影误差并不保证整个三维几何准确。',
 '两组实现检查与登记的合成几何误差不增加条件通过，但均未通过numerical stationarity。A最终缩放梯度='+data.A.optimizer.final_diagnostic.optimizer_gradient_max+'；B='+data.B.optimizer.final_diagnostic.optimizer_gradient_max+'；门槛=1e-9。所有五个正式阶段以objective_stagnation结束。schedule_complete=true不等于numerical_converged，accepted_for_registered_fixture=false。',
 'A耗时'+data.A.result.budget.seconds+'秒，'+data.A.result.budget.evaluations+'次评估；B耗时'+data.B.result.budget.seconds+'秒，'+data.B.result.budget.evaluations+'次评估。GPU任务0、实拍拟合0、检测器/特征网络前向0、高斯训练0。无重试或参数扫描。',
 '本固定合成样本未显示B优于A，不能用这一差别解释实拍严重扭曲。非零姿态与弯曲先验使生成真值通常不是正则化目标的最小值。本结果也未独立检验FAN绝对精度、实拍标定、微动或全部94帧。',
 '来源：'+root+'/result.json；A和B下的initial-metrics.json、final-metrics.json、optimizer-result.json、result.json。来源记录SHA256：'+await filehash(root+'/result.json')+'。',
 '合成数值汇总：'+JSON.stringify({bodyDifference,faceDifference,values,A:data.A.final,B:data.B.final})
].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123根节点初始化合成对照_审校2.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:prior.final_sha256,candidatePath,finalPath,source_result:root+'/result.json',source_result_sha256:await filehash(root+'/result.json'),table_values:values,bodyDifference,faceDifference},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_root_policy123.py','utf8')).replaceAll('root-policy123',tag);
await fs.writeFile(b+'/preserve_deck_root_policy123_reviewed.py',preserve);
execFileSync(py,[b+'/preserve_deck_root_policy123_reviewed.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','52'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[52],requiredNativeChartOwnerSlides:[],
 // Artifact Tool normalizes new table runs to the English family name. Both
 // names are verified localized names in the existing Windows msyh.ttc file.
 // Use an explicit design policy for these aliases rather than falsely claim
 // the source encoded its Chinese font name in English.
 fontPolicy:{basis:'design',families:['微软雅黑','Microsoft YaHei']},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
await fs.mkdir(b+'/'+tag+'-renders',{recursive:true});
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 await fs.writeFile(b+'/'+tag+'-renders/slide-'+String(i+1).padStart(2,'0')+'.png',bytes);if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
if(await filehash(source)!==prior.final_sha256)throw Error('Canonical changed during authoring');
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:prior.final_sha256,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,first51_render_hashes_identical:true,canonical_replaced:false,native_table_slide:52,real_photo_on_slide52:false,actual_changed_slide_review_pending:true,font_alias_evidence:{file:'C:/Windows/Fonts/msyh.ttc',localized_names:['微软雅黑','Microsoft YaHei'],policy_basis:'design',note:'Artifact Tool normalizes new table text to English localized family name. Source encoded Chinese localized family name. Same installed typeface, no source-font substitution.'}},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
