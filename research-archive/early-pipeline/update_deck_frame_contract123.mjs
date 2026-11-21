import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const work='C:/Users/Administrator/Documents/New project', b=work+'/shuolee-slide-update', tag='frame-contract123';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1026.10807/skills/presentations';
const py='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/smplifyx-reference-control/framewise-canonical-route/training-interface-audit';
const hash=x=>crypto.createHash('sha256').update(x).digest('hex'), filehash=async p=>hash(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(b+'/root-policy123-reviewed-deck-receipt.json'), tests=await read(root+'/cpu-tests-final.json');
const sourceHash=await filehash(source);
if(sourceHash!=='b2722fe072498056ec28ae2a04c20565b731ab57513804859651509ab96d79a3'||prior.final_sha256!==sourceHash||!prior.canonical_replaced)throw Error('Source changed');
if(tests.tests_run!==27||!tests.passed||tests.errors||tests.failures||tests.model_forwards||tests.gaussian_steps||tests.physical_quality_accepted)throw Error('Test scope changed');
const values=[
 ['帧身份','像素映射','输入防串','实际边界'],
 ['真实帧ID关联\n图像、相机、姿态\n明确静态参考帧','完整K保留主点与skew\n显式半像素缩放\n区分像素中心与边界','图像与遮罩SHA绑定\n拒绝缺项、重复ID\n拒绝哈希不符','27项CPU检查通过\n0模型前向、0 GPU\n0新拟合、0高斯训练'],
 ['按ID匹配姿态行\n不依赖数组顺序','旧训练K已有边界约定\n尚无证据证明旧映射错','共享一套身份形状\n加载不代表允许训练','渲染后端尚未连接\n自由衣物姿态运输待接入']
];
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text']})).records;
if(snap.length!==52)throw Error('Slide count');
const last=snap[51],slide=p.resolve(last.id);
await fs.writeFile(b+'/'+tag+'-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(b+'/'+tag+'-before-layout.json',JSON.stringify(await p.inspect({kind:'layout'}),null,2));
const replace=(prefix,value)=>{const e=last.elements.find(e=>e.text?.startsWith(prefix));if(!e)throw Error(prefix);p.resolve(e.id).text.replace(e.text,value);};
replace('根节点初始化的合成对照','逐帧图像与人体姿态的关联接口');
replace('三帧已知几何中，两种策略结果几乎一致','帧与相机的关联已明确，实际训练连接仍待完成');
replace('身体与留出脸RMS组间最大差仅','上轮合成A/B结果几乎一致，数值尚未收敛。实拍亚像素与重影验收仍未通过。');
replace('小投影误差仍伴随毫米级几何误差','下一步接入渲染后端与自由衣物姿态运输，验证指定参考帧的静态导出。');
const tables=last.elements.filter(e=>e.kind==='table');
if(tables.length!==1)throw Error('Expected existing table');
for(const e of tables)p.delete(e.id);
const table=slide.tables.add({rows:3,columns:4,left:66,top:204,width:1148,height:310,columnWidths:[258,310,280,300],values});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#D6E0E5',width:1});
table.cells.block({row:0,column:0,rowCount:3,columnCount:4}).assign({textStyle:{fontFamily:'微软雅黑',fontSize:22,color:'#193544'},margins:{left:12,right:12,top:9,bottom:9},anchor:'center'});
table.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({fill:'#EAF0F3',textStyle:{fontFamily:'微软雅黑',fontSize:22,bold:true,color:'#193544'}});
for(let r=1;r<3;r++)for(let c=0;c<4;c++)table.getCell(r,c).fill=r===1?'#FFFFFF':'#F6F8F9';
slide.speakerNotes.text=[
 '2026-10-29 逐帧图像与人体姿态关联接口。此次是工程输入合同和解析相机的CPU检查，没有模型前向、GPU、实拍拟合或高斯训练。不能称为实拍亚像素、重影改善或目标验收。',
 'frame_pose_contract.py以显式真实frame_id连接图像、遮罩、K_integer_centers、w2c和SMPL-X vertices_world的pose行。参考帧通过reference_frame_id指定。它支持通过ID消除数组排序差异，拒绝重复ID、缺项、哈希不符等不一致输入，不应把合法乱序描述为错误。一个bundle共享身份beta，不创建每帧独立高斯。',
 'for_size保留完整K包含skew，使用明确half-pixel resize，同时提供K_integer_centers与K_renderer_corners。实际图像和遮罩resize、renderer backend并未在本模块执行。旧训练的K经reviewed_admission.py 140–145行确认为corner convention；纯诊断缩放差异不足以证明历史运行存在半像素错误。',
 '当前测试结果：'+JSON.stringify(tests)+'。27项覆盖合同解析、解析相机和CLI，模型前向0，高斯步数0，physical_quality_accepted=false。',
 '首次23项记录保留于cpu-tests-initial.json，其中1个测试误假设for_size数组可变而报错。实现返回immutable数组是预期行为。修正此测试，并加入空NPZ、重复beta、legacy K和CLI检查后，最终27项通过。测试修正不能表示实拍质量改善。',
 '渲染后端、可自由贴皮肤与衣物的高斯姿态运输和完整训练循环仍未接通。旧静态入口的保护保留。底层本地SMPL-X适配已支持三角局部坐标运输，但其数学控制不等于新的端到端训练接口已完成。',
 '上轮唯一合成根初始化A/B最终身体RMS最大差6.936037994e-6像素，留出脸RMS最大差6.529874435e-5像素。五阶段均完成但以objective_stagnation结束，梯度收敛门槛未通过，不据此选择B或解释实拍唯一根因。',
 '来源：'+root+'/cpu-tests-final.json，SHA256 '+await filehash(root+'/cpu-tests-final.json')+'；'+root+'/cpu-tests-initial.json；'+root+'/result.json；D:/GS-SLAM-Data/shuolee-portrait-new-capture-118/src/shuolee_portrait/frame_pose_contract.py，SHA256 '+tests.source_sha256+'。'
].join('\n\n');
const candidatePath=b+'/'+tag+'-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_123逐帧相机关联接口_审阅.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(b+'/'+tag+'-deck-build.json',JSON.stringify({source,source_sha256:sourceHash,candidatePath,finalPath,source_result:root+'/cpu-tests-final.json',source_result_sha256:await filehash(root+'/cpu-tests-final.json'),table_values:values},null,2));
const preserve=(await fs.readFile(b+'/preserve_deck_root_policy123_reviewed.py','utf8')).replaceAll('root-policy123-reviewed',tag);
await fs.writeFile(b+'/preserve_deck_frame_contract123.py',preserve);
execFileSync(py,[b+'/preserve_deck_frame_contract123.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:b+'/'+tag+'-exact-candidate.pptx',finalPath,pythonExecutable:py,
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','52'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[52],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑','Microsoft YaHei'],referencePath:source,referenceSha256:sourceHash},verifyArtifactToolImport:true,receiptPath:b+'/'+tag+'-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
await fs.mkdir(b+'/'+tag+'-renders',{recursive:true});
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=hash(bytes);
 await fs.writeFile(b+'/'+tag+'-renders/slide-'+String(i+1).padStart(2,'0')+'.png',bytes);
 if(i===51)await fs.writeFile(b+'/'+tag+'-final-52.png',bytes);else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected changed slide '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
if(await filehash(source)!==sourceHash)throw Error('Canonical changed during authoring');
await fs.writeFile(b+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:sourceHash,finalPath,final_sha256:await filehash(finalPath),changed_slides:[52],renders,first51_render_hashes_identical:true,canonical_replaced:false,native_table_slide:52,real_photo_on_slide52:false,actual_changed_slide_review_pending:true,source_result:root+'/cpu-tests-final.json',source_result_sha256:await filehash(root+'/cpu-tests-final.json')},null,2));
console.log(JSON.stringify({finalPath,first51_render_hashes_identical:true,canonical_replaced:false}));
