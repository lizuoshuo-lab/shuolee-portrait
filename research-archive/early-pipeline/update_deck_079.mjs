import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-surface-consistency-fresh-079';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='166ab72ff633a212f91907ece27a8d64c03e9099e2c8ca1f1fe38325cc76d4e4';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/review/visual-decision.json'),result=await read(run+'/gaussian-run/result.json');
const previous=await read(build+'/depth078-deck-receipt.json');
if(!d.actually_inspected||result.steps!==18000||!result.from_scratch)throw Error('Review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snaps=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
const text=(i,k,value)=>{const e=snaps[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
const note=(i,value)=>{p.resolve(snaps[i].id).speakerNotes.text=value;};
const img=async(i,k,path)=>{const e=snaps[i].elements.filter(e=>e.kind==='image')[k],v=p.resolve(e.id);v.replace({blob:await fs.readFile(path),contentType:path.endsWith('.png')?'image/png':'image/jpeg',fit:'contain',alt:'本轮真实高斯渲染对照'});v.crop={left:0,top:0,right:0,bottom:0};};
text(50,0,'整图定位与人体初始化的训练对照');
text(50,1,'实拍原图、077、079：固定同一整图相机和像素裁片。');
await img(50,0,run+'/review/head-033.jpg');
text(50,5,d.ppt_effect);
text(50,6,'175帧、20万点、18,000步。本轮仅增加局部表面分散约束。');
note(50,'079本地单变量对照，2026-10-24。相机由全部原始整图的SfM与全局BA确定，共175帧、0个插值相机。重新做1200步人体粗拟合，以全部人物轮廓和765个衣物纹理支撑初始化20万点。相机与077一致，重新计算的初始点也逐点一致，不读取旧高斯和优化器。\n局部表面约束：设第i个高斯在相机坐标中的单位法向为n_i、平面距离为d_i，光栅化权重w_i=T_i alpha_i。渲染混合平面与像素射线相交得到共同表面点X，该点在本次求导中停止梯度。E=sum_i w_i(n_i dot X+d_i)^2 / sum_i w_i。新增损失为0.01*mean_valid(2*(sqrt(1+E/(0.01 H_init)^2)-1))。H_init为新初始化点云主轴的99.5%至0.5%分位跨度，固定为5.520190世界单位，不是米。第3000步后启用，仅使用9乘9掩码内部、累积不透明度大于0.8、正深度与稳定法线分母的像素。每步重新渲染X，不把它当真实深度。\n使用15个累积通道求平面二阶矩，RGB路径仍是普通三维高斯，原来的PGSR五通道路径保留。冻结检查中原有前向输出逐像素一致、旧梯度相对差异小于0.001，新项的有限差分与直接求和检查通过。当前适配不是2DGS论文的原版深度畸变项。其余权重、训练采样表、18,000步预算和077一致。\n图中左实拍、中077、右079。全部175帧轮廓平均距离2.33108至2.29713个1536高训练像素，只是小幅变化，不代表人脸几何改善。最终质量与限制集中在下一页。\n源码参考：PGSR https://github.com/zju3dv/PGSR ；2D Gaussian Splatting https://github.com/hbb1/2d-gaussian-splatting 。后者仅作方法参考，未复现。证据：'+run+'/analysis-before-training.json ; gaussian-run/config.json ; review/same-camera-result.json ; review/head-033.jpg');
text(51,2,d.ppt_summary);
await img(51,0,run+'/review/below25-077-gray.png');
await img(51,1,run+'/review/below25-079-gray.png');
text(51,4,'同一下方25度相机：左为077，右为增加局部表面约束的079。');
text(51,5,d.ppt_limit);
note(51,d.summary+'\n'+d.observed_conclusion+'\n'+d.novel_conclusion+'\n'+d.mechanism_conclusion+'\n'+d.next_step+'\n固定共同717组下半脸对应的投影中位数6.17246至6.36134完整去畸变像素，588组上半脸4.52832至4.56477，覆盖均100%。人物内部RGB平均绝对误差0.00625958至0.00633988，外侧覆盖指标0.274759至0.275106。对应对共享轨迹，不能当独立样本；当前只有一次训练对照。未拍摄视角没有真实图像，连续路径仅检查6个固定采样点，未声称逐帧看完150帧。\n本轮最终305174点，PLY统一+Y，位置、协方差、全阶球谐和相机一并转换。浮点近等深度排序造成少量像素差异，逐项诊断通过；几何没有因导出局部变形。文件可导入不代表场景外参真实准确。当前不采用新项，不推广为最佳，不在00部署。\n开源替代参考：2DGS的二维面元和Gaussian Opacity Fields的畸变及法线约束，https://github.com/hbb1/2d-gaussian-splatting ，https://github.com/autonomousvision/gaussian-opacity-fields 。均未在此素材训练。优先核查观测矛盾，不能用换表示宣称外参问题已消失。\n图源与测量：'+run+'/review/visual-decision.json ; paired-transfer-comparison.json ; depth/result.json ; below25-077-gray.png ; below25-079-gray.png');
const changed=[51,52],candidatePath=build+'/surface079-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_表面约束对照.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/surface079-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(changed.includes(i+1))await fs.writeFile(build+'/surface079-final-'+(i+1)+'.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:changed.includes(i+1)});
}
await fs.writeFile(build+'/surface079-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:changed,renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed,canonical_replaced:false}));
