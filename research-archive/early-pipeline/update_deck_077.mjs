import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-observed-correspondence-fresh-077';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='d7e90fafb38c7b46702322b9158912e2fbb11205989ae17ad1a4860237400503';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const decision=await read(run+'/review/visual-decision.json'),numeric=await read(run+'/review/same-camera-result.json');
const transfer=await read(run+'/review/paired-transfer-comparison.json'),protocol=await read(run+'/analysis-before-training.json');
const result=await read(run+'/gaussian-run/result.json'),previous=await read(build+'/constraint076-deck-receipt.json');
if(!decision.actually_inspected||result.steps!==18000||!result.from_scratch)throw Error('Wait for completed real review');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snaps=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
const text=(i,k,value)=>{const e=snaps[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
const note=(i,value)=>{p.resolve(snaps[i].id).speakerNotes.text=value;};
const img=async(i,k,path)=>{const e=snaps[i].elements.filter(e=>e.kind==='image')[k],v=p.resolve(e.id);v.replace({blob:await fs.readFile(path),contentType:path.endsWith('.png')?'image/png':'image/jpeg',fit:'contain',alt:'本轮真实渲染对照或训练目标'});v.crop={left:0,top:0,right:0,bottom:0};};
text(19,0,'静态人物的 PGSR 与观测对应约束');
await img(19,0,build+'/077-objective.png');
text(19,3,'固定整图标定相机，用同一纹理在另一实拍帧中的位置监督表面投影。\n每步至多32组头部、96组身体对应，鲁棒尺度为2个训练像素。');
text(19,5,'新约束权重为0.002，在第1,000至3,000步线性增加。\n保留原有7×7图像块约束和4°至24°邻居范围。');
note(19,(snaps[19].notes?.text||'')+'\n077新增一项本地研究适配，不是PGSR原论文的默认损失。相机和075保持完全相同。每个实拍对应u_i,u_j，取高斯渲染平面法向n和距离d，按含半像素修正的光线r求深度z=d/(-n^T r)，由固定外参逆变换得到世界点，再投影到相机j。以2像素为尺度，惩罚2[sqrt(1+(||投影位置-u_j||/2)^2)-1]，对当前有效对应取平均。有效条件为源透明度>0.8、正深度和有限投影，不按当前残差剔除需要纠正的观测。加入权重0.002和从1000到3000步的线性系数w_k，后续为1。其余075损失及18000步预算不变。16个邻域中取最接近12度的实际匹配，离线每个SIFT轨迹每个源最多选2个目标，正相关NCC>=0.8并要求多视图重投影检查。158714组对应、31316组头部对应覆盖175帧，排除距离073诊断特征6像素内的整个SIFT轨迹。评价仍来自同一视频，是开发集的特征隔离对照，不是未见图像的真值测试。梯度及像素约定预检通过。每轮重新做1200步人体粗拟合、765衣物支撑三角化和200000点初始化，点数组与075相同，但前1000步仍有少量CUDA数值波动。第一次实现尝试在1001步前因变量重名退出，修复后从0步重跑。\n证据：'+run+'/analysis-before-training.json ; '+run+'/controlled-prefix-check.json ; '+run+'/gaussian-run/config.json');
text(50,0,'真实纹理对应约束的训练对照');
text(50,1,'实拍原图、075、077：固定同一整图相机和像素裁片。');
await img(50,0,run+'/review/head-033.jpg');
text(50,5,decision.ppt_effect);
text(50,6,'两轮均从20万点训练18,000步，本轮仅增加观测对应约束。');
note(50,'预先固定的原视频34帧。左为实拍原图，中为075，右为077，两次渲染使用相同相机和像素裁片，没有二维图像对齐或生成细节。077保持175帧整图相机、20万点初始化和18000步训练，只增加真实图像对应约束。上半脸共同588组对应的RGB像素中心一致投影误差中位数从4.702降至4.528完整去畸变图像像素。对应对可能共享轨迹，这些仍是同一视频的开发集观测，没有未知视角GT。当前是一次比较，轻微改善需重复验证。最终问题与限制集中在下一页。图源：'+run+'/review/head-033.jpg ; 数据：'+run+'/review/paired-transfer-comparison.json');
text(51,2,decision.ppt_summary);
await img(51,0,run+'/review/below25-075-gray.png');
await img(51,1,run+'/review/below25-077-gray.png');
text(51,4,'同一下方25度相机：左为075，右为新增观测对应约束的077。');
text(51,5,decision.ppt_limit);
note(51,decision.summary+'\n'+decision.novel_conclusion+'\n'+decision.next_step+'\n全部175实拍相机下，轮廓平均距离从2.3107至2.3311训练像素，外侧覆盖指标从0.27218至0.27476。训练画幅高度1536像素。两轮均保留717组下半脸、588组上半脸开发对应，点对可共享轨迹。已有相机估计误差与特征噪声尚未排除。\n未拍摄方向没有GT，图中两份高斯使用完全相同的检查相机。来源：'+run+'/review/below25-075-gray.png 和 below25-077-gray.png');
const changed=[20,51,52],draft=build+'/observed077-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_纹理约束训练.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/observed077-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(changed.includes(i+1))await fs.writeFile(build+'/observed077-final-'+(i+1)+'.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:changed.includes(i+1)});
}
await fs.writeFile(build+'/observed077-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:changed,renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed,canonical_replaced:false}));
