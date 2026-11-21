import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-native-align-fresh-113',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='13ef39712f9bec337145b6ed83d4d038def1a6a110e249811fe360424a2c74c9';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const prior=await read(build+'/human112-readable-deck-receipt.json');
const evidence=await read(run+'/presentation-evidence.json'),d=await read(run+'/decision.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','layout','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slidecount');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/human113-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('训练期对齐：先确定可靠表面','训练期对齐：同一特征共享空间位置');
replace('原生交点与几何梯度已验证','共享三维特征点已接入训练，同步优化颜色、轮廓与位置');
replace('球形初值的图像不随方向变化','先逐层计算位置误差再聚合，避免前后高斯靠平均位置抵消误差。');
replace('先建立可信法向再约束共享点','18,000步训练进行中；最终以全身边界和连续新视角评估。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one evidenceimage');
const image=p.resolve(images[0].id);image.replace({blob:new Uint8Array(await fs.readFile(run+'/presentation-stage.png')),contentType:'image/png',fit:'contain',alt:`本轮第${evidence.step}步的三组实际同相机实拍与渲染；训练中间结果，尚未通过质量验收`});
image.frame={left:65,top:177,width:1150,height:384};
const notes=[d.summary,
 '方法：每个可匹配特征各有一个共享XYZ。重新用原始二维对应和固定074相机估计2,714个目标，包含50个命名面部特征；辅助观测54,514条覆盖全部175帧。相机估计误差与遮挡歧义仍存在，目标不是3DGT。',
 '数学：视线o+z d与每个有限高斯的精度矩阵Q=Sigma^-1，求密度最大深度z=d^T Q(mu-o)/(d^T Q d)。用原生RGB贡献确定权重，逐高斯平方残差聚合后再作鲁棒化。辅助几何分支固定当前贡献和尺度，只对中心和方向求导；原颜色、透明度、尺度等仍从正常损失学习。',
 '切平面版本1400观测89.6%有效，未通过预先90%门槛。保留失败记录，按有限协方差修正后同批98.6%有效，梯度与RGB不变检查通过。剩余空射线继续记录；不能当作成功零误差。',
 '初始化：全部175视角mask生成穿衣轮廓表面，重新拟合中性头部和人体范围；20万高斯中心，法向来自新轮廓梯度和新中性头部，法向标准差为切向0.25倍。最终点不被吸回裸模。旧高斯、优化器与旧人物修正均未作为初值。',
 '这是方向初始化与新目标的组合实验，不能拆成单项因果结论。静态模型不保证能同时解释真实微动或错误对应。每500步若有效对齐观测不足90%则停止并保存待查。',
 `图源：113本轮原生渲染第${evidence.step}步；原图帧34、70、100。每组左为实拍、右为同相机训练图。由实际阶段图选取预定相机，未作美化修复；尚非最终效果。`,
 '结束后同098比较全身和面部实拍视图、辅助约束留出与连续新视角；只有标准及+Y完整协方差/所有SH的PLY回读和实际打开文件通过后才更新交付模型。00尚未部署。',
 '原生3DGS/PGSR源码：https://github.com/zju3dv/PGSR 。本轮对齐分支是本地研究扩展，不声称上游官方方案。研究目录：'+run];
p.resolve(slide.id).speakerNotes.text=notes.join('\n\n');
const candidatePath=build+'/human113-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_训练中对齐.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human113-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human113-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slidechange '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human113-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
