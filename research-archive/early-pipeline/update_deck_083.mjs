import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/person-static-human-surface-init-fresh-083',diag=base+'/current/human-head-surface-initialization-082';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='1b975c89c67002a23d36f8ce6357930be9c7e20d4caa2b63e16764d48d30502a';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const a=await read(run+'/clothing-init/audit.json'),review=await read(run+'/initialization-review.json');
if(a.replaced_face_points!==1772||!review.actual_visual_review||!review.allow_training)throw Error('Reviewed initialization required');
const previous=await read(build+'/consensus081-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snaps=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
const text=(i,k,value)=>{const e=snaps[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
const note=(i,value)=>{p.resolve(snaps[i].id).speakerNotes.text=value;};
const img=async(i,k,path,alt)=>{const e=snaps[i].elements.filter(e=>e.kind==='image')[k],v=p.resolve(e.id);v.replace({blob:await fs.readFile(path),contentType:'image/jpeg',fit:'contain',alt});v.crop={left:0,top:0,right:0,bottom:0};};
text(50,0,'整图相机下的人体表面初始化');
text(50,1,'整图确定相机，人体先验给出中央脸部的初始表面。');
await img(50,0,run+'/clothing-init/patch-review-000136.jpg','第136帧，原图、轮廓交集初始点及人体表面投影后的点，均未进行颜色训练');
text(50,5,'1,772点投向模板曲面，其余198,228点保持原位。');
text(50,6,'175帧，20万初始点。共享静态人物，训练中点可继续离开模板。');
note(50,'083初始化位置对照，2026-10-24。整图SIFT、匹配、SfM和束调整产生175帧相机，0个插值相机。相机内参及外参保持074结果，人物关键点不再解相机。重新做1200步人体粗拟合、衣物特征三角化和175掩码轮廓交集，生成200000点。此阶段人体只给范围和尺度，重算结果与077初始数组逐点相同，旧模型仅作验证参照。\n在同一相机坐标下，从中性FLAME2023开始重新三角化51个固定面部地标并求一个整体相似变换，1000步刚性配准，形状、表情、颈部和眼球参数为零。140帧用于拟合，35帧用于先验投影诊断。原生网格与投影接口检查通过。没有加载082学到的个体脸型。\n中央面部区域在中性模板坐标定义：横向为眉毛地标17至26范围外扩5毫米，纵向从嘴部48至67最低点下方10毫米至眉毛最高点上方5毫米，深度为0至150毫米。三角面片3个顶点必须都在区域内，且至少在3个拟合帧中可见。毫米仅为FLAME模板空间单位，不表示SfM场景已得到公制真值。\n对区域内2135个新轮廓点计算它到可见模板面片的精确最近点。只有新位置在全部175张人物掩码内部时才移动，最终1772点移动，363点保持轮廓原位。其他198228点逐位一致，总数200000。衣服、头发、耳部与颈部外部区域不重采样，也不把衣服压回裸模。图左为原图，中为原始轮廓点，右为初始化更新，青色为调整点，橙色为保留点。图片完全保留，无局部重绘或二维配准。\n本轮只改变初始位置，不增加表面吸附或网格绑定。仍使用PGSR本地静态适配，前景内L1/SSIM、透明度0.1、最小尺度100、法向0.015、多视图几何0.03、NCC0.15、原有观测对应0.002保持077设置。079未通过的表面分散项关闭。分辨率1536高，总预算18000步。新的高斯与优化器从零建立。\n这项适配参考GaussianAvatars的头像表面机制，不是其完整官方网格绑定系统复现。参考https://github.com/ShenhanQian/GaussianAvatars ，https://github.com/jmaraval/GaussianAvatars_SMPLX ，https://github.com/zju3dv/PGSR 。\n证据：'+run+'/analysis-before-training.json ; initialization-review.json ; clothing-init/audit.json ; clothing-init/head-sample-map.npz ; source-final/。质量与限制在末页。');
text(51,2,'083检验人体表面初始化，按同相机和未知视角分别验收。');
text(51,4,'图为077、079的同一下方25度渲染。083尚未完成质量验收。');
text(51,5,'个体头像拟合仍有形状与颈部偏差，未采用其结果。\n本轮仅改变中央脸部起点，未证明能解决重影。');
note(51,snaps[51].notes.text+'\n\n082、083更新，2026-10-24，优先于前文的下一步计划。082在固定整图相机下对比中性刚性配准和共享100维FLAME形状，两臂各1000更新。35留出帧、666个可见鼻口观测的中位误差6.920982至6.127013原像素，降低11.47%。上半部p90为15.913954至12.622516，初始可见性保留95.67%。然而最大形状参数绝对值2.960975超过预先规定2.95，全部12张原生叠图实际检查仍显示耳部、颈部偏差，所以不采用这个个体形状，不把关键点降低称为几何正确。\n083最初的中性面片全区域重采样也在0步被拒绝：平均掩码支持97.97%，最差第84帧88.95%，低于预设99%。原图检查定位到脸颊越出人物轮廓。原候选没有训练，也没有降低阈值。随后在任何高斯更新前修订初始化规则，只移动得到全部175张掩码支持的原轮廓点，不受支持的363点保持原位。修订前协议、失败候选、照片和修订理由均保留。掩码支持是构造条件，绝不是独立质量指标或留出验证。\n当前083已从头启动。最终须报告共同717组下半脸框对应及覆盖、588组上半脸保护、140条固定射线深度宽度、175帧前景和轮廓，并审阅原视角、固定上下25度路径、手脚及衣服。下半脸框不等于已确认的鼻口解剖点。未拍摄视角没有GT。只有通过实际渲染与导出核验才发布新PLY，最新不代表最佳。当前已发布079不变，未部署00。\n证据：'+diag+'/decision.json ; '+run+'/initialization-protocol-v1.json ; analysis-before-training.json ; clothing-init/rejected-face-preflight.json ; initialization-review.json ; gaussian-run/progress.json。');
const candidatePath=build+'/human083-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_人体表面初始化.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/human083-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i>=50)await fs.writeFile(build+'/human083-final-'+(i+1)+'.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i>=50});
}
await fs.writeFile(build+'/human083-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[51,52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed:[51,52],canonical_replaced:false}));
