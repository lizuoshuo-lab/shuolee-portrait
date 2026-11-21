import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=process.cwd();
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const canonical=base+'/reports/02_有限视角重建优化路线.pptx';
const source=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_排序对照修订.pptx';
const expected='1ca52c141762119fa8b6be4c3bb391a6dad1fa0fe1fba0abdb5e7b5ee7b816cd';
const sha=async f=>crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex');
if(await sha(source)!==expected||await sha(canonical)!==expected)throw Error('Source changed');
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/initial-view-coverage';
const result=JSON.parse(await fs.readFile(root+'/result.json','utf8'));
const review=JSON.parse(await fs.readFile(root+'/root-review.json','utf8'));
if(result.optimizer_updates!==3540||review.steps_per_arm!==1770||review.quality_accepted!==false)throw Error('Unexpected experiment');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count');
const inspection=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile('initialization-source-inspection.ndjson',inspection.ndjson);
const row=inspection.records[0],slide=p.resolve(row.id);
const set=(old,value)=>{const e=row.elements.find(e=>e.text===old);if(!e)throw Error('Text not found: '+old);p.resolve(e.id).text=value;};
set('当前动态模型与排序对照','初始化视角与几何继承');
set('原动态人体与59个观测不变，两组各做1,180次补充更新。','相同19.2万个初始点，两组各1,770步，仅早期视角安排不同。');
set('头发训练 L1：原模型 0.01508，全局排序 0.01439，层次排序 0.01805。','头部训练 L1：先8帧再59帧 0.01997，全程59帧 0.01925。');
set('俯仰碎片仍在，两组均未采用。层次排序是逐像素排序的近似。\n继续保留动态人体、独立衣物和真实背景，主轮停在18,000步。','两组头顶较旧模型连贯，仰视脸部与颈部仍有伪影。\n新旧差异包含几何重置等因素，下一轮单独区分几何继承与位移限制。');
const imageRow=row.elements.find(e=>e.id.startsWith('im/'));
const img=p.resolve(imageRow.id),frame=img.frame;
img.replace({blob:await fs.readFile(root+'/pitch-45-comparison.png'),contentType:'image/png',fit:'contain',alt:'同一俯视的旧18k模型、先8帧再59帧、从第一步用59帧的完整三列原生渲染'});
img.frame=frame;img.crop={left:0,top:0,right:0,bottom:0};
slide.speakerNotes.text=row.notes.text+`
\n[INITIAL_VIEW_COVERAGE_20261022]
本页现在展示初始化视角对照。前面的排序、发束和运动诊断是历史记录，不对应当前图片。图片完整保留三列：保留的18,000步动态人体、重新初始化后先8帧再59帧、重新初始化后从第一步使用59帧。所有列使用相同人物时刻23与相同虚拟俯视相机。虚拟俯仰没有实拍真值。
\n问题与历史依据：早期人体高斯拟合、透明度学习与UV512增密只使用000011、000013、000017、000019、000023、000025、000029、000031这8个观测。后来18,000步主轮采用全部59帧，但从8帧的UV512模型继承位置、尺度和外观。头部新增位置限制在局部4/2/4模型毫米，身体6/3/6模型毫米，且未进行自适应增密。少视角早期拟合可能形成后期难以修正的非凸优化偏差，这是假设而非已证实的唯一原因。
\n匹配对照：在同一注册网格上生成191,595个UV512重心采样点。两组使用相同初始点、初始像素渲染、损失、可学习参数与更新次数。第一组前590步只见8帧，第二组前590步见全部59帧。之后两组按完全相同的59帧打乱顺序训练20轮，共1,180步。每组1,770步，总3,540次真实更新，原主轮仍18,000步。初始化所有RGB为0.5、透明度0.5、非直流球谐为0、局部旋转为单位四元数。各向同性标准差取最近3邻居距离中位数的0.7倍，限定0.5至3模型毫米。
\n共同先验：已拟合的12帧SMPL-X/FLAME网格、相机及原18k模型的128节点时间形变保持不变。本试验只隔离高斯早期训练的观测安排，不是从原始视频重做全部相机估计、人体注册与运动。59帧为不同时间的水平绕拍，不能称为59台同步相机，也没有补充未拍摄的上下视角。
\n参数与更新：位置按参考帧局部模型毫米表达，头部位移范数上限30毫米、身体50毫米。标准差下限0.25毫米，头部上限8毫米、身体12毫米。局部旋转上限90度，直流颜色对应RGB范围0至1，头部非直流球谐系数限定正负0.15，其余区域为0。透明度范围0.01至0.999。Adam学习率依次为位置0.08、对数尺度0.004、旋转0.0015、直流颜色0.01、非直流球谐0.001、透明度0.02。590步后按共同日程衰减。损失包括0.8像素L1、0.2池化SSIM、0.25头部L1、0.05轮廓项，并有较小的位移、尺度、相邻点位移差和球谐幅值约束。人体点数固定，没有增密。背景不参与这一轮白底人物目标的梯度优化，其原参数保持不变。
\n结果：先8帧再59帧/从第一步59帧的头部训练L1为0.01996815/0.01925118，后者约低3.59%。人物L1为0.03115384/0.02980782。头发L1为0.01863599/0.01906706，后者反而约高2.31%。头部指标覆盖59训练观测，头发与脸部区域指标受可靠语义掩膜覆盖限制，使用38帧。没有独立测试集。前590步时头部L1为0.07847/0.02466，但前者没有看过大量评价帧，不能把这项差异当作未知俯仰提升。增加共同训练后差距变小。
\n视觉观察与限制：已检查正面000023、侧面000058、俯仰路径索引15和45。两组新种子在俯视中的头顶较旧模型连贯，但两组间差异较小，仍可见裂缝。仰视脸部和颈部有伪影，没有通过采用条件。旧18k模型与新1770步模型预算不同，并同时改变了点几何继承、初始尺度、位置范围与旋转自由度，因此不能将新旧外观差异归因于视角数。短程单次试验也不足以判定最终收敛差异。完整背景和动态连续路径尚未对新候选验收。
\n下一步：所有组从第一步采用相同59帧顺序与预算。先将旧参数无损转换到共同物理参数化，要求零次更新原生渲染等价，保持初始点不变比较紧/宽位置增量范围。再在共同宽范围内比较继承/新采样几何。不得先裁剪旧位置或尺度后称为等价基线。继续以连续俯仰稳定性、实拍纹理与人物身份共同评判。此下一轮尚未开始。当前两组只保留作原因对照，没有替换原模型、没有部署00，眼镜专项继续停止。
\n前置排除：完整头冠轮廓检查中，没有可靠头发中心持续在至少6帧超出估计轮廓5像素，不能据此大幅删点。弱观测球谐方向投影只带来极小训练误差变化，没有可见碎片改善。轮廓一致仍不证明内部深度正确。
\n可复核原始材料：${root}/protocol.json、lineage-audit.json、result.json、root-review.json、两组schedule.json、next-factor-isolation-plan.json。实现：train_initial_view_coverage.py。原主模型、衣物模型与查看器点云经SHA-256核验保持不变。
`;
const candidatePath=path.join(build,'initialization-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_初始化对照修订.pptx';
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'initialization-final-validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));await fs.mkdir('initialization-final-renders',{recursive:true});
const parity=[];
for(let i=0;i<q.slides.items.length;i++){
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 const name=String(i+1).padStart(2,'0')+'.png',out='initialization-final-renders/'+name;
 await fs.writeFile(out,new Uint8Array(await png.arrayBuffer()));
 if(i<54)parity.push({slide:i+1,identical:await sha(out)===await sha('sorting-final-renders/'+name)});
 if((i+1)%10===0)console.log('Final renders '+(i+1)+'/55');
}
if(parity.some(v=>!v.identical))throw Error('An unchanged slide differs visually');
await fs.writeFile('initialization-first54-parity.json',JSON.stringify(parity,null,2));
await fs.writeFile('initialization-final-inspection.ndjson',(await q.inspect({select:'$.slides[54]',include:['id','index','elements.text','notes.text']})).ndjson);
console.log(JSON.stringify({finalPath,finalSha256:await sha(finalPath),changedSlide:55,slides:q.slides.items.length,first54Identical:true,receipt}));
