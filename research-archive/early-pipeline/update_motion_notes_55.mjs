import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project', build=process.cwd();
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_实拍约束修订.pptx';
const canonical=base+'/reports/02_有限视角重建优化路线.pptx';
const sha=async f=>crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex');
const expected='6b5ecf3bf9c704504744d6cbe4be17bc3ca391acdf7cc68d8ae95d49a9c5f9fb';
if(await sha(source)!==expected||await sha(canonical)!==expected)throw Error('Source has changed');
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/crown-attribution';
const read=async f=>JSON.parse((await fs.readFile(f,'utf8')).replace(/^\uFEFF/,''));
const local=await read(root+'/local-motion/result.json');
const flow=await read(root+'/real-flow/result.json');
const inter=await read(root+'/real-flow-intermediate/result.json');
const trial=await read(root+'/measured-patch-pilot/result.json');
if(local.summary.triangle.all_loo_consistent!==7||inter.local_all_loo_passed!==28||trial.local_control_optimizer_updates!==472)throw Error('Evidence does not match');
const p=await PresentationFile.importPptx(await FileBlob.load(source));if(p.slides.items.length!==55)throw Error('Slide count');
const inspection=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile('motion-source-inspection.ndjson',inspection.ndjson);
const row=inspection.records[0],slide=p.resolve(row.id);
if(row.notes.text.includes('[DYNAMIC_REAL_MOTION_20261022]'))throw Error('Already updated');
slide.speakerNotes.text=row.notes.text+`
\n[DYNAMIC_REAL_MOTION_20261022]
实拍对应核验与局部控制点试验：当前继续采用原18,000步动态人体、独立衣物和真实静态背景；59张标定图是同一视频的不同时刻。以下结果均来自同一训练资料，不是独立新视角或物理深度真值。
\n对原17条SIFT轨迹改为逐个观测留出检查。每次三角化均排除待检查坐标，局部三角形绑定也只由余下观测选择；现有运动场仍使用过全部59张训练图。静态相机通过4条，整头刚体与局部三角形运动各通过7条。前一轮的6/8只针对最后观测留出，检查标准不同，不能直接解释成模型退化。局部绑定变换与原生投影最大差8.81e-5像素。可靠对应稀疏，不足以监督整个头顶。
\n扩充实拍轨迹采用Shi-Tomasi与OpenCV金字塔Lucas-Kanade，窗口9×9、最大金字塔层级1；前后向误差不超过0.5像素，9×9图像块的零均值归一化相关至少0.88、归一化绝对差不超过0.08。点到已核验头发掩膜外部的距离至少18像素；至少5个观测、跨度至少6个标定帧。直接跟踪得到30条去重轨迹，26条通过粗几何检查，22条通过局部运动的全留一检查；这些轨迹集中在侧面，0.5像素扰动对应的3D位置95分位变化约3.80至7.60模型毫米，尚不足以充当精确深度。
\n检查原取帧配方后，确认59张标定图来自原视频每3帧取1帧。解码176个真实帧，并复用原去畸变映射；59张标定图的头部ROI均与原图逐像素一致。中间帧仅用于连接光流，不插值相机，不增加伪标定视角，不生成训练图片。每个中间跟踪位置还须满足相邻已测头发支持的保守交集及前后向、纹理一致性条件。相同种子设置得到119条去重轨迹，74条通过粗检查，28条通过更严格的局部全留一检查，其中3条达到0.5像素扰动的3D变化95分位小于2模型毫米。它们新增少量正面对应，但仍不能恢复缺失的俯仰观测。
\n对3条较稳定轨迹进行原生深度合成核验。预计算颜色使用各高斯的相机Z、常数1和可靠头发标记，黑背景渲染后用深度和除以不透明度，得到透射率加权的高斯中心深度。这不同于真实表面首交深度。容限为像素扰动95分位与刚体/局部运动差异之和，并设置1模型毫米下限。只有轨迹76在多数观测中具有超过容限的同向差异，且绝对差中位数超过容限至少1毫米，进入有限试验。其余对应只作检查。
\n局部训练不改原高斯参数文件。围绕轨迹76的三维位置设置5个控制点，8模型毫米宽度的RBF权重与半径30毫米的C2紧支撑权重组合，影响800个可靠头发高斯。控制位移映射回原三角形局部坐标并随原运动变化，规范空间位移上限3模型毫米。颜色、协方差、透明度、脸部、身体、原运动与背景均固定。两组从零位移开始，使用完全相同的59帧顺序、Adam学习率0.03，各236次控制点更新，总472次；主轮仍为18,000步。
\n仅光度组使用0.75头部L1与0.25可靠头发L1；无可靠语义时使用头部L1。另一组额外对观测深度差的容限外部分使用平滑绝对值，权重0.02；另有控制点平方幅值惩罚1e-5。这里的试验图“局部颜色对照”表示光度损失对照，两组都只优化局部位置，颜色参数没有更新。
\n轨迹76的绝对深度差中位数：原模型2.9826、光度对照2.8089、加入深度约束2.0111模型毫米；最大差分别4.9242、5.0416、5.2876毫米。因此中位数改善不等于所有观测变好。实际最大规范空间位移分别0.124与0.683模型毫米。头发训练L1：原0.01507587、光度对照0.01507424、加入深度0.01508431；脸部L1三组逐值相同0.01816296。
\n固定时刻的同一61帧俯仰路径和正面/侧面源图对照显示：局部约束未带来可见的头顶碎片改善，脸部和衣领原有伪影仍在。本轮不采用、不部署00，不继续该小片区域的训练；保留原动态主模型与实拍对应材料。下一步检查头发整体几何、轮廓支持和高斯协方差朝向之间的关系，避免把少数深度对应扩展成全头几何真值。眼镜专项保持停止。
\n证据目录：${root}/local-motion/result.json；real-flow/result.json；real-flow-intermediate/video-parity.json与result.json；measured-depth-flow/preflight.json；measured-patch-pilot/result.json。以上均为本地适配，未宣称完整OmniRe、COLMAP或新的人体重建官方流程复现。
`;
const candidatePath=path.join(build,'motion-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_运动对应修订.pptx';
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'motion-final-validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));await fs.mkdir('motion-final-renders',{recursive:true});
for(let i=0;i<q.slides.items.length;i++){
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 await fs.writeFile('motion-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 if((i+1)%10===0)console.log('Final renders '+(i+1)+'/55');
}
await fs.writeFile('motion-final-inspection.ndjson',(await q.inspect({select:'$.slides[54]',include:['id','index','elements.text','notes.text']})).ndjson);
console.log(JSON.stringify({finalSha256:await sha(finalPath),notesUpdatedSlide:55,visibleSlideChanges:0,slides:q.slides.items.length,receipt}));

