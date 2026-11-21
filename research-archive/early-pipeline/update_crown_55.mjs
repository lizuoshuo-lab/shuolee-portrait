import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=process.cwd();
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const canonical=base+'/reports/02_有限视角重建优化路线.pptx';
const source=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_发束修订.pptx';
const sha=async f=>crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex');
const expected='ae5b4c6cb5c918c6ee992727b08cca44c069688c04bdcc2214a0f7d4a54cd3fd';
if(await sha(source)!==expected||await sha(canonical)!==expected)throw Error('Preserve source edits');
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/crown-attribution';
const probe=JSON.parse(await fs.readFile(root+'/radiance-probe/result.json','utf8'));
const corr=JSON.parse(await fs.readFile(root+'/real-correspondence/result.json','utf8'));
if(!corr.descriptor_support_interior||probe.statistics.selected_points!==688||corr.summary.dynamic.consistent_tracks!==8)throw Error('Unexpected evidence');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count');
const inspect=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.kind','elements.text','elements.position','notes.text']});
await fs.writeFile('crown-source-inspection.ndjson',inspect.ndjson);
const row=inspect.records[0];
const anchor=row.elements.find(v=>v.text?.startsWith('衣物与地面已有局部改善'));
if(!anchor)throw Error('Conclusion anchor missing');
const slide=p.resolve(row.id);
await fs.writeFile('crown-before-55.png',new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer()));
p.resolve(anchor.id).text='俯仰碎片仍在，直接调暗偏白高斯会损失实拍纹理。\n保留原动态模型，后续用真实特征对应核验局部几何。';
slide.speakerNotes.text=row.notes.text+`\n2026-10-22补充：本次为原模型的只读归因与可逆颜色诊断，没有增加训练步数，模型参数、时间形变和背景均恢复并核验。原18k人体SHA256为676675cc2f38913ee95e00f4ceb2ba1fb7d488bf6a1fbaa711516d0e69411ec2。\n原生颜色导数给出各高斯对像素的实际T_i alpha_i合成权重。头顶区域取投影头部上方35%，亮像素阈值取可靠真实头发像素亮度的95分位。这只定位问题，不把亮像素自动判为伪影。俯视20度的亮头顶中94.09%的贡献来自实拍头发占比高的高斯，脸部占比接近零。它们的有效观测数量中位数为35，最近真实观测方向间隔的贡献加权中位数约21.07度。由此不能把问题简单归为少数完全无观测点，也不能宣称头部几何正确。正向20度的上方35%框包括部分额头，不能把该框中脸部贡献解读为串色。\n进一步选择688个实拍头发高斯：头发贡献比例大于0.8，脸部小于0.02，直流颜色均值大于0.8，至少6个可靠头发语义帧且累计合成权重大于1，直流颜色比其所贡献真实像素的加权均值高0.25以上。这些点的直流亮度中位数0.9258，关联实拍混合像素亮度中位数0.3390。对其直流颜色向实拍加权均值插值50%和100%，保持位置、尺度、旋转、不透明度和球谐其他项不变。真实头发L1分别为0.017044和0.020843，原模型为0.015076。实际俯仰仍破碎，不能将亮点变暗作为修复。混合像素颜色不是单个高斯的真实反射颜色，该试验仅排除简单调色路线。\n实拍对应采用OpenCV SIFT与RootSIFT描述子、双向0.75比率筛选和至少4帧无冲突轨迹。最初只要求关键点中心在头发掩膜内，实际审阅发现描述窗口混入背景，因此该版本不作为几何监督。修订后要求关键点到头发掩膜外部的距离至少为5.4倍关键点尺寸加2像素，这是一项保守的窗口支持检查。38帧共有414个内部特征，形成17条至少4帧轨迹。\n两组都使用相同的描述子轨迹，分别采用原场景相机和现有动态头皮439顶点拟合的刚性补偿。三角化使用DLT初始化和soft-L1像素重投影优化，像素索引加0.5后使用完整K。正深度、光线夹角至少3度、全轨迹最大重投影误差不超过3像素、最后一个观测不参与拟合的检查误差不超过3像素时才暂时保留。静态通过6条，动态通过8条。两组全部17条轨迹的重投影误差中位数约2.47与1.47像素，最后观测检查约5.06与3.13像素。动态补偿仍是拟合结果，头皮刚性近似残差的逐帧95分位最大4.39模型毫米。8条稀疏轨迹不能约束整个头顶，也不能作为独立的未知俯仰真值。\n下一步保留原动态结构，扩展真实头发内部的跨帧对应，并检查非刚性运动造成的重投影残差。只有经过区域覆盖、重投影与运动误差核验的对应才可用于局部点位置约束。停止简单调色、整发替换和盲目增加点数。具体证据：${root}/result.json、radiance-probe/result.json、real-correspondence/result.json。`;
await fs.writeFile('crown-after-55.png',new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer()));
const candidatePath=path.join(build,'crown-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_实拍约束修订.pptx';
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'crown-final-validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
await fs.mkdir('crown-final-renders',{recursive:true});
for(let i=0;i<q.slides.items.length;i++){
 const im=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 await fs.writeFile('crown-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await im.arrayBuffer()));
 if((i+1)%10===0)console.log('Final renders '+(i+1)+'/55');
}
await fs.writeFile('crown-final-inspection.ndjson',(await q.inspect({select:'$.slides[54]',include:['id','index','elements.text','notes.text']})).ndjson);
console.log(JSON.stringify({receipt,finalSha256:await sha(finalPath),changedSlides:[55],slides:q.slides.items.length}));
