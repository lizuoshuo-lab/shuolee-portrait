import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=process.cwd();
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const canonical=base+'/reports/02_有限视角重建优化路线.pptx';
const source=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_运动对应修订.pptx';
const expected='bad81c8a5743df30a50b77acc07717b6e23e938607818813ecd786337e64c7ee';
const sha=async f=>crypto.createHash('sha256').update(await fs.readFile(f)).digest('hex');
if(await sha(source)!==expected||await sha(canonical)!==expected)throw Error('Source changed');
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/crown-attribution/covariance-support';
const result=JSON.parse(await fs.readFile(root+'/matched-refit/result.json','utf8'));
if(result.optimizer_updates!==2360||result.steps_per_arm!==1180||result.quality_accepted!==false)throw Error('Unexpected experiment');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count');
const inspection=await p.inspect({select:'$.slides[54]',include:['id','index','elements.id','elements.text','elements.position','notes.text']});
await fs.writeFile('sorting-source-inspection.ndjson',inspection.ndjson);
const row=inspection.records[0],slide=p.resolve(row.id);
const set=(old,value)=>{const e=row.elements.find(e=>e.text===old);if(!e)throw Error('Text not found: '+old);p.resolve(e.id).text=value;};
set('当前效果与后续约束','当前动态模型与排序对照');
set('校正局部尺度与监督来源后，两组分别训练1,180步。','原动态人体与59个观测不变，两组各做1,180次补充更新。');
set('头发训练 L1：原模型 0.01508，发束 0.05135，加先验 0.04280。','头发训练 L1：原模型 0.01508，全局排序 0.01439，层次排序 0.01805。');
set('俯仰碎片仍在，直接调暗偏白高斯会损失实拍纹理。\n保留原动态模型，后续用真实特征对应核验局部几何。','俯仰碎片仍在，两组均未采用。层次排序是逐像素排序的近似。\n继续保留动态人体、独立衣物和真实背景，主轮停在18,000步。');
const imageRow=row.elements.find(e=>e.id.startsWith('im/'));
const img=p.resolve(imageRow.id),frame=img.frame;
img.replace({blob:await fs.readFile(root+'/matched-refit/pitch-45-comparison.png'),contentType:'image/png',fit:'contain',alt:'同一未知俯仰的原动态模型、全局排序拟合、层次排序拟合及其常规排序显示'});
img.frame=frame;img.crop={left:0,top:0,right:0,bottom:0};
slide.speakerNotes.text=row.notes.text+`
\n[DYNAMIC_SORTING_20261022]
本页图像已更新为最新排序对照。前面的发束和真实对应记录是历史排除依据，不对应本页新图片。图中“逐像素排序联合拟合”使用StopThePop层次近似，不是完整逐像素排序参考，也不是真实几何。
\n方法依据：https://github.com/r4dl/StopThePop ，主仓库提交859b11bde9195e19a1c40536e1ab16765b64b365，原生光栅化子模块https://github.com/r4dl/StopThePop-Rasterization ，提交d8cad09919ff49b11be3d693d1e71fa792f559bb。使用未修改的官方源代码独立构建，保留原生GaussianAvatars后端。当前为本地动态输入适配，不是完整官方训练流程复现。
\n传统高斯按中心相机深度排序。在同一高斯的不同像素射线上，最大密度位置可能对应不同的深度。StopThePop提供按射线估计深度和层次重新排序的方法。由于透射率与前后顺序相关，改变顺序也改变颜色与几何的优化目标。当前先做只读渲染，再做匹配补充训练。
\n官方全局排序和原后端的源图平均绝对差为2.4593e-5，单像素最大差0.02294，未通过最初的严格逐像素等价门槛。没有放宽这个门槛。为隔离后端差异，两组补充训练都使用同一个官方后端，仅改变排序模式。完整逐像素参考不支持官方反向传播；训练使用64/20/16队列的层次近似，其源图与完整参考的平均绝对差约0.00649。Proper EWA scaling、新增剔除和负载均衡均关闭。
\n原18,000步参数不变，另设可逆控制变量。全部191,595个点、59个顺序视频观测，两臂使用相同20轮打乱序列，每臂1,180次更新，共2,360次。控制变量为局部位置、四元数方向、对数尺度与直流颜色。头部位移上限3模型毫米，身体6毫米；方向上限90度，尺度乘数0.7至1.3，直流RGB改变量上限0.15。Adam学习率分别0.004、0.0015、0.0005、0.002。损失由0.75已知像素L1、0.25头部L1、0.05池化DSSIM和1e-4控制幅值组成。方向球谐、透明度、点数、相机、时间形变和背景参数固定。没有恢复眼镜专项。
\n真实头发训练L1依次为原全局排序0.01507618，直接换层次排序0.04015221，全局排序补充拟合0.01439335，层次排序补充拟合0.01805272。头部L1依次0.01585969、0.03037118、0.01491647、0.01714917。全部为训练观测，头发和脸部使用38个可靠语义帧。原59帧没有独立测试集，也没有未知俯仰实拍真值。
\n实际检查源图000023、000045与固定俯仰路径索引15、45。全局排序虽降低训练误差，未知俯仰头顶碎片仍在。层次排序补充训练后仍在正面头发与脸部出现额外斑块，未超过原结果。把层次排序候选放回普通中心排序显示时，头发L1进一步变为0.02492746，说明普通PLY查看器不能自动等价显示该候选。没有进行完整背景、全身动态和连续路径验收，因为这两组已经未达到局部视觉采用条件。
\n本轮两组均不采用，不续训，不替换本地查看器，不部署00。原动态人体、独立衣物和恢复地面保持不变，主轮严格停在18,000步。下一项以真实轮廓、运动误差及高斯支持范围作几何检查；先诊断，不将缺少观测的位置补成几何真值，不重复排序权重搜索。
\n可复核证据：${root}/matched-refit/protocol.json、result.json、metrics.json与root-review.json；原生构建记录backend-full-build.json；预检matched-refit/preflight-high/result.json与review.json。
`;
const candidatePath=path.join(build,'sorting-candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_20261022_排序对照修订.pptx';
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:55,
 requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'sorting-final-validation.json')});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));await fs.mkdir('sorting-final-renders',{recursive:true});
for(let i=0;i<q.slides.items.length;i++){
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 await fs.writeFile('sorting-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 if((i+1)%10===0)console.log('Final renders '+(i+1)+'/55');
}
await fs.writeFile('sorting-final-inspection.ndjson',(await q.inspect({select:'$.slides[54]',include:['id','index','elements.text','notes.text']})).ndjson);
console.log(JSON.stringify({finalPath,finalSha256:await sha(finalPath),changedSlide:55,slides:q.slides.items.length,receipt}));
