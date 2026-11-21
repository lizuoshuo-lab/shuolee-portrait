import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const build=process.cwd(),workspace='C:/Users/Administrator/Documents/New project';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const root=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background';
const cloth=root+'/observed-clothing-pilot',surface=root+'/surface-gaussian-pilot';
const original=base+'/reports/02_有限视角重建优化路线.pptx';
const buffer=await fs.readFile(original),sha=crypto.createHash('sha256').update(buffer).digest('hex');
if(sha!=='b11a18d22ecf5c393b06ee79f62c1ccb8d039fa47b4f940c6ff50f774ff55269')throw Error('Source changed; preserve external edits');
const source=path.join(build,'source-before-clothing.pptx');await fs.writeFile(source,buffer);
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({kind:'slide,textbox,image,shape',maxChars:1000000});
await fs.writeFile('clothing-source-inspection.ndjson',snap.ndjson);
const rows=snap.ndjson.trim().split('\n').map(JSON.parse);
function text(slide,name,value,left,top,width,height,size=26,bold=false,color='#183342'){
 const sh=slide.shapes.add({geometry:'textbox',name,position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={fontSize:size,typeface:'微软雅黑',color,bold};sh.text.insets=0;return sh;
}
async function picture(slide,name,file,left,top,width,height){
 return slide.images.add({blob:new Uint8Array(await fs.readFile(file)),contentType:'image/png',alt:name,fit:'contain',position:{left,top,width,height}});
}
function footer(slide,n){text(slide,'page-number',n+' / 54',1130,679,86,20,14,false,'#536674');}
function setExact(old,newText){const r=rows.find(x=>x.kind==='textbox'&&x.text===old);if(!r)throw Error('Text anchor missing: '+old);p.resolve(r.id).text=newText;}
setExact('人体共享一套高斯外观，随时间更新表面位置。静态背景保留原场景坐标，\n与人物共同渲染遮挡。','人体与独立衣物分别共享高斯外观，随时间更新位置。静态背景保留原场景坐标，\n与人物共同渲染遮挡。');
const s53=p.resolve(rows.find(x=>x.kind==='slide'&&x.slideIndex===52).id);
p.delete(rows.filter(x=>x.slideIndex===52&&['textbox','shape','image'].includes(x.kind)).map(x=>x.id));
text(s53,'title','观测衣物高斯与独立时间形变',66,42,1148,67,42,true);footer(s53,53);
text(s53,'method','从旧场景筛出 30,592 个衣物高斯，排除头、手与地面。\n固定原人体，比较静态衣物与独立时间形变，两组各训练 885 步。',66,122,1148,72,25);
await picture(s53,'真实人物和动态人体两种衣物对照',cloth+'/matched-training/000058-clothing.png',66,210,1148,398);
text(s53,'result','两侧衣服轮廓更完整。身体训练 L1 从 0.02436 降至 0.01407。',66,627,1148,42,25);
s53.speakerNotes.text='本页为本地实际渲染，不是论文原图。主人体191595点、原有128节点三次时间形变、原相机和258633背景高斯全部冻结。候选仅引入30592个已有观测衣物高斯，总人体及衣物222187点，包含背景480820点。主体19.2万点主轮仍为18000步，新增衣物候选的885步单独计数。没有眼镜专项。\n衣物来自同坐标系原场景316044点高斯中的主体候选57411点。其来源模型曾用51张本视频图像训练，因此所有结果都是训练诊断，不是独立测试。筛选至少40视图有效，身体掩膜投票比例不少于0.85，头部投影比例小于0.02，排除任何接触头部映射顶点的三角面片，SMPL-X腕手关节及手指LBS影响低于0.2。高度超过地面0.12模型米，中心距最近三角形中心小于0.18模型米，最大高斯标准差小于0.04模型米，不透明度大于0.05。手部排除是先验语义，不宣称真实衣物标签。\n固定时刻23，用三角面片朝向R_f、中心c_f和尺度s_f换算局部参数。局部中心为R_f转置乘(世界中心-c_f)/s_f，局部尺度为世界尺度/s_f，局部四元数为q_f的逆乘世界四元数。随时间的世界中心为R_f(t)乘局部中心再乘s_f(t)并加c_f(t)。相同参考时刻坐标往返最大误差1.2e-7量级，先运行原生RGB对齐检查。保留完整3D协方差和原始SH3外观，后续SH非DC与局部旋转冻结。\n直接套用原人体已拟合形变使衣物侧面拉偏，因此短程对照分别用固定参考衣物和独立衣物形变。独立分支复用此前ENeRF残差图：128节点，每顶点4邻居，三次时间函数，时刻23固定。将残差系数归零初始化，由当前59实拍图监督。原人体动态不动。\n两组使用相同15轮59视角顺序，每组885次更新，共1770次optimizer.step。仅优化新增衣物局部位置、DC、不透明度、尺度，独立分支同时优化衣物运动。XYZ学习率0.00035，DC0.0015，opacity0.01，scale0.0006，motion0.001，590步之后乘0.3。局部位置相对初始化各坐标限4模型毫米，尺度限0.8到1.25倍。损失为0.8RGB L1+0.2DSSIM+0.25头部保护L1+0.05轮廓+小参数锚定。独立运动沿用0.005局部刚性与0.001系数幅度正则。\n身体L1：原模型0.024360814，静态衣物0.014124026，独立时间形变0.014073269。身体alpha<0.5比例由2.4698%降至0.5805%。头部L1从0.015859447变为0.015862721，头部参数逐元素未变，轻微差异来自衣物遮挡/区域边界。独立动态相对静态仅约0.36%的训练L1差异，没有证据宣称动态分支优于静态分支。\n证据：'+cloth+'/matched-training/protocol.json\n'+cloth+'/matched-training/result.json\n'+cloth+'/review/result.json\n复用原代码：GaussianAvatars https://github.com/ShenhanQian/GaussianAvatars ，原有本地ENeRF形变图。当前是局部组合实验，不是完整OmniRe官方复现。';
const s54=p.slides.add({background:{fill:'#ffffff'}});
text(s54,'title','当前不足与后续验收',66,42,1148,67,42,true);footer(s54,54);
await picture(s54,'同一俯仰下原人体与独立衣物候选的头部',cloth+'/review/pitch-45.png',66,151,704,445);
text(s54,'limits','头顶和脸部在未知俯仰下仍有碎片外观。\n衣物改善没有修复头部几何。',815,166,399,142,26);
text(s54,'next','衣物独立动态与静态对照的差别很小。\n下一轮检查侧面轮廓、时间稳定性及脚边背景缺口。',815,356,399,170,26);
text(s54,'boundary','59 张均参与训练，未知俯仰没有真值。候选仅供本地对比，尚未部署 00。',66,625,1148,44,24,false,'#536674');
s54.speakerNotes.text='原头像参数完全保持，未知俯仰无法因衣物替换而获得真实新观测。本页完整放置同一相机的两列实际图，未生成细节、未锐化。旧表面2DGS路线已完成325次单参数替换诊断以及3组各885步弱约束续训。单独替换尺度/旋转/位置没有修复碎片；从纯图像预热状态分别加入较弱法向或深度约束后，训练拟合回退减轻，但未知俯仰没有显著突破。该轮几何分支未采用，失败大断点删除前已保存哈希、协议、指标和实际图片。\n衣物实验取得局部轮廓与覆盖改善，但仍不能等同完整人物/场景重建质量通过。静态与动态衣物两组训练差异很小，需要持续时间视频与未拍摄角度实际观察，后续只在有证据时增加复杂度。原所有59水平观测均参与训练，不报告独立测试PSNR/SSIM。当前保留原主模型，另发布显式候选入口。PLY只是第23时刻快照，真实时间播放在独立视频。背景未重新优化，脚边缺口仍待处理。\n证据：'+cloth+'/matched-training/root-review.json\n'+cloth+'/review/result.json\n'+surface+'/staged-controls/root-review.json';
s53.speakerNotes.text='本地观测衣物初始化与时间形变试验。主人体191595点、主轮18000步、原动态、相机和258633点背景保持不变。新增30592个真实观测支持的衣物高斯，总场景480820点。源场景曾用本视频51张图训练，当前59张均参与训练，没有独立测试。\n筛选至少40个有效视图，身体掩膜比例不少于0.85，头部比例低于0.02。排除接触头映射的面片及腕手指LBS影响达到0.2的点。高度超过地面0.12模型米，中心距面片中心小于0.18模型米，最大sigma小于0.04模型米，opacity大于0.05。\n参考时刻23：局部中心等于面片旋转的逆乘世界中心偏移，再除面片尺度。局部尺度除以面片尺度，局部四元数左乘面片四元数的逆。运动时再恢复世界参数。该三角面片绑定复用GaussianAvatars原代码。\n静态衣物和独立动态衣物使用相同点、相同顺序，各15轮59图，共1770次更新。后者复用ENeRF三次时间128节点图，系数归零初始化。原人体运动不变。\n仅更新衣物XYZ、DC、opacity和scale。学习率分别0.00035、0.0015、0.01、0.0006，运动0.001，590步后乘0.3。位置每坐标限4模型毫米，尺度0.8至1.25倍，SH非DC与局部旋转冻结。\n损失0.8L1+0.2DSSIM+0.25头部保护+0.05轮廓与参数锚定。独立运动增加0.005局部刚性与0.001系数正则。身体L1：原0.0243608，静态0.0141240，动态0.0140733。低覆盖比例2.47%降至0.58%。动态对静态的差异约0.36%，不能据此断言优势。\n头像参数不变，不做眼镜专项。当前为局部组合，未完整复现OmniRe。来源：观测衣物试验的protocol.json、result.json和原生导出核验，均在本地observed-clothing-pilot目录。实际对照：http://127.0.0.1:8870/body-dynamic.html#clothing';
s54.speakerNotes.text='头像参数保持。未知俯仰没有真实观测，头顶碎片与脸部伪影仍在。衣物局部改善不能视为完整模型通过。两组衣物结果接近，后续依据连续视角和时间视频决定是否采用复杂度更高的动态。\n表面2DGS分支已完成325次单参数替换和三组各885步弱约束，没有取得明确新视角突破。失败几何断点已在保留精简证据后删除。\n主18k模型保留，衣物另列本地候选。PLY为时刻23静态快照，时间播放单独提供。背景未重新优化，脚边缺口仍在。未部署00。\n来源：observed-clothing-pilot/matched-training/root-review.json、review/result.json，以及surface-gaussian-pilot/staged-controls/root-review.json。';
for(const r of rows)if(r.kind==='textbox'&&r.slideIndex!==52&&/^\d+\s*\/\s*53$/.test(r.text))p.resolve(r.id).text=r.text.replace(/53$/,'54');
await fs.mkdir('clothing-renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile('clothing-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
}
const inspect=await p.inspect({kind:'slide,textbox,image,shape',maxChars:1000000});await fs.writeFile('clothing-final-inspection.ndjson',inspect.ndjson);
const montage=await p.export({format:'png',montage:true,scale:.22});await fs.writeFile('clothing-montage.png',new Uint8Array(await montage.arrayBuffer()));
const candidatePath=path.join(build,'clothing-candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath:path.join(workspace,'shuolee-slide-output/02_有限视角重建优化路线_20261022_衣物迭代.pptx'),
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:54,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:sha},
 verifyArtifactToolImport:true,receiptPath:path.join(build,'clothing-validation.json')});
console.log(JSON.stringify(result));

