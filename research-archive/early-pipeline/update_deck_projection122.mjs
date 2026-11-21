import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/face_pinhole_radial_native_six_views.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/admission122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/projection122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('材料对应与输出内参核验','残余畸变与共享三维对齐');
replace('新版原生裁片','上排针孔，下排共享径向模型。绿圈为观测，紫色为三维投影。');
replace('双向误差小','修复求解后，脸部候选误差中位数由10.13降至3.11像素。');
replace('输出K与裁剪取整','加入径向项后为3.14像素，仍未对齐。残余畸变尚未定论。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'相同原生视角的针孔与共享残余径向模型投影对照，图内无文字',fit:'contain'});
image.frame=frame;image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-27至28 N5/N6：用户要求不把去畸变输出默认为精确针孔，以近似K先推进粗诊断。未二次处理原始鱼眼畸变，未二维扭曲视频。N5c初始求解失败和N6保存失败均保留。',
 '本页六列为230、246、262、294、310、326帧，均为320x320原生裁片、等比排列，没有图中文字。上排为已修复求解器的固定K针孔对照，下排为仅增加共享径向k1的模型。绿圈为候选观察，紫十字为排除目标自身后的三维投影，橙线为误差。第278帧为版面排布省略，报告完整保留全部7帧和57脸部观测，未按误差筛选。',
 '冻结7帧原生背景特征，1125条三视循环轨迹，935FIT与190整族留出。近重复SIFT按8px邻接族隔离。只用FIT背景求相机，9条N1脸部候选从未进入相机优化。背景以近地面为主，远水面与重复纹理仍有身份歧义，不当物理真值。',
 '配对固定K PoseLib极线Sampson的792次背景留出比较最大0.704px，但不能替代共享3D欧氏重投影。首次全局SciPy稀疏求解以xtol停止，cost1289.530659、gradient155187，几乎未优化。独立解析J和数值差分相对最大3.25e-8，明确导数正确，数值停止不可信。',
 '同Huber1px、同K、同FIT、相同固定第一相机和单位第一基线，采用已验证的显式阻尼SchurIRLS修复。18个下降步，cost329.508078、gradient0.031787。681背景留出预测中位0.434743px、P90 0.969532px、最大3.199422px。57脸部候选中位3.105781px、P90 8.286258px、最大10.142233px；修复前中位10.134402px最大23.006013px。',
 'N6只添加共享输出域径向系数k1，u=c+f*[x,y]*(1+k1*r²)，K与主点固定。R/t与FIT三维点是两模型共同的优化变量。它不是原始FISHEYE_8，也不是对导出图像再次去畸变。梯度及射线正反变换四项控制通过。',
 '估计k1=0.0014931978，25点固定射线网格最大位移2.522956px；该值是候选模型影响量，不是物理标定的真实畸变幅度。背景中位0.432217px、P90 0.971649px、最大3.157540px；脸候选中位3.141976px、P90 8.232170px、最大10.060944px。没有改善脸部对齐，所有57脸候选仍均不满足小于1px。',
 '本地线性分析径向导数在消除R/t与点坐标后仅保留1.204%范数，与场景几何高度耦合。不能据此把残差唯一归为镜头，也不能因单个径向项无效就排除切向、偏心、空间非对称、K/crop问题、材料偏差或人物微动。下一步先检查空间/时间残差与深度材料支持，不自动堆高阶模型。',
 'N6首轮全部计算完成，在写JSON时NumPybool导致序列化失败，最终向量尚未落盘。必要修复只把bool转换为Pythonbool并先保存NPZ，原数据/配置/预算/种子不变。恢复运算的最终cost和k1与首轮progress逐值完全相同，失败证据保留；不能当两次候选择优。',
 'R1e真实原生UI的4次合成记录相机针孔/KB4鱼眼渲染，RGB和Alpha与冻结原生参考最大差0，实际下载SHA一致。仅合成记录相机，不代表当前MP4物理标定、任意连续视角或00部署。',
 '右侧#robust122-n6和原工作日志已更新，119历史PLY与89相机保留，没有新人像训练。验收仍要求所有独立物理可信面部点严格<1原生px，当前候选材料与定位不确定性未认证。',
 '证据：robust-portrait-122/video-update-20261027/background-geometry/comparison-review.json、shared-path/numerical-audit/result.json、shared-path/solver-repair/result.json、residual-radial/serialization-repair/result.json、comparison-per-view.json，以及native-interactive/result.json。',
 '以下历史说明保留溯源：',previous
].join('\n\n');
const candidatePath=build+'/projection122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122投影模型对照.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/projection122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_projection122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/projection122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/projection122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/projection122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide'+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/projection122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
