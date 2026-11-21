import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const build=process.cwd(),workspace='C:/Users/Administrator/Documents/New project';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const g=base+'/current/novel-view-face/head-structure/flame-photometric-registration/whole-body-gaussianavatar/all-views-background/ground-support-pilot';
const sourceOriginal=base+'/reports/02_有限视角重建优化路线.pptx';
const buffer=await fs.readFile(sourceOriginal),sha=crypto.createHash('sha256').update(buffer).digest('hex');
if(sha!=='7342d60a349f4cf958043deb0cd7f1d43567249fcfa65aa764a8406b6b5d3114')throw Error('Source changed; preserve user edits');
const source=path.join(build,'source-before-ground.pptx');await fs.writeFile(source,buffer);
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({kind:'slide,textbox,image,shape',maxChars:1000000});
await fs.writeFile('ground-source-inspection.ndjson',snap.ndjson);
const rows=snap.ndjson.trim().split('\n').map(JSON.parse);
function text(s,name,value,left,top,width,height,size=26,bold=false,color='#183342'){
 const sh=s.shapes.add({geometry:'textbox',name,position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 sh.text=value;sh.text.style={fontSize:size,typeface:'微软雅黑',color,bold};sh.text.insets=0;return sh;
}
const s=p.resolve(rows.find(r=>r.kind==='slide'&&r.slideIndex===51).id);
p.delete(rows.filter(r=>r.slideIndex===51&&['textbox','shape','image'].includes(r.kind)).map(r=>r.id));
text(s,'title','按合成贡献恢复静态地面',66,42,1148,67,42,true);
text(s,'method','从原场景恢复被人物分离误删的地面高斯。\n统计真实地面像素的合成贡献，保留原人体与衣物的时间形变。',66,122,1148,78,25);
text(s,'weight','合成权重  wᵢ(p) = Tᵢ(p) αᵢ(p)',66,215,1148,47,29);
text(s,'criteria','地面权重占比超过 98%，至少 6 个观测有支持；再检查位置与尺度。\n恢复 9,733 个原有高斯，沿用原参数，无新增训练。',66,270,1148,70,23);
s.images.add({blob:new Uint8Array(await fs.readFile(g+'/review/000023-ground.png')),contentType:'image/png',alt:'同一实拍脚边、恢复前和恢复9733个原有高斯的原生渲染对照',fit:'contain',position:{left:66,top:345,width:1148,height:249}});
text(s,'result','脚边地面训练 L1：0.09918 → 0.07592，下降 23.45%。',66,610,1148,40,25);
text(s,'source','同一正面相机 · 全部 59 个观测均为训练资料 · 原生渲染',66,665,1040,26,17,false,'#536674');
text(s,'page-number','52 / 53',1130,679,86,20,14,false,'#536674');
s.speakerNotes.text=`本页为实际原生高斯渲染与方法说明。沿用人体191595点、18,000步完成的128节点三次时间形变，以及独立衣物30592点、885步候选。两个运动分支与原人物高斯参数保持不变，背景仍为静态。
原因：原人物分离按中心投影在人物掩膜中的支持删除背景点。中心落入人物，不代表高斯只渲染人物；其屏幕足迹仍可能给脚边地毯提供颜色。原高度保护带也与学习出的地毯中心存在偏差。本轮从原场景已删除集合恢复，而非生成新点或修改地毯颜色。
前后合成满足 C(p)=Σᵢ wᵢ(p)cᵢ，wᵢ(p)=Tᵢ(p)αᵢ(p)，Tᵢ(p)=∏ⱼ<ᵢ[1−αⱼ(p)]。固定位置、旋转、尺度和不透明度，渲染颜色的导数给出各高斯对指定像素区域的实际合成权重。分别对地面区域F和人物内部H累加，得到 Wᶠᵢ=ΣᵥΣₚ∈Fᵥ wᵢᵥ(p) 与 Wʰᵢ=ΣᵥΣₚ∈Hᵥ wᵢᵥ(p)。使用 Wᶠᵢ/(Wᶠᵢ+Wʰᵢ)>0.98 的保守条件，避免仅用高斯中心投影归属。
区域：原生1514×2694相机，像素边界K保持不变，像素采样坐标为索引加0.5。脚底附近地面圆盘半径0.65模型米，地面排除11×11膨胀的人物掩膜，人物内部使用3×3腐蚀掩膜。该圆盘和高度只定义诊断区域，不是真值地面。
候选来自原316044点场景中被分离删除的57411点。世界高度相对先前近似地面限定为−0.20至0.12模型米，脚部水平距离小于0.80模型米，最大标准差小于0.08模型米。累计地面贡献超过3像素等效面积，至少6个视角的地面贡献超过0.05，最终9733点。占比0.90的较宽条件得到10187点，只作为诊断对照，没有采用。
原地面高度保护带为−0.06至0.045模型米，仅恢复42点。本次9733点中99.21%超出旧带；其旧平面相对高度中位数为79.04模型毫米。稳健拟合后的地面法向与旧方向相差1.42度，8073个内点距拟合面的95百分位约5.06模型毫米。这些是学习高斯的自洽性诊断，未经实物测量，不作独立几何真值。
直接复制恢复点的原位置、完整SH、协方差与透明度，新增优化更新为0。背景258633增加到268366点；人体191595、衣物30592，共490553点。时刻23导出静态PLY，真实时间播放在网页另存，不把PLY称为动态模型。
全部59张均为训练观测。脚边地面平均RGB L1：恢复前0.09918237，保守0.07592306，较宽0.07345101。地面alpha<0.5比例：3.7074%、0.1395%、0.0504%。较宽条件数值更好，但仍采用保守集合。人物下部L1由0.01441982到0.01435057，没有用此微小差异宣称人体质量改善。
固定相机59时刻，固定人物时刻23的俯仰±12度与侧向±25度，均由同一原生渲染器直接生成。实拍图与虚拟视角分开，后者没有真实对照。整体限制统一见最后一页。
本地来源：${g}/preflight/result.json；${g}/review/result.json；${g}/root-review.json。网页：http://127.0.0.1:8870/body-dynamic.html#ground 。合成公式背景：Kerbl等3D Gaussian Splatting，https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ 。本轮是已有动态框架的局部恢复，不是完整OmniRe官方复现。`;
const old='未知俯仰的头部碎片与脚边背景缺口仍在。静态和动态衣物效果接近，整体质量尚未通过。';
const remaining=rows.find(r=>r.kind==='textbox'&&r.slideIndex===52&&r.text===old);
if(!remaining)throw Error('Conclusion anchor missing');
p.resolve(remaining.id).text='衣物与地面局部改善；头部与鞋底边缘仍有伪影。静态和动态衣物差异小，整体尚未通过。';
const last=p.resolve(rows.find(r=>r.kind==='slide'&&r.slideIndex===52).id);
last.speakerNotes.text=String(last.speakerNotes.text)+'\n最新背景增量：第52页恢复9733个旧场景地面点，背景总数268366，当前候选总数490553。此页衣物图为恢复地面前的单独人体对照，衣物参数不变。局部地面改善不代表头部、接触阴影或整体几何已通过。孤立恢复点仍有暗色阴影与杂散外观，完整渲染未见明确第二双鞋，鞋底白边仍有残留。全部为同视频训练观测，未知俯仰没有GT；尚未部署00。完整Gaussian Haircut未复现，方向损失的2×885步组件对照未改善冠部，已停止并清理失败断点。后续先核验头皮与显式发束的真实多视图证据，眼镜专项停止。';
await fs.mkdir('ground-final-renders',{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile('ground-final-renders/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));
 if((i+1)%10===0)console.log('Rendered '+(i+1)+'/53');
}
const montage=await p.export({format:'png',montage:true,scale:.22});await fs.writeFile('ground-final53-montage.png',new Uint8Array(await montage.arrayBuffer()));
const candidatePath=path.join(build,'ground-final-candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const result=await finalizePresentation({workspaceDir:workspace,candidatePath,finalPath:path.join(workspace,'shuolee-slide-output/02_有限视角重建优化路线_20261022_地面修订.pptx'),
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:53,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:sha},verifyArtifactToolImport:true,receiptPath:path.join(build,'ground-final-validation.json')});
console.log(JSON.stringify(result));
