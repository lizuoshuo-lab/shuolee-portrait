import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-nlf-template-fresh-038';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='f56e3022cfc8eb5f1515f64e922289e0b7225fa23bae801ac3fabae3cab0b1b5';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='template_family_effect_verified_alignment_not_accepted')throw Error('Review required');
const previous=JSON.parse(await fs.readFile(build+'/nlf037-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const beforePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(beforePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{const e=row.elements.find(x=>x.text===before);if(!e)throw Error('Missing text '+before);p.resolve(e.id).text.replace(before,after);};
replace('身体表面先验与姿态','人体模板与共享体形');
replace('对应表面点与不确定性，约束共享体形和逐帧姿态。','模板定义形状空间，真实相机与表面观测约束人物对齐。');
replace('左组：实拍、关节先验、关节与表面先验。右组：相同侧面相机下的两种身体。',
 '左组：实拍、中性模板、男性模板。右组：同一世界侧面相机。');
replace('与先验的表面距离由 31.75 降至 23.66 毫米，不是人体重建精度。\n预测一致性改善，整体几何仍未通过。',
 '胸腹轮廓随模板明显变化，肢体对齐仍未通过。\n两组从零体形拟合；二维残差为 8.18 与 9.01 像素。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two images');
for(const [i,file] of ['source-000045.jpg','fixed-left90.jpg'].entries()){
 const im=p.resolve(images[i].id);
 im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',
  alt:i?'固定23帧同一世界侧面：中性模板与男性模板':'45帧原图与两种模板从零体形拟合'});
 im.crop={left:0,top:0,right:0,bottom:0};
}
slide.speakerNotes.text=`一、模板与身体先验的职责
SMPL-X模板包含平均人体、形状基、姿态修正基、关节回归与蒙皮权重。图像网络则估计人物的形状、姿态与相机相关位置。两者必须匹配；模板变了，不等于给同一个身体换了文件名。模板族只是参数空间名称，不用于推断身份属性。
姿态变形前，T_g(β)=T̄_g+Σ_k β_k S_(g,k)。g表示模板族，T̄是平均身体，S是形状基，β是体形系数。之后加入姿态修正B_P(θ)，通过线性混合蒙皮W得到V_g(β,θ)。这次完整更换平均身体、形状基、关节回归和蒙皮，不声称只改变平均外形。NEUTRAL与MALE的体形系数不能直接互用。

二、已有模型包与接口核对
使用用户已经下载并许可的models_smplx_v1_1.zip，无新权重下载。包内中性模板与工程SMPLX_NEUTRAL.npz散列一致：376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992。没有发现文件版本错配。男性模板散列ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5。
两者均有10475顶点、20908三角形、55骨骼；面索引和骨骼层级一致。形状基有400个分量，本地使用10个体形、10个表情分量。均值顶点最大坐标差约0.0578模型米，前10维形状基、关节回归和蒙皮权重不同，因此两组均将体形置零。原生模板重放两组最终网格的最大差均为0；此结果只验证参数接口。

三、原始证据与世界坐标
输入59张顺序视频标定图，非59台同步相机。复用原图、原遮罩、原始检测、许可模板和官方NLF原始预测；不读取旧人物拟合、高斯、时间形变或优化器作为起点。NLF的10475个参数化表面点和1024个非参数化对应点以相机毫米输出，转换为模型米。原始预测在中性模板的重放误差约8.34×10⁻⁷米。
原相机固定。将官方全局旋转搬入标定世界坐标，使用近似静止骨盆估计一个公共尺度和逐帧平移。两组共同原始观测和根初始化逐数组相同。该步骤不是身体测量；世界尺度仍是模型米到SfM任意单位的换算。原始NLF根旋转偏离序列平均的中位角约2.11°、95%分位4.00°，说明没有明显的大幅方向跳变；不能由此排除真实运动或先验偏差。

四、匹配实验中的变量与阶段
两组都从β=0、新Adam优化器独立开始，各1200次身体更新，共2400次，0次高斯更新。唯一模型条件差异为完整模板族；相同的原始NLF关节、局部旋转、非参数化点、相机、手部、可见性和目标都保留。它不是038接着旧036或037训练。
1—300步只更新根旋转、平移和尺度；301—700步开放共享关节与体形；701—1200步开放有界逐帧关节修正。共享关节每轴界0.30弧度，逐帧0.10弧度；左乘SO(3)旋转，非直接相加轴角。β每维在−1至1之间，所有时刻共享。手指使用原始HaMeR局部旋转并减去相应模板手均值，不移植裁切相机的手腕根旋转。

五、目标函数与表面点
身体、脚和手的二维稳健残差尺度分别20、18、10像素，脚权重0.2、手权重0.5并在300—500步渐增。保留裸人体不穿出衣物的单向轮廓项、原始局部关节与旋转先验、β正则、估计地面上方1—7厘米脚底软约束、对应脚底顶点稳定及相邻时刻平滑。原鞋和裤脚纹理相对运动项权重0.3，在100—300步渐增；未优化轨迹仅作诊断。
1024个非参数化对应点先减官方骨盆并逆转官方根朝向。局部人体顶点同样减本帧骨盆。L_surface=(3TN)⁻¹ΣₜΣᵢwₜᵢΣₖρ((vₜᵢₖ−pₜᵢₖ)/τ)，T=59，N=1024，τ=0.06模型米，ρ(r)=√(1+r²)−1。w由网络不确定性的−1.5次幂得到，逐帧归一化到均值1；不是已校准概率。两组表面项权重都在300—500步由0增至1。坐标、有限非零梯度和有限差分检查通过。
此前037在同一中性初始化下只增加表面项，把与预测的中位距离31.75降至23.66模型毫米，整体几何却未明显改善。这说明拉近一个有偏先验并不自动获得更准确人体，不能把先验距离解释为扫描误差。

六、此次能够解释的问题
在同一23帧、同一世界相机和显示比例下，男性模板明显改变中性模板的胸部突出与腰臀轮廓；正面、两侧及俯仰都能看到这一差异。因此身体外形问题需要单独核查模板空间，不能全部归因于姿态优化或初始化高斯点数。平均模板、体形变化和局部姿态共同决定结果，仅凭一个二维残差无法选出最准确身体。
本页左图是45帧原图及两组真实网格叠加，右图是23帧固定侧面。已查看全部59帧联系表、六个源视角合图、45帧全尺寸叠图，以及固定正面、左右90°、上下25°。未完成连续视频验收，不声称每帧每个细节都有独立真值。

七、未解决问题与后续判据
中性/男性两组可见二维关节残差中位数8.177/9.007像素，95%分位31.260/34.200像素。男性组左右肘残差30.47/22.80像素，手臂和部分肢体对齐仍有明显偏差。未优化鞋/裤脚诊断中位数1.442/1.393像素。对NLF表面预测的中位距离23.828/23.730模型毫米，躯干20.760/23.296毫米；这不是人体重建准确率。
固定23帧左右膝模型弯曲从11.47/6.23变为14.03/9.42度，缺少关节真值，不能假定腿越直越准确。裸人体无需填满宽松衣物。两种模型都不作为已通过几何的高斯底模，不替换查看器，不部署00。未通过模型与重复初始化在证据核验后清理。
下一步检查NLF对应表面点的二维投影在真实相机下的可见性和跨视角一致性，区分网络深度与图像对应。先核对遮挡筛选、像素中心和梯度，再从相同原始输入、零体形、新优化器比较三维表面项与可见二维表面项。男性模板仅作已核验候选，不继承此次个体参数。SAM3DBody暂缓，继续用已下载的VPoser、GVHMR、NLF。几何通过后才从20万新高斯、新优化器训练18000步。

八、来源与证据
NLF官方源码 https://github.com/isarandi/nlf ，NeurIPS2024；已核验v0.3.2模型。其官方模型用于非商业研究，本次仅本地研究。SMPL-X官方实现 https://github.com/vchoutas/smplx ，模板来自用户已下载模型包。
本页为本地受控扩展，不是缺失的GaussianAvatars_SMPLX官方完整训练入口复现。所有照片与网格渲染来自真实输入和本次数值模型，无生成补全。证据：${phase}/template-audit.json、neutral/protocol.json、male/protocol.json、review/comparison.json、phase-conclusion.json。`;
const after=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']});
await fs.writeFile(build+'/template038-edited-layout.json',JSON.stringify(after.records,null,2));
const draft=build+'/template038-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_身体模板_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/template038-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/template038-final-renders';await fs.mkdir(renderdir,{recursive:true});
const checks=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
 const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
 if(i<54&&!unchanged)throw Error('Unintended change '+(i+1));
 checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/template038-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,
 renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,unchanged_first54:true,review_required:renderdir+'/55.png'}));
