import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-nlf-shoe-motion-fresh-036';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='edb8540910df900a24ab74f5713a0771ed84de2f863611837929aebdbb579edb';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex');
const hash=async path=>digest(await fs.readFile(path));
if(await hash(source)!==expected)throw Error('Canonical deck changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='nlf_shared_fit_reviewed_geometry_not_accepted')throw Error('Missing actual review');
const previous=JSON.parse(await fs.readFile(build+'/identity033-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const sourcePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(sourcePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{
 const item=row.elements.find(x=>x.text===before);if(!item)throw Error('Missing text '+before);
 p.resolve(item.id).text.replace(before,after);
};
replace('外表面深度与人体几何','身体先验与共享体形');
replace('原始衣物纹理提供外表面深度，约束身体沿观察射线的前后次序。',
 'NLF预测三维人体点，为共享体形与逐帧姿态提供起点。');
replace('左组：实拍、从头对照、深度约束。右组：同一世界相机下的两种身体。',
 '左组：实拍、原先验拟合、NLF拟合。右组：相同侧面相机下的两种身体。');
replace('共同诊断射线的深度冲突由 477 降至 282，侧面姿态未明显改善。\n身体体积减小约 1.95%，二维残差增加，本轮未通过几何验收。',
 '二维关节残差由 8.71 降至 8.06 像素，整体姿态仍未通过。\n下一步保留表面点与不确定性约束，再检验身体几何。');
const im=row.elements.find(x=>x.kind==='image');const image=p.resolve(im.id);
image.replace({blob:await fs.readFile(phase+'/review/source-000045.jpg'),contentType:'image/jpeg',fit:'contain',
 alt:'45帧实拍与原GVHMR先验拟合、NLF先验从头拟合的完整同相机叠图'});
image.frame={left:235,top:174,width:723/886*382,height:382};
image.crop={left:0,top:0,right:0,bottom:0};
slide.images.add({blob:await fs.readFile(phase+'/review/fixed-left90.jpg'),contentType:'image/jpeg',fit:'contain',
 alt:'固定23帧与同一世界侧面相机，左原先验拟合，右NLF先验拟合',
 position:{left:580,top:174,width:1040/826*382,height:382}});
slide.speakerNotes.text=`一、身体先验提供什么信息
VPoser V02_05描述合理人体姿态的统计分布，以32维潜变量解码21个局部身体关节旋转。它本身不读取当前照片，不能提供人物在场景中的位置。NLF则根据完整图像、目标人体框与相机内参预测三维人体点及不确定性，再将参数化身体拟合到这些预测点。本页比较既有GVHMR图像/视频先验与NLF独立图像先验提供的身体起点，不能将下载或参数转换成功称为几何正确。

二、NLF官方接口与参数
本次使用官方NLF-L multi v0.3.2 patch6模型，原始非参数化表面点1024个，SMPL-X参数化网格10475顶点，完整姿态165个轴角分量，体形10个无量纲系数。完整姿态为全局朝向、21个身体关节、下颌、左右眼、左右手各15关节。官方输出顶点以毫米计、平移以米计，接入统一换成模型米。原始1024点的顶点索引依据模板坐标、体形基与蒙皮权重共同核对，不能直接给它套10475顶点的拓扑。
官方拟合分别归一化表面点和关节点的不确定性权重。对于表面点，wᵢ=σᵢ^(−3/2)/[(1/N)Σⱼσⱼ^(−3/2)]，σᵢ为网络不确定性输出，N=1024。它是相对预测可靠性，未经本素材真实几何误差标定。官方参数化转换运行3次迭代。本轮的036后续身体优化并没有直接使用这些表面点或权重，不能称作完整官方目标的复现。

三、相机、时间与单位
原素材HOVER_VERSA_0102.mp4含59张原始标定观测，来自顺序视频的原帧1、4、…、175，不能称为59台同步相机。标定图1514×2694，焦距约1461.49077像素、主点(757,1347)，COLMAP角点原点转OpenCV像素中心时主点减0.5。相机固定，图像和相机输入有逐项散列记录。
NLF世界尺度由原相机和预测骨盆在近似静止条件下求出，每模型米约3.0783世界单位。它不是实际测量尺度。局部网格先减去骨盆，再用世界根旋转、共享尺度与根平移变换。NLF官方轴角使用绝对手指旋转，本地flat_hand_mean=True可以重放原网格，全部59帧最大坐标差小于8.35×10⁻⁷模型米。该数值验证转换一致，不代表亚微米人体精度。

四、共享体形与受限姿态
同一人物在所有时刻使用一组β，初始化取59帧官方体形系数的逐系数中位数。每帧身体初始旋转为官方NLF预测，不读取旧个体拟合、旧高斯或旧形变。局部校正通过旋转矩阵左乘组合，避免直接把不同姿态轴角相加。
本轮β=β₀+tanh(b)，每系数偏离起点最多1。身体旋转校正轴角为0.30tanh(a共享)+0.10tanh(a逐帧)，上限以每轴弧度计。根旋转校正每轴0.35弧度，根平移每轴0.30s₀，尺度为s₀exp[0.30tanh(u)]。所有优化变量与Adam状态新建，原023仅在评测时只读。
原始HaMeR观测用于局部手指与可见手点。绝对手指轴角减去模板手均值后传给flat_hand_mean=False模型。遮挡手使用最近可见的局部手姿态，未把裁切手部的绝对根朝向移植进场景。人体体形和皮肤也无需填满宽松衣物。

五、损失与训练阶段
二维鲁棒项采用ρ(r)=√(1+r²)−1。身体、脚部与手部坐标分别以20、18、10像素归一化。保持既有可见性掩码，被遮挡的对侧关节剔除，髋部COCO检测不用于该身体投影项。优先匹配官方COCO17表面回归的定义，不将同名SMPL-X内部骨架点混用。
总目标为L身体+0.2L脚+0.5r手L手+2L包含+0.15L先验+0.04L体形+L接触+0.4L足部稳定+0.05L时间平滑+0.3r纹理L纹理，并含相机前方有效深度项。手权重在300–500步线性启用，纹理权重在100–300步线性启用。人体包含是惩罚网格伸出人物掩膜的单向距离，不将衣物边缘作为皮肤表面。
先验项包含21个局部姿态校正与前22个局部骨架点，归一化分别为0.15弧度与0.06模型米。体形项为tanh(b)的系数均方。接触项把脚底约束在估计地面上方1–7模型厘米，对应脚底点的跨时刻稳定以0.025模型米归一化。鞋/裤脚纹理只约束踝坐标系的相对运动，不确定参考时刻的绝对形状。
前300步只优化根旋转、位置和尺度。301–700步开放共享姿态与体形。701–1200步允许受限逐帧姿态修正。Adam根、共享、逐帧学习率分别0.008、0.006、0.004，1000步后乘0.4。每步梯度范数上限10。本轮1200次身体更新、0次高斯更新。

六、实际对照结果
NLF原始59帧标准姿态的模型高度最小1700.97毫米、最大1791.99毫米，跨度约91.02毫米。这是逐帧预测的身份差异，不是实测人物身高变化。共享体形用于约束这种变化，但不能保证共享后的体形准确。
同相机、同59帧、同可见性和同损失预算下，原023可见关节二维残差中位数8.7126像素，036为8.0591像素。最大残差则由56.82变为67.79像素。23帧模型左右膝弯曲由12.38/6.89度变为10.72/4.97度，全序列右膝中位数由6.47增至8.66度，所以不能只挑某一时刻下结论。未优化鞋/裤脚纹理的诊断残差由1.36增至1.45像素。
所有59帧联系表、六视图源图对照、同23帧世界正侧面及上下25度均已检查。局部投影有改善，侧面屈膝、胸腹比例和手臂位置仍没有清晰整体突破。全部照片用于拟合，没有独立三维关节或皮肤表面真值，不能把残差当准确率。

七、解决的问题与当前不足
身体先验已可本地运行，真实相机内参、模板对应、坐标单位、手部均值与世界变换均已核验。用全新优化器从59帧拟合共享体形的流程已运行完成。全身几何仍未通过，不替换查看器，不部署00，不开始20万点18000步新高斯训练。
036只测试了更换初始化和关节先验，未直接约束NLF的1024表面点。10个体形系数中7个偏移超过0.85，说明优化显著改变初始体形，但不能据此确证哪个损失导致错误。下一轮使用现有先验，先检查原始表面点与参数化网格的对应残差及不确定性，再做独立从头对照。SAM3DBody按用户最新要求暂缓，不继续请求下载。

八、方法来源与证据
NLF官方源码：https://github.com/isarandi/nlf 。权重发布：https://github.com/isarandi/nlf/releases/tag/v0.3.2 。VPoser方法与软件：https://github.com/nghorbani/human_body_prior 。SMPL-X：https://github.com/vchoutas/smplx 。NLF官方权重声明非商业研究用途，本轮仅作本地研究，未把用户的其他商业授权视为该权重授权。
本页图像来自实际原图与本地网格渲染。三列叠图为原始45帧，固定侧面图为时刻23。无生成图像或人为身体转正。实验协议、完整目标、度量和判断见${phase}/protocol.json、review/comparison.json、prior-change-audit.json、phase-conclusion.json。原032/033仅保留精简对照证据，本页不再逐轮堆叠过期试验说明。`;
const after=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']});
await fs.writeFile(build+'/nlf036-edited-layout.json',JSON.stringify(after.records,null,2));
const draft=build+'/nlf036-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_NLF身体先验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nlf036-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/nlf036-final-renders';await fs.mkdir(renderdir,{recursive:true});
const checks=[];
for(let i=0;i<55;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
 await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
 const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
 if(i<54&&!unchanged)throw Error('Unintended visible slide change '+(i+1));
 checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/nlf036-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,
 final_sha256:await hash(finalPath),changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,
 renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slide:55,unchanged_first54:true,review_required:renderdir+'/55.png'}));
