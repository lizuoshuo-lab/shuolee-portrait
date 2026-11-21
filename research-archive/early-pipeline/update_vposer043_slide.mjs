import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project',build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/smplx-nlf-vposer-soft-fresh-043';
const contact=base+'/current/smplx-nlf-vposer-compatibility-042';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='1c9c2169126570254e14999ed259f494fd47b02263519178e4a705a68aead06c';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical changed');
const evidence=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(evidence.state!=='soft_vposer_no_clear_alignment_gain')throw Error('Review required');
const previous=JSON.parse(await fs.readFile(build+'/contact-shape041-deck-receipt.json','utf8'));
if(previous.final_sha256!==expected)throw Error('Source provenance mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position','notes.text']});
const row=snap.records[0],slide=p.resolve(row.id);
const beforePNG=new Uint8Array(await(await p.export({slide,format:'png',scale:1})).arrayBuffer());
if(digest(beforePNG)!==previous.renders[54].render_sha256)throw Error('Unexpected source appearance');
const replace=(before,after)=>{const e=row.elements.find(x=>x.text===before);if(!e)throw Error('Missing text '+before);p.resolve(e.id).text.replace(before,after);};
replace('身体拟合中的体形与接触约束','身体姿态先验的适配与软约束');
replace('体形范围限制身体的变化，地面接触限制脚底的空间位置。','先核查编码解码对关节的影响，再用原图约束个体姿态。');
replace('左组：实拍、体形范围±1、±3。右组：同一世界侧面相机。','左组：实拍、原拟合目标、增加软约束。右组：同一侧面相机。');
replace('扩大体形范围改变胸腹轮廓，整体配准仍待解决。\n取消接触虽降低二维残差，却增大相对地面的偏离。',
        '软约束改善姿态表示的一致性，整体对齐仍未通过。\n两组从头拟合，图像关节残差均约 9 像素。');
const images=row.elements.filter(x=>x.kind==='image').sort((a,b)=>a.position.left-b.position.left);
if(images.length!==2)throw Error('Expected two images');
for(const [i,file] of ['source-000045.jpg','fixed-left90.jpg'].entries()){
  const im=p.resolve(images[i].id);
  im.replace({blob:await fs.readFile(phase+'/review/'+file),contentType:'image/jpeg',fit:'contain',
              alt:i?'固定23帧同一侧面相机，对照与VPoser软约束':'45帧实拍及不加VPoser项、增加软约束的两组从头拟合'});
  im.crop={left:0,top:0,right:0,bottom:0};
}
slide.speakerNotes.text=`一、姿态先验的输入和输出
VPoser是一个变分自编码器形式的人体姿态先验。当前使用human_body_prior原生V02_05模型，输入为21个局部身体关节的63维轴角，编码器输出32维正态潜分布的均值与标准差。解码器先生成6D旋转表示，再构造合法的三维旋转矩阵，最后转换为轴角。根旋转、根平移、整体尺度、体形和手指由身体拟合另行估计。
训练先验学习常见姿态分布。本地实验冻结全部网络权重，仅使用编码器与解码器进行个体姿态约束，没有重新训练VPoser。均值编码记为z₀=μϕ(θ₀)，第j个关节的解码旋转记为Dⱼ(z)。潜变量范数或编码重建残差不能作为真实姿态准确率。

二、为什么先检查编码解码
一个网络能够成功加载并输出合法旋转，不表示它在当前姿态上能够无损重建。042对全部59帧原始NLF姿态执行确定性的编码均值和解码，没有随机采样、身体优化或高斯训练。
两种姿态使用完全相同的原始根变换、尺度、59个真实相机、许可SMPLX_MALE模板、零体形及HaMeR局部手指。这样原图投影的变化来自身体姿态编码，而不是另一次配准、换模板或相机变化。原始NLF预测仅用作输入，历史个体拟合不参与初始化。
局部旋转变化中位5.443°、95%分位13.823°，左右肘变化中位11.976°/15.344°，局部顶点位移中位15.478毫米。可见关节投影残差中位21.635变24.097像素，95%分位51.430变79.016像素。左右手腕残差中位46.800/39.127变69.443/66.944像素。固定原相机的23/45帧叠图与59帧联系表能看到手部位置偏移，侧面站姿没有明显修复。
这个诊断支持不直接用解码输出替换原始身体姿态。它并不证明VPoser完全无用，也不证明原始NLF就是真实身体。

三、保持原始身体起点的软耦合
043显式优化身体关节旋转Rₜⱼ，同时给每一帧引入一个32维潜变量zₜ。身体仍从原始NLF姿态开始，zₜ从原始姿态的编码均值开始。新损失只要求两种姿态表示相互接近，允许图像观测使身体偏离先验。
定义旋转矩阵弦距离平方dₜⱼ²=‖Rₜⱼ−Dⱼ(zₜ)‖F²/2。它与相对旋转角α满足d²=4sin²(α/2)，小角度时d近似α。其数值无量纲，不能直接把0.20叫作严格角度截断。
令ρ(r)=√(1+r²)−1，Lpose=(1/TJ)Σₜⱼρ(dₜⱼ/σR)+γ/(32T)Σₜ‖zₜ‖₂²。T=59，J=21，σR=0.20，γ=0.01。总目标为原有图像与几何目标加λₖLpose。λₖ在前300步为0，第300到500步线性渐增到0.10，之后保持0.10。
这是本地实验中的软约束，不是原论文完整SMPLify-X拟合实现，也没有给它附加官方最优参数的含义。解码器固定，潜变量与显式姿态独立优化。相比硬性令R=D(z)，这种方法避免编码重建误差直接改变初始身体位置。

四、输入、自由度与优化日程
两组分别从原始59帧去畸变图、相机、遮罩、原始NLF预测、许可模板和原始HaMeR结果开始。新Adam、新零体形，没有读取历史个体网格、高斯、形变或优化器。两组真实起点的身体姿态、根、尺度、体形与潜变量逐数组相同。
每组1200步。前300步拟合根旋转、平移和共享尺度，300步后开放共享关节、体形与潜变量，700步后开放逐帧身体修正，1000步后学习率乘0.4。根组学习率0.008，共享姿态/体形0.006，逐帧姿态和潜变量0.004，梯度范数上限10。
局部姿态通过SO(3)左乘组合，共享每轴修正界限0.30弧度，逐帧0.10弧度。根每轴0.35弧度，根位移相对于原尺度每轴0.30模型米。共享尺度为s₀exp(0.30tanh(q))。10维体形从0开始，以tanh约束到±1，体形损失0.04mean(β²)。这些都是有界本地试验设置，不是独立测得的身体参数范围。

五、保留的观测和几何约束
原有身体、脚、手二维稳健尺度20、18、10像素，权重1、0.2、0.5。使用COCO17官方表面回归点与原始检测比较，保留既有遮挡剔除规则。HaMeR只提供局部手指旋转，先减去当前模板手均值，不复制裁切相机中的根旋转。
保持人物遮罩包含、原始局部关节/旋转先验、时序平滑、正深度、绝对脚底高度、对应脚底顶点相对稳定与鞋纹相对运动。裸人体不强行填满外套、裤子和鞋。
NLF的1024个非参数表面点转换到去根、以骨盆为原点的局部坐标，约束同索引SMPL-X顶点。尺度0.06模型米，不确定性权重σ的负1.5次幂逐帧归一化为均值1。权重1，第300到500步渐增。原始预测具有偏差，不作为测量真值。
两组完全保留上述约束，只改变VPoser软耦合总权重0和0.10。对照也计算先验诊断，但其潜变量不更新。本轮没有额外权重扫描。

六、实际结果与图像解释
当前页左图为45帧实拍与两种从头拟合，右图为固定23帧、同一世界侧面相机。全部59帧联系表以及23/45源图、正侧面和上下25度均已查看。身体外观与站姿没有明显整体改善。
可见关节二维残差中位9.007/9.070像素，95%分位34.200/34.559像素。未优化鞋纹诊断中位1.393/1.385像素。固定23帧左右膝模型弯曲14.03/9.42变14.47/9.60度。膝角是模型量，没有独立解剖真值。
与各组当前潜变量解码结果的旋转差中位10.862/5.282度，但软组潜变量也在联合优化，因此这项损失的下降不能单独证明姿态更准确。为了避免只报告优化目标，把最终姿态重新经同一个固定编码器/解码器处理，重建旋转差中位9.361/7.220度，95%分位22.142/16.808度。该结果说明更接近先验的姿态表示，仍不是原图或几何准确度。
联合优化潜变量均方从初始0.397到1.515，原生均值重编码的均方在两组为0.578/0.581。不能把优化潜变量范数当成已校准的人体合理性评分。

七、解决范围和不足
软耦合避免了直接解码替换造成的初始位置跳变，并让姿态更接近先验表示。当前素材的整体身体配准仍未通过。原生模板重放最大差为0、姿态与潜变量梯度的有限差分核验，只说明实现和导出一致。
没有新高斯更新、查看器替换或00部署，也未验收连续动画。单目先验不能测得宽松衣物下的真实肘膝或遮挡侧厚度。继续加大先验权重可能只是更强地服从先验，没有证据支持这么做。
下一步复用既有检测器差异和可见性证据，检查可见手腕/皮肤边界与颈部的实际图像对应，区分表面回归点、骨架关节和检测语义。先建立可复核观测，再从头拟合。现有VPoser、GVHMR、NLF继续可用，SAM3DBody下载暂缓。整体几何通过后才用20万新高斯训练18000步。

八、来源
VPoser官方模型代码：https://github.com/nghorbani/human_body_prior ，本机代码提交6b883ecc3000697409a8caa2744523afe041e48a。现有V02_05权重来自公开镜像并与LFS哈希一致，官方权重哈希未取得，不能声称官方哈希确认。严格加载且网络参数冻结。
NLF：https://github.com/isarandi/nlf 。SMPL-X：https://github.com/vchoutas/smplx 。模板由用户提供。NLF默认权重许可用于非商业研究，本轮仅本地验证。
本地编码诊断：${contact}/audit.json、phase-conclusion.json。软耦合：${phase}/control/protocol.json、soft_prior/protocol.json、review/comparison.json、phase-conclusion.json。所有图片均由真实原图和实际网格生成，没有图像生成补全。`;
await fs.writeFile(build+'/vposer043-edited-layout.json',JSON.stringify((await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','elements.position']})).records,null,2));
const draft=build+'/vposer043-candidate.pptx';await(await PresentationFile.exportPptx(p)).save(draft);
const finalPath=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_姿态先验_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath,
  pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
  integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
  fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
  verifyArtifactToolImport:true,receiptPath:build+'/vposer043-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const renderdir=build+'/vposer043-final-renders';await fs.mkdir(renderdir,{recursive:true});const checks=[];
for(let i=0;i<55;i++){
  const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer());
  await fs.writeFile(renderdir+'/'+String(i+1).padStart(2,'0')+'.png',bytes);
  const renderHash=digest(bytes),unchanged=renderHash===previous.renders[i].render_sha256;
  if(i<54&&!unchanged)throw Error('Unintended change '+(i+1));checks.push({slide:i+1,render_sha256:renderHash,unchanged});
}
await fs.writeFile(build+'/vposer043-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),
  changed_slides:[55],changed_notes_slides:[55],all_slides_rendered:55,renders:checks,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,unchanged_first54:true,review_required:renderdir+'/55.png'}));
