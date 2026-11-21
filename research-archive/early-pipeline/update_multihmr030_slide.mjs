import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
process.env.RUNTIME_NODE_MODULES='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const workspace='C:/Users/Administrator/Documents/New project';
const build=workspace+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const phase=base+'/current/multihmr-camera-prior-030';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const expected='833baa3da9b4dec1c28576eec57708febf8d4e04f860d6aa7f8d8536f903c87f';
const hash=async p=>crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
if(await hash(source)!==expected)throw Error('Source changed');
const conclusion=JSON.parse(await fs.readFile(phase+'/phase-conclusion.json','utf8'));
if(conclusion.state!=='camera_aware_prior_not_adopted')throw Error('Unreviewed result');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==55)throw Error('Slide count changed');
const r=await p.inspect({select:'$.slides[54]',include:['id','elements.id','elements.kind','elements.text','notes.text']});
const row=r.records[0],slide=p.resolve(row.id);
const replace=(before,after)=>{
 const el=row.elements.find(x=>x.text===before);if(!el)throw Error('Missing text anchor '+before);
 p.resolve(el.id).text.replace(before,after);
};
replace('人体运动约束与几何检查','整体配准与人体姿态');
replace('真实纹理约束跨帧运动，固定世界相机检查身体形态。','固定体形与关节，只调整整体旋转和平移，检查摆放误差。');
replace('左：实拍；中：接触约束；右：加入纹理约束。白圈为观测，红叉为投影。','左组：原始预测。右组：整体配准后。每组依次为实拍、GVHMR、Multi-HMR。');
replace('纹理残差：足部 30.65 降至 1.36 像素，躯干与腿部 24.23 降至 2.97 像素。\n二维观测重加权未修正侧面形态，下一步比较独立的人体预测。',
 '整体配准减轻脚部悬空，局部屈膝与手臂偏差仍在。\n新的身体先验尚未通过，几何核验通过后再训练高斯。');
const im=row.elements.find(x=>x.kind==='image');if(!im)throw Error('Missing image');
const image=p.resolve(im.id);
image.replace({blob:await fs.readFile(phase+'/placement-comparison.jpg'),contentType:'image/jpeg',fit:'contain',alt:'实际23帧：左为原始预测，右为固定关节体形的整体配准；两组均依次为原图、GVHMR、Multi-HMR'});
const width=1746/796*382;
image.frame={left:(1280-width)/2,top:174,width,height:382};
let notes=row.notes.text;
notes=notes.replace('官方权重需要Hugging Face账号访问批准，当前未下载、未运行此素材。','官方权重需要Hugging Face账号访问批准，尚未运行此人物的SAM预测。027已验证四个公开样例的MHR到SMPL-X转换，028已检查六张本地原图的真实内参、遮罩与射线预处理。');
const section=`十一、相机条件先验与整体配准
029与030独立比较了HMR2.0a和Multi-HMR，不从旧人物拟合初始化。HMR2在23帧出现更明显屈膝，左右膝模型弯曲角45.22/49.34度，未采用。030选择Multi-HMR的原因是其将相机内参转换成射线方向，通过傅里叶特征参与人体属性预测，并直接输出SMPL-X。它并非为本素材训练过的个体模型，也不保证未知侧面正确。
Multi-HMR 896-L官方权重1,286,462,544字节，SHA256与官方HF LFS一致。六张原图分别为0/11/23/33/45/58，保持完整背景，按官方ImageOps.contain/pad放入896方形，使用ImageNet归一化。标定K由COLMAP像素边缘约定转为OpenCV像素中心，再经实际整数缩放与黑边变换A得到K'=AK。未使用默认60度视场作为真实相机。
固定官方源码及DINOv2版本，复用现有torch2.7.1/CUDA12.8以支持RTX5080。浮点32推理，DINOv2使用其原生SDPA后备路径；没有修改模型结构或训练权重。全部活动网络参数加载成功；发布权重中的16个旧输出层参数在当前官方前向中不存在，明确记录为未使用。SMPL-X许可模板单独加载，双手采用flat_hand_mean=True。多人检测通过既有红衣人物遮罩与关节点覆盖选择目标，六张均经原图检查。
Multi-HMR以头部关节为位移参考。转换到标准SMPL-X时，先计算无平移的骨盆位置，再令平移量等于预测骨盆位置减去该值。标准模型重放与原始预测的最大顶点差为4.77×10⁻⁷米；这是坐标接口一致性，不是人体重建精度。
原始预测中23和33帧的脚部明显悬空。为区分整体摆放与局部姿态，分别对GVHMR和Multi-HMR六帧做独立刚体配准：冻结所有体形、局部关节、手型和网格，只优化3维整体旋转和3维平移。以骨盆J₀为中心，顶点变为X'=R(X−J₀)+t。两者均使用相同的官方COCO17表面回归器、原始RTMW观测及既有保守遮挡权重，排除髋部观测并降低面部权重。没有加载旧个体优化状态。
目标为归一化重投影残差的soft-L1损失，像素尺度10；scipy least_squares每个问题最多200次函数求值，旋转增量各分量限制±45度。共12个独立问题，报告126次函数求值。关节间距离最大变化小于10⁻¹⁰米，证明局部几何没有被修改。这不是共享世界配准，也没有优化统一体形或相机。
23帧Multi-HMR可见点中位残差83.60降至16.66像素，GVHMR25.16降至23.51像素。图像显示脚部悬空随整体配准减轻，但Multi-HMR的膝部形态、手臂及遮挡侧仍未通过。它的23帧左右膝弯曲为21.60/12.20度，GVHMR为13.69/10.53度；均为模型角度，无关节真值。不同方法对应的残差不是三维准确率，裸人体也不应完全填满衣物轮廓。相机相对侧视不能作为世界重力方向。
该候选不采用，0次高斯更新、不替换当前查看器、不部署00。下一步检查原始稳定三维纹理对绝对表面位置的支持；衣物表面与裸人体骨骼必须区分。SAM授权权重到位后再做独立原始预测。Multi-HMR的NAVER许可限制商业用途，本次仅作本地研究，已有其他模型商业授权不自动覆盖它。
资料：https://github.com/naver/multi-hmr/tree/651fb411e1cbcc626aaa5f38805ecab9cc891f7a ；https://huggingface.co/naver/multiHMR_896_L/tree/b315f5490ea0e174a0c79d4b80a227af9740b971 。证据：${phase}/phase-conclusion.json、protocol.json、root-only/diagnostic.json。图为本地原图与真实网格渲染，无生成图像。

十二、目前解决的问题与不足`;
if(!notes.includes('十一、目前解决的问题与不足'))throw Error('Notes anchor missing');
const actualNfev=conclusion.diagnostic.total_scipy_reported_function_evaluations;
notes=notes.replace('十一、目前解决的问题与不足',section.replace('报告126次函数求值',`报告${actualNfev}次函数求值`));
slide.speakerNotes.text=notes;
const draft=build+'/multihmr030-candidate.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const final=workspace+'/shuolee-slide-output/02_有限视角重建优化路线_身体先验配准_定稿.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:workspace,candidatePath:draft,finalPath:final,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:55,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/multihmr030-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(final));
const renders=build+'/multihmr030-final-renders';await fs.mkdir(renders,{recursive:true});
let same=0;
for(let i=0;i<55;i++){
 const name=String(i+1).padStart(2,'0')+'.png';
 const png=await q.export({slide:q.slides.items[i],format:'png',scale:1});
 await fs.writeFile(renders+'/'+name,new Uint8Array(await png.arrayBuffer()));
 if(i<54&&await hash(renders+'/'+name)===await hash(build+'/observation026-final-renders/'+name))same++;
}
if(same!==54)throw Error('Unrelated slide render changed: '+same+'/54');
await fs.writeFile(build+'/multihmr030-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath:final,
 final_sha256:await hash(final),changed_slides:[55],all_slides_rendered:55,unchanged_first54_render_hashes:same,
 canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath:final,unchanged_first54:same}));
