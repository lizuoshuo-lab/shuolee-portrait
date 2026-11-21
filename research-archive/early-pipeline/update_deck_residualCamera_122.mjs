import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-camera';
const figure=root+'/residual-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/gsplatDistortion-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.checks_passed!==26||review.human_training_started!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/residualCamera-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('gsplat 的畸变相机与训练边界','残余畸变的统一相机链路');
replace('无畸变 / 给定畸变 / 差值×10：六个已知高斯。','同一 K、外参与畸变参数贯穿投影、训练与回读。');
replace('本机已验证畸变渲染与高斯梯度；标定须独立完成。','合成训练与回读一致；实拍畸变系数仍待验证。');
replace('1.5.3 此分支不提供 K、畸变系数或外参梯度。','26项集成、11项冻结检查通过；未启动人物训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=620;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'残余畸变合成控制：独立射线、CUDA及差值；两台训练相机。原生96×80，图内无文字',fit:'contain'});
image.frame={left:330,top:190,width,height:width*160/288};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 R2 输出域残余畸变工程链路。", "新增独立版本的输出域Brown6+2+4适配。原视频像素不重采样，原始鱼眼参数不能进入该合同。既有v1/v2/v3运行源码及人物参数保持原样。", "每个相机单独记录完整K、w2c和径向/切向/薄棱镜系数；前向特征投影、反投影射线、UT范围、三维高斯响应、训练与模型包读取使用相同参数。", "26项集成检查通过：独立OpenCV投影最大差8.53e-14px，解析Jacobian中心差分误差2.13e-10；独立OpenCV反投影加NumPy各向异性高斯射线响应与CUDA最大差4.78e-7（RGB/alpha值）。这些是已知合成几何，不是人脸定位精度。", "96个全协方差/三阶SH自由高斯、两套不同完整相机、空优化器训练20步，前景损失0.026934降到0.004113。相机与畸变冻结，原生数组及两台相机RGB/alpha导出回读误差均为0。", "代码审阅发现逐步快照不能拦截步骤之间的D字典修改；补创建时相机摘要，11项CPU检查验证K/E/径向/切向/薄棱镜修改在渲染和导出前被拒绝。原GPU结果与修改前源码哈希保留。", "已有N6失败候选做一次只读传输校验：7相机935个FIT点，显式K和像素同时+0.5后新旧公式最大差9.10e-13px。未发现该一参数模型正反方向或像素起点接反，旧相机/系数未重新拟合或晋升。", "新v4原生包保存精确线性数组、标准PLY和同坐标含畸变相机；旧读取器拒绝新合同，避免把相机畸变静默丢掉。新的畸变交互界面和完整原生分辨率容量尚未验证。", "这轮仅完成工程接入与合成验证，未确定真实残余畸变参数、未改变人物相机、未开始人物训练。实际面部13.049px及全部464帧和材料身份准入仍未通过。", "图上排：独立射线、CUDA响应、绝对差×1000。下排：两台不同畸变相机20步合成训练结果；右下留白。均96×80，显示等比例。", "本地证据：robust-portrait-122/native-residual-camera/result.json、frozen-view-addendum.json、historical-n6-bridge.json。下面保留历史内容。"].concat([previous]).join('\n\n');
const candidatePath=build+'/residualCamera-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122残余相机链路.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/residualCamera-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_residualCamera_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/residualCamera-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/residualCamera-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/residualCamera-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/residualCamera-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
