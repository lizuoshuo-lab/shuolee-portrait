import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/gsplat-distortion-research';
const figure=root+'/distortion-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/nativeUI3-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.checks_passed!==8||review.human_training_started!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/gsplatDistortion-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('原生参数包加载与相机回放','gsplat 的畸变相机与训练边界');
replace('同一合成几何的竖屏、横屏全幅与原生裁片。','无畸变 / 给定畸变 / 差值×10：六个已知高斯。');
replace('打开包后，两种相机的原生浮点回放与参考一致。','本机已验证畸变渲染与高斯梯度；标定须独立完成。');
replace('通用PLY保留相机，真实人物重影仍未通过验收。','1.5.3 此分支不提供 K、畸变系数或外参梯度。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'本机gsplat六高斯合成CUDA控制：无畸变、给定Brown畸变、绝对差十倍；图内无文字',fit:'contain'});
image.frame={left:75,top:225,width,height:width*64/304};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 gsplat畸变相机源码调研及一次合成CUDA能力检查。", "gsplat1.5.3已支持OpenCV透视畸变6径向+2切向+4薄棱镜、KB四系数鱼眼、F-Theta及滚动快门；不能按系数个数把未定义的FISHEYE_8直接塞入这些模型。", "本机实际worker使用1.5.3源码937e299；非worker裸Python首次缺PYTHONPATH导入失败，按现有worker环境复核后成功。没有升级环境。", "唯一6高斯96×64合成CUDA试验8项通过，耗时6.42s。Brown6+2+4改变像素，五类高斯参数梯度有限；K/径向/切向/薄棱镜.grad均为None，外参反向明确NotImplementedError。", "现有本地v2适配层拒绝透视相机的畸变参数；它仅开放零畸变针孔和KB4鱼眼，训练准入同样只接零畸变针孔。因此底层支持不代表当前人体训练已做残余畸变补偿。", "对已去畸变裁剪MP4，应检验最终像素域的低自由度残差模型，先固定K与相机、独立校验残差再传给原生投影和射线求值；不能重新套原始鱼眼D。单纯camera_model=pinhole的名字不表示其不能包含非线性畸变。", "官方3DGUT例程要求with_ut和with_eval3d；示例推荐MCMC增密，不能直接复用依赖可微2D投影均值的默认增密规则。当前本地CVO只有全视角XYZ梯度统计，还未实现适配增密。", "官方main512d366增加自定义rays、外部BivariateWindshield模型等；本轮只读固定源码，没有安装/运行，不能把这些接口或梯度能力算成本机实测。", "本轮没有估计人物内外参或畸变、没有重跑D1-D4、没有人物训练。实际面部误差13.049px及464帧相机/材料准入缺口仍在。", "图中每幅原生96×64，三个区域用8像素白色间隔，最后为绝对RGB差值放大10倍。没有人物数据，不表示标定或重影改善。", "官方版本说明：https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html", "官方3DGUT训练指导：https://github.com/nerfstudio-project/gsplat/blob/512d366b67073d77ca099ede742683c165dfc23b/docs/3dgut.md", "实测：robust-portrait-122/gsplat-distortion-research/probe-result.json 与 sources.json。旧历史结果保留如下。"].concat([previous]).join('\n\n');
const candidatePath=build+'/gsplatDistortion-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122畸变接口调研.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/gsplatDistortion-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_gsplatDistortion_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/gsplatDistortion-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/gsplatDistortion-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/gsplatDistortion-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/gsplatDistortion-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
