import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-ui-v2';
const figure=root+'/ui-completed.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/nativeV2-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.CPU_checks!==20||review.actual_UI_renders!==4)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeUIV2-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('原生渲染修复与点云回读','原生界面的相机与点云包');
replace('上排针孔，下排已知鱼眼。三列为相同三组相机。','按包内完整相机渲染，RGB、Alpha与配套下载已核验。');
replace('20万点恢复30,681点更新。','4次实际界面渲染与参考一致，点云和相机下载字节一致。');
replace('组件41项和命令行2项通过。','仅完成合成场景加载链路验证，实拍人脸标定仍未通过。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=frame.height*1265/712;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'实际原生v2界面：按包内记录相机显示已完成合成高斯，无新增图片注释',fit:'contain'});
image.frame={left:frame.left+(frame.width-width)/2,top:frame.top,width,height:frame.height};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-28 T2原生v2界面相机与点云包。独立回环端口8875，旧v1端口8874和历史119人物不变。复用T1已经完成的合成包，没有新增优化器、人体拟合、相机求解或特征提取。服务保持单GPU限制，四项预登记渲染完成后不再启动新任务。',
 '实际由Cua操作本机IAB tab42：选择针孔包相机0、相机1；已知KB4包相机0；20万点包相机0。四个实际WSL进程PID为380、383、378、368，全部退出0，投影、完整K、唯一w2c、图像尺寸、后端版本和SHA来自包，不接受请求改K/位姿/投影/任意路径。20项CPU保护检查通过。',
 '针孔0和KB4的原生浮点RGB/Alpha与T1独立CLI参考最大差均为0；针孔1和20万点PNG与T1直接渲染参考逐像素一致。320×240的六颗各向异性SH3合成高斯正常显示；64×64的20万点结果呈暗椭球和彩色边缘，没有全黑漏点。这里验证新界面调用同一后端，并非独立证明整个CUDA内核无误。',
 '实际切换记录相机时清除旧图，避免用前一相机结果冒充当前视图；完成后可看缓存，切回针孔0不启动第五个任务。RGB/Alpha切换实际查看。界面截图保持1265×712宽高比，无裁切变形，图片上没有新增文字；截图本身的文字是应用原有界面。',
 '实际浏览器下载文件：Downloads/T2_pinhole_native_v2.zip与native-rgb-alpha (1).npz。完整SHA与服务文件一致。ZIP的scene.json、point_cloud.ply、cameras.json保持原始字节，解包后重新读取完整K、协方差及SH。这里不是只检查HTTP返回成功；没有把旧Downloads文件冒充新下载。',
 'CLI工作进程的回执native_interactive_UI=false继续原样保留，因为该子进程本身只能证明CLI范围。实际UI点击、显示和下载的验收证据单独保存在T2 review.json、result.json、ui-state.json及截图，不通过修改旧字段制造GUI成功。',
 '明确范围：三组已登记包、四个固定记录相机、320×240/64×64合成数据的加载与下载通过。任意OpenFile导入、连续漫游、全视频分辨率20万点容量、自适应密度、真实人像质量、官方SIBR/独立Spark原生等价与00部署均不在本次通过范围。',
 '当前用户说明仍以第一组YAML K为4000×3000去畸变输出基准，然后下采样、居中裁剪。重新核对原文件：fx1658.0353708、fy1656.5394007、cx2001.5157183、cy1505.8269491。N5/N6/N12g早已使用4000基准，不能把当前大误差解释为仍在按4096计算。YAML记录标定像素尺寸，实际去畸变输出K_rect是否完全相同仍是条件假设。缩放主点、裁剪偏移、整数取整和半像素约定必须显式记录，不二次去畸变。',
 '实拍面部探索性最大误差约13.049px，未满足物理身份可信且独立逐点<1px；白衣新片464训练帧尚未完整注册。下一步把既有N3/N13材料证据及完整帧相机/遮罩准入接入实际训练入口，在准入失败时于初始化和创建优化器之前停止。没有新因果变量时不重复提特征、BA或旧人物训练。',
 '来源：robust-portrait-122/native-ui-v2/plan.json、cpu-checks.json、result.json、review.json、ui-state.json、ui-completed.png；实现native_interactive122_v2.py/.html、native_interactive_worker122_v2.sh。以下保留既有诊断历史。',previous
].join('\n\n');
const candidatePath=build+'/nativeUIV2-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生界面.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeUIV2-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeUIV2_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeUIV2-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeUIV2-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeUIV2-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeUIV2-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
