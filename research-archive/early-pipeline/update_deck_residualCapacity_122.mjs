import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-residual-capacity';
const figure=root+'/r3-landscape-rgb.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/residualCamera-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.checks_passed!==19||review.human_training_started!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/residualCapacity-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('残余畸变的统一相机链路','完整畸变相机的全尺寸验证');
replace('同一 K、外参与畸变参数贯穿投影、训练与回读。','20万高斯，双视角更新与实际界面回放已核验。');
replace('合成训练与回读一致；实拍畸变系数仍待验证。','两路浮点回放差为0；实拍标定仍未通过。');
replace('26项集成、11项冻结检查通过；未启动人物训练。','原生1520×2704与2704×1520，完整K、外参与畸变。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=620;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'20万完整协方差三阶SH高斯的2704×1520合成原生渲染，图内无文字',fit:'contain'});
image.frame={left:330,top:190,width,height:width*1520/2704};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 R3全尺寸残余相机容量与R4实际原生界面回放。", "gsplat 输出残余畸变分支在两个完整原生尺寸上完成一次从头联合更新：1520×2704 与2704×1520、20万自由高斯、三阶SH、空优化器。19项检查通过，20万个中心实际更新。不是人物训练或18000步稳定性测试。", "本机RTX5080中，一次双视角更新0.820秒，分配显存峰值1,285,911,552字节。两相机保存前后浮点RGB与Alpha最大差均为0；内参、外参、12个输出域畸变系数和线性高斯数组逐项保持一致。", "新界面实际下载原生ZIP，再通过打开文件重新导入，运行两个已记录视角，查看RGB与Alpha并下载浮点响应。20项结果检查通过，两个全尺寸界面回放同参考浮点响应均完全一致。页面重载使用已保存结果，未额外运行GPU。", "旧版不含输出残余模型的包被明确拒绝，已加载场景保持有效。标准PLY仍作为预览资产配套；不能用不读取畸变的查看器证明原生图像一致。", "注册前检查发现，旧校验把float32外参矩阵往返误差混入了镜头逆映射。新版本在相机坐标检查镜头，在世界坐标另报3.44e-5与4.82e-5px舍入闭合差。8项CPU控制通过，1e-6px阈值与数值渲染公式均未改变；不是实拍重影的原因证据。", "首次界面启动在Bash第2行因Windows CRLF失败，尚未进入Python/CUDA。保留失败后只修复启动脚本换行为LF，完成原计划两个视角；旧服务与历史119模型未更改。", "实际审阅两个原生RGB和Alpha：已知合成椭球的位置、方向及画面比例一致。没有人物对照图、任意第三方包、自由环绕、官方SIBR或独立Spark、00验收。", "实拍面部最大误差仍为13.049px，全部464帧相机和独立物理材料对应尚未通过。渲染器支持畸变不代表实际畸变系数已求准；没有新人物训练、没有晋升失败相机。", "图为原生2704×1520合成椭球；保持比例，无图内文字。本轮仅一联合更新，不是人物训练、18000步稳定性或真实亚像素验收。", "证据：native-residual-capacity/result.json、review.json及native-residual-ui/review.json。以下保留历史备注。"].concat([previous]).join('\n\n');
const candidatePath=build+'/residualCapacity-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122畸变全尺寸回放.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/residualCapacity-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_residualCapacity_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/residualCapacity-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/residualCapacity-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/residualCapacity-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/residualCapacity-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
