import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/shared-uncertainty-bundle-106',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='b47cbc53ebd215865cbd41ab097590168b0c135ba2c56b982cfcd907b40e47aa';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),prior=await read(build+'/human105-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/human106-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('静态人物重建：共享特征与坐标核验','静态人物重建：共享特征的观测精度');
replace('175帧映射一致。291个轮廓外观测','检测输入中的面部特征约28像素高，定位误差需要独立核验。');
replace('下一步核验物理特征对应','归一化后身体保护通过，下半脸改善8.3%，尚未达到验收要求。');
const images=slide.elements.filter(e=>e.kind==='image');
if(images.length!==1)throw Error('Expected one evidence image: '+JSON.stringify(images));
const image=p.resolve(images[0].id);const frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(run+'/projection-review-slide.png')),contentType:'image/png',fit:'contain',alt:'106实际照片中的面部观测与共享三维点投影，原视频第1、70、124帧'});
image.frame=frame;
const previous=await read(base+'/current/shared-mapping-contract-105/decision.json');
p.resolve(slide.id).speakerNotes.text=previous.presentation_notes.join('\n\n')+'\n\n106观测精度与联合求解：\n\n'+d.presentation_notes.join('\n\n')+
 '\n\n公式：每个特征k只有一个X_k。面部二维残差r_ik=[pi(K[R_iX_k+t_i])-u_ik]/sigma_ik，sigma_ik=sqrt(4^2+delta_ik^2)，delta来自原图/翻转检测差异，单位为原图像素。场景纹理保持原1像素单位，两类观测均采用尺度2的Soft-L1稳健损失。sigma是既有启发式而非校准概率。相机内参固定。场景和面部XYZ全部重新三角化，175个外参及共享点联合优化。独立纹理留出观测从不进入求解。\n\n数据来源：shuolee/current/shared-uncertainty-bundle-106的measurement-audit.json、input-receipt.json、result.json及原始照片投影。六个固定视角的完整对照见网页。本页展示第1、70、124帧。没有新高斯点云或00部署。';
const candidatePath=build+'/human106-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_观测精度核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/human106-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human106-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human106-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
