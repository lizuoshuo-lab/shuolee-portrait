import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='4d0336d0b3c3a2b25719d8f501096eb1725ba77a15580260017ca91f7311f921';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const audit=base+'/current/static-depth-mixture-audit-078',decision=await read(audit+'/decision.json');
const result=await read(audit+'/fixed-ray-result.json'),previous=await read(build+'/observed077-deck-receipt.json');
if(!result.model_unchanged||result.gaussian_updates!==0)throw Error('Diagnostic source mismatch');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snaps=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
const text=(i,k,value)=>{const e=snaps[i].elements[k];p.resolve(e.id).text.replace(e.text,value);};
await fs.writeFile(build+'/depth078-before-52.png',new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer()));
text(51,5,'下半脸：高斯误差6.17像素，固定相机的单深度拟合为3.78像素。\n下一步检查完整对应库的约束覆盖，相机与微动影响仍待区分。');
p.resolve(snaps[51].id).speakerNotes.text=(snaps[51].notes?.text||'')+'\n\n078冻结诊断补充，2026-10-24：本页图像仍为075和077的真实高斯渲染。078没有新高斯训练，没有相机更新。图下3.78像素来自独立特征的单射线深度拟合，绝不是新渲染模型的效果。\n'+decision.findings.join('\n')+'\n'+decision.next_step+'\n完整方法：固定原始特征的源像素和175帧整图SfM+BA相机，对每个源特征只优化一个正深度。深度用该轨迹的非留出观测拟合，连续中间三分之一观测只作检查，稳健尺度为2完整去畸变像素。下半脸78条轨迹形成717组对应，156条源射线全部收敛且所有观察深度为正。高斯当前717组误差中位6.172462，单射线深度的留出误差3.775065。每组可能共享轨迹，不能视作717个独立样本。逐对最短极线距离中位1.553846使用了目标观测，是几何下界，不是留出估计器。固定源射线的约束也会受源特征位置误差限制。相机估计、特征漂移、反射和人物微动都尚未排除。\n开源参考：2D Gaussian Splatting，https://github.com/hbb1/2d-gaussian-splatting ，https://surfsplatting.github.io/ 。其深度畸变约束配合surfel光栅器；普通3D高斯中心深度方差不是等价实现。078只作源码方法参考，没有接入或声称复现。\n证据：'+audit+'/depth-result.json ; gradient-result.json ; epipolar-result.json ; fixed-ray-result.json ; decision.json';
const candidatePath=build+'/depth078-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_深度诊断.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/depth078-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/depth078-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/depth078-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed:[52],canonical_replaced:false}));
