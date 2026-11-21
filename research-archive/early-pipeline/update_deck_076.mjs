import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/fullscene-person-constraint-audit-076';
const source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='9d4bdd95dcc585e86036caa0689e0a03365863d4a7b3e26fa37b2c9a1b07cd67';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const contributions=JSON.parse(await fs.readFile(run+'/contribution-result.json','utf8'));
const correspondences=JSON.parse(await fs.readFile(run+'/correspondence-result.json','utf8'));
const previous=JSON.parse(await fs.readFile(build+'/fullscene075-deck-receipt.json','utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[51]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records[0];
const text=(k,value)=>{const e=snap.elements[k];p.resolve(e.id).text.replace(e.text,value);};
text(2,'整图定位与人体初始化已贯通，跨视图对应仍出现偏差。');
text(5,'82次下半脸图像块筛选中，29次通过但仍错位超过2像素。\n下一步检验真实纹理对应约束；相机误差仍未排除。');
p.resolve(snap.id).speakerNotes.text=(snap.notes?.text||'')+'\n076冻结075诊断：不改变原PLY、相机或优化器。原始帧100人像外侧重影覆盖的90%来自1473点；这组点跨175视角的可见贡献73.80%落在人物内部，简单删除会损伤其他视角。独立图像纹理跟踪投影：下半脸717次跨视图对应，当前高斯平面深度误差中位数6.748整幅去畸变像素，直接纹理三角化为2.664像素。96次属于实际训练邻域，其中82次通过几何和完整图像块NCC筛选，29次仍偏离观测超过2个1536高训练像素。82次并非82个独立特征或真实历史训练日志，同一轨迹可产生多对观测。当前NCC使用平方相关系数，82次中2次负相关，但该少数情况不足以解释整体重影。075相机仍有约2.66像素的独立下半脸残差，因此不排除相机模型/估计或特征跟踪误差，也不把微动当作既定原因。模型冻结情况下的自洽深度与高相似图像块可对应错误位置，支持下一步先做真实图像对应约束的有界梯度检查，再决定是否从头训练。\n证据：'+run+'/contribution-result.json ; '+run+'/correspondence-result.json ; '+run+'/patch-witnesses.jpg\n'+JSON.stringify({contributions:{points:contributions.selected_points,foreground:contributions.all_views_foreground_mass_fraction},correspondences:correspondences.summary});
const draft=build+'/constraint076-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_观测对应核验.pptx';
await(await PresentationFile.exportPptx(p)).save(draft);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:draft,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,
 receiptPath:build+'/constraint076-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/constraint076-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected other slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/constraint076-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,canonical_replaced:false}));
