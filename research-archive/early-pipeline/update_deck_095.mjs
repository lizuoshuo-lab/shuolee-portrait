import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/whole-body-early-trace-fresh-095',source=base+'/reports/02_有限视角重建优化路线.pptx';
const gradientUpdate=process.argv.includes('--gradient');
const tag=gradientUpdate?'human095-gradient':'human095';
const expected=gradientUpdate?'d91219d2c468b6e34c2734bf821a4443017e926a53455c17cb4bfe1ff6624a1f':'cf68ff793591884b40770e39e4cd88b5d7f869c999612d2ad78cf7128787c265';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),prior=await read(build+'/human094-deck-receipt.json');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51];
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing text '+prefix);p.resolve(el.id).text.replace(el.text,value);};
if(!gradientUpdate){
replace('静态人物重建：全身投影与边界一致性','静态人物重建：特征对应与高斯投影范围');
replace('175帧全身检查：肩袖','鼻口对应覆盖不足，高斯形状变化已参与全身边界外溢');
replace('当前模型的同相机全身轮廓','左：第499步全身对照。右：实拍跨视角特征匹配。扩散在首次增密前已出现。');
replace('躯干、腿脚的表面投影仍不一致','先复核多视角特征，再约束投影范围。当前尚未通过几何与新视角验收。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==2)throw Error('Unexpected image layout');p.delete(images.map(e=>e.id));
const paths=[run+'/gaussian-run/lineage/step-000499-post_step/body-000100.jpg',base+'/current/face-feature-match-audit-096/face-matches-000100.jpg'];
for(const [n,path] of paths.entries())p.resolve(slide.id).images.add({blob:await fs.readFile(path),contentType:'image/jpeg',position:{left:66+n*574,top:178,width:550,height:380},fit:'contain',alt:n===0?'原生训练第499步同相机完整人物对照':'原始第100与107帧同编号特征匹配复核'});
}else{
replace('鼻口对应覆盖不足','鼻口对应覆盖不足，颜色拟合与轮廓约束存在局部冲突');
}
p.resolve(slide.id).speakerNotes.text=d.presentation_notes.join('\n\n');
const candidatePath=build+'/'+tag+'-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_'+(gradientUpdate?'局部优化冲突':'特征与投影范围')+'.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,
 pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/'+tag+'-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/'+tag+'-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/'+tag+'-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
