import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/c0a122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source rendering changed');
await fs.writeFile(build+'/r1c122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('人像对齐：','人像重影：相机约束与原生渲染');
replace('122局部诊断：','背景低残差仍不能证明人脸对齐，内参与位姿的独立约束仍弱。');
replace('内参与位姿高度耦合','原生包六组回读通过。实拍逐点<1 px尚未通过，未开始新训练。');
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-27 R1c更新：已有六个合成各向异性高斯、三台相机、针孔与KB4两种投影，按SHA绑定的原生包加载完整K、唯一OpenCV w2c和标准PLY三阶SH。固定gsplat1.5.3 UT投影和三维响应。六组原生RGB/alpha回读对照均通过预登记最大绝对差3e-6，RGB最大1.37090683e-6、alpha最大2.08616257e-6。单位是0..1强度差，不是像素重投影误差。',
 '21类无效输入在GPU调用之前被拒绝，包括错误SHA、路径越界、重复或双重位姿、缺失K、未声明坐标约定、不兼容后端/求值方式、已去畸变输入使用鱼眼投影。当前四段MP4已去畸变、下采样和裁剪且未加防抖，仍须针孔输出K。',
 '六幅合成原生图和alpha已审阅，完整记录相机/PLY原字节保持不变。这是headless加载与原生渲染的合成验证。没有新人物训练或PLY，原生交互界面、官方SIBR/独立Spark及00未因此验收。旧鼻区图仍仅为历史材料对应证据。',
 '下一步R1d已预登记：121特征/K为整数像素中心，新原生接口要求像素边界原点。检查显式且可逆的坐标转换，不更改已有相机、不重估K、不把接口差异宣称为旧重影原因。真实材料对应、实际K_rect和人物微动仍须分别核验。',
 '证据：current/robust-portrait-122/native-package-loader/result.json、review.json、fixture-package/scene.json及六组raw RGB/alpha；native-package-loader-plan.json、pixel-convention-bridge-plan.json。工作日志与查看器body-smplx.html#robust122-r1c同步。',
 previous
].join('\n\n');
const candidatePath=build+'/r1c122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生包核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/r1c122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath},null,2));
execFileSync(python,[build+'/preserve_deck_r1c122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/r1c122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/r1c122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/r1c122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected render change on slide'+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/r1c122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
