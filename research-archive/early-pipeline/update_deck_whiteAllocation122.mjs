import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const evidence='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video-update-20261027/background-geometry';
const figure=evidence+'/../clip03-structure-allocation/material_evidence.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/whiteMatching122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected 52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/whiteAllocation122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('白T恤人物的跨视角材料对应','固定物体角点在筛选中流失');
replace('三排依次为面部','三排为消防栓、台阶与远景重叠区域，原生裁片无形变。');
replace('104组已审阅','上方灯具230个合格角点只留5个，最终0条循环轨迹。');
replace('固定建筑材料支持不足','固定表面留出支持仍不足，本轮未求新相机或训练。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expected one slide52 image');
const image=p.resolve(images[0].id),frame=image.frame;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'消防栓台阶和远景混合深度区域的原生128像素裁片',fit:'contain'});
image.frame={left:frame.left+(frame.width-frame.height)/2,top:frame.top,width:frame.height,height:frame.height};image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 N12c完成。继续按第一组YAML的4000×3000工作基准，经下采样与居中裁剪映射到1520×2704。实际导出K未独立核实。没有改变K、进行二次去畸变或估计新外参。", "在已实际审阅的184、232、280帧登记24处搜索多边形，包含灯具、消防栓、台阶、建筑及围栏。只读追踪冻结N12b的检测、资格排除、名额筛选、配对和三视角循环，不重新检测、计算描述子或匹配。搜索区域包含重复尺度、方向和混合材料，不等同固定物理点。", "184帧上方灯具232个检测点、230个合格点，保留5个背景点，配对0、循环0。下方灯具163个合格点保留1个，配对0。232帧远处建筑区域6432个检测点、6417个合格点，保留11个，10个有配对、5条循环。区域可能交叠，不能汇总为唯一物理点。", "源码核验表明背景只在上中下三个大横带各选2000个最高SIFT响应点，没有更细空间格或材料分配。现有证据证实选择阶段遗漏固定物体区域，但不能证明被舍弃的角点必然匹配成功，也不能量化该原因对重影的影响。", "43个此前未审阅背景族触及搜索区域，按登记规则取20组，全部128像素观测实际审阅，其中16组另查256像素上下文。其余23组仍未审阅，不能称全部错误。新增粗固定表面候选2组，分别为消防栓阀门和台阶立面，均为FIT。另11植被、2消防栓轮廓边缘、2灯杆台阶混合深度、3远景建筑植被围栏混合。", "合并N12b，仅3个拟合族、0个留出族属于较清楚的非地面固定表面粗候选，仍未通过6FIT/3heldout预登记条件。本轮没有运行相机求解或训练。粗材料候选也没有获得独立亚像素测量认证。", "本页三排原生裁片：row3族156取184、200、216帧；row11族2382取248、264、280帧；row18族2037取232、248、264帧。每格128×128，观测在中心，无图像文字、重采样或二维形变。不同排的时间不同，不应解读为同帧空间拼图。", "下一步先登记一项只改变特征名额分配的受控比较，保持同一原生图、检测器参数及4000基准K，核验原检测身份并保存完整元数据，不扫描检测阈值或复用优化外参/几何。当前只完成诊断，没有执行该新比较。", "7帧诊断不能替代464帧完整注册。119历史模型保留，帽子案例与亚像素目标仍未解决。本轮没有00部署、官方SIBR或独立Spark验收。", "来源：robust-portrait-122/video-update-20261027/clip03-structure-allocation/allocation.json、visual-review.json、visual-index.json、context-index.json和material-evidence-index.json，执行前方案clip03-structure-allocation-plan.json。", "以下历史记录保留溯源："].concat([previous]).join('\n\n');
const candidatePath=build+'/whiteAllocation122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122固定物体特征分配核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/whiteAllocation122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_whiteAllocation122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/whiteAllocation122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/whiteAllocation122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/whiteAllocation122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/whiteAllocation122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
