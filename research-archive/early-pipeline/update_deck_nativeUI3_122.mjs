import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const source='D:/00/报告/02_有限视角重建优化路线.pptx';
const root='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-ui-v3';
const figure=root+'/../native-cvo-control/cvo-native-control.png';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const python='C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
const prior=await read(build+'/nativeCVO-122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected||!prior.canonical_replaced)throw Error('Source changed; no overwrite');
const review=await read(root+'/review.json');
if(review.state!=='passed'||!review.actual_open_file||review.person_training!==false)throw Error('Review incomplete');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source render changed');
await fs.writeFile(build+'/nativeUI3-122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('跨视角更新与原生参数保存','原生参数包加载与相机回放');
replace('同一合成几何的竖屏、横屏全幅与原生裁片。','同一合成几何的竖屏、横屏全幅与原生裁片。');
replace('双视角统计已核验，20万点无损回读与参考一致。','打开包后，两种相机的原生浮点回放与参考一致。');
replace('PLY预览仍有响应差，真实人物重影尚未验收。','通用PLY保留相机，真实人物重影仍未通过验收。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==1)throw Error('Expectedoneimage');
const image=p.resolve(images[0].id),frame=image.frame,width=1130;
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',alt:'本机T8合成双视角控制，两幅全图等比缩略及各372×384原生裁片，图内无文字',fit:'contain'});
image.frame={left:75,top:190,width,height:width*192/672};
image.fit='contain';image.crop={left:0,top:0,right:0,bottom:0};image.lockAspectRatio=true;
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=["2026-10-28 T9 无损原生包实际导入与记录相机回放。", "新建隔离的v3原生入口，真实执行了下载原生ZIP、文件选择导入、两记录相机渲染、RGB/Alpha切换、下载浮点结果及预览ZIP。旧v1/v2服务保持原样，没有替换119历史人物模型。", "导入后的20万点五类线性参数逐位一致，完整K和外参逐项一致。两种原生分辨率1520×2704、2704×1520的RGB及alpha与T8导出前浮点参考最大差均为0。此项验证文件与原生回放一致性，不是实拍标定误差。", "26项预登记CPU检查通过。实际UI只执行两个GPU推理任务，切回相机与重载读取已保存结果，不重复计算。输出按原宽高比显示。", "原生ZIP和标准预览ZIP独立下载。原生ZIP内含无损NPZ、原字节PLY、同坐标相机和后端合同；预览包含3DGS相机、COLMAP完整K、SIBR目录。两份下载与生成包SHA一致。", "实际将预览ZIP送入原生文件入口时被拒绝，原先加载的原生模型及缓存保持可用。包类型不会静默转换。", "验收限于已登记合成包和记录视角。任意第三方模型、连续交互轨迹、独立Spark官网及SIBR官方GUI尚未验收，也未在00运行。", "DirectFisheye-GS可借鉴的CVO训练组件仍需绑定实际可信物理关联图，不能代替内外参标定。人物最大面部误差仍13.049px，全部相机和材料身份未通过，未启动人物训练。", "实测证据：robust-portrait-122/native-ui-v3/plan.json、review.json、ui-state.json及原始CUDA浮点输出。", "配图为T8已知合成几何的等比例全幅和原生裁片。T9实际导入后回放与这两幅参考的RGB/alpha逐像素最大差均为0。图片内没有文字，不是实拍人物或官方DirectFisheye训练结果。", "算法思想参考：https://arxiv.org/html/2604.00648v1 ，https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS 。后续保留历史技术记录。"].concat([previous]).join('\n\n');
const candidatePath=build+'/nativeUI3-122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122原生包回放.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/nativeUI3-122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath,figure,figure_sha256:await hash(figure)},null,2));
execFileSync(python,[build+'/preserve_deck_nativeUI3_122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/nativeUI3-122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/nativeUI3-122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/nativeUI3-122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected slide '+(i+1)+' change');
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/nativeUI3-122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
