import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/human-surface-attribution-084',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='27d77b0b93fa13811b83a1c1c6e5c5edeb2295a6fbccc3029d44c667628040c2';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),visual=await read(run+'/visual-review.json'),previous=await read(build+'/human083result-deck-receipt.json');
if(!visual.actually_inspected)throw Error('Actual evidence review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snaps=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snaps.length!==52)throw Error('Unexpected slide count');
const slide=snaps[51];
const text=(k,value)=>{const e=slide.elements[k];if(e.kind!=='textbox'&&e.kind!=='shape')throw Error('Not text '+k);p.resolve(e.id).text.replace(e.text,value);};
const title=slide.elements.find(e=>e.text?.startsWith('静态整体重建：'));
if(!title)throw Error('Expected conclusion title missing');
p.resolve(title.id).text.replace(title.text,'静态人物重建：表面约束与剩余问题');
text(2,'整图相机与人体初始化已接通；仅更换初始点仍有重影。');
text(4,'同一相机：左实拍，右重影贡献点；红色归于中央面片，青色为其他区域。');
text(5,'中性模板跨视图误差9.12像素，特征三角化2.28像素。\n先修正静态表面，再验证约束；当前质量仍未通过。');
const images=slide.elements.filter(e=>e.kind==='image');if(images.length!==2)throw Error('Unexpected image layout');
for(let i=0;i<2;i++){const e=p.resolve(images[i].id),old={...e.frame};e.replace({blob:await fs.readFile(run+'/'+(i?'raw100-attribution':'raw100-photo')+'.png'),contentType:'image/png',fit:'contain',alt:i?'083冻结模型重影贡献归因，红色最近模板三角面属于中央面片，青色为其他区域':'同相机原视频第100帧实拍'});const width=old.height*440/365;e.frame={left:old.left+(old.width-width)/2,top:old.top,width,height:old.height};e.crop={left:0,top:0,right:0,bottom:0};e.fit='contain';}
p.resolve(slide.id).speakerNotes.text=[d.summary,d.measured,d.template,d.historical_limit,d.binding_decision,d.next_step,
 '本页末态诊断不等于证明初始点的漂移历史。冻结077和083同一175个整图相机，对第100帧2420个固定脸外像素求每个高斯颜色的梯度，得到真实前向alpha合成贡献。140条原固定射线不按新误差重新挑选；083的48条脸外射线中46条仍有贡献，缺失保持在分母。',
 '第100帧以90%贡献选中1681个高斯；在175视角下总可见贡献252321.75，其中前景185190.82、背景54327.22、剩余为掩码不确定边界。该组也参与正常脸部显示，不宜直接删除。中央面片只是最近三角面归属，并非皮肤语义真值。面片贡献距离中位19.629模板等效毫米、法线锐角55.924度；下部纹理面片距离中位3.876模板等效毫米。模板等效长度由世界长度除以本轮尺度3.405981568取得，不是人体实测。',
 '固定24条下部纹理对应全部命中中性模板：参考视图误差中位9.119240像素，083贡献加权中心深度射线11.345180像素，原特征三角化2.281354像素；模板14/24好于083，13/24仍超过5像素。上部20条包含镜框/镜腿，不能将其视为裸面皮肤。预定例图取每源视图每组首条记录，不按误差排名。所有输入照片保留原像素位置。',
 'GaussianAvatars源码的get_xyz为mu_i=c_f+s_f R_f u_i，世界旋转为R_f R_i，尺度为s_f exp(l_i)。固定网格下这本身是重参数化；默认lambda_xyz=0.01、threshold_xyz=1以及局部尺度正则才限制自由度。绑定不会自动修正错误模板，不直接照搬含逐帧表情的模型。本阶段只有共享静态人物。',
 '083仍为20万初始点、18,000步、300220个最终高斯，训练和标准PLY导出已完成，但717组下半脸框误差6.172462→6.224015，质量验收未通过。084没有训练或新的PLY，没有00部署；最新不等于最佳。',
 '源码参考：https://github.com/ShenhanQian/GaussianAvatars ，本地scene/gaussian_model.py:get_xyz/get_scaling/get_rotation，train.py局部xyz和scale约束。证据目录：'+run+'；protocol.json、summary.json、template-transfer.json、cross-view-support.json、visual-review.json。'
].join('\n\n');
const candidatePath=build+'/human084v2-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_表面归因校验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human084v2-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);if(i===51)await fs.writeFile(build+'/human084-final-52.png',bytes);else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected change '+(i+1));renders.push({slide:i+1,render_sha256:h,changed:i===51});}
await fs.writeFile(build+'/human084-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
