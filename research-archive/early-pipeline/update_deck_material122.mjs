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
const prior=await read(build+'/r1c122-deck-receipt.json'),expected=prior.final_sha256;
if(await hash(source)!==expected)throw Error('Source changed; no overwrite');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Expected52 slides');
const slide=snap[51];
const before=new Uint8Array(await(await p.export({slide:p.slides.items[51],format:'png',scale:1})).arrayBuffer());
if(digest(before)!==prior.renders[51].render_sha256)throw Error('Source rendering changed');
await fs.writeFile(build+'/material122-before-52.png',before);
const replace=(prefix,value)=>{const el=slide.elements.find(e=>e.text?.startsWith(prefix));if(!el)throw Error('Missing '+prefix);p.resolve(el.id).text.replace(el.text,value);};
replace('人像重影：','人像对齐：原生像素与材料对应');
replace('背景低残差仍','旧版脸框含帽纹与背景，区域误差不能代替真实面部点验收。');
replace('原生包六组回读','新版戴帽人物已接入1520×2704。重新建立对应，尚未训练。');
const previous=p.resolve(slide.id).speakerNotes.text;
p.resolve(slide.id).speakerNotes.text=[
 '2026-10-27 R1d/C1a及新视频接入：R1d四组条件K共18项CPU检查通过，整数中心K与观测共同加0.5后射线最大差2.22e-16，逆变换误差0。漏转换会在新像素边界接口产生每轴0.5、欧氏0.707像素偏移；121原SIFT/求解/残差内部一致，因此这不是已证实的历史重影根因。',
 'C1a按照预登记选择全部冻结留出脸框相关轨迹，未按几何残差挑点。115条轨迹、476个观测、19页原生128像素裁片全部浏览。旧草地13/54、沙发46/195、戴帽人物31/133、室内25/94为轨迹/观测数。448个观测在既有脸框内，其余也保留。',
 '真实材料验证缺口：戴帽人物有帽纹/后脑轨迹，编号13与15为经过脸框的河岸背景；室内编号25为经过脸框的背景玩偶。沙发很多特征在胡须或眼镜附近。没有按结果删点、重新分配fit/heldout或重算漂亮指标。浏览不是逐点物理身份、静止性或测量不确定性认证。旧版faceROI误差不能当作眼鼻嘴真实物理点误差。',
 '用户新同步5视频。新版戴帽人物为视频-777d211114ab4f39a1b5612466dda6d5.mp4，SHA32be30636adabfc4c25a1c8387cf19302d3ca39b4f040072081646d36523317c，1520x2704、557帧、23.180586秒；替代原1080x1920 online4供后续处理。另外4个新人物已做元数据和原生样帧接入，尚未训练。',
 'N0在原帧0/266/556比较两份编码。只使用按尺寸预先确定的边界对齐缩放矩阵，不拟合二维变换/K/外参，不重采样训练图。1840个双向ratio0.75外观匹配中1785个误差<1新版像素，55个>=1全部保留，三帧中位数0.198/0.230/0.330px、P900.408/0.574/0.759px，最大误差196.34/696.74/261.21px。支持大体尺寸映射，也保留错误匹配/定位不稳的证据；不能等同跨视角面部亚像素验收。',
 '当前PPT中的鼻区图片仍为既有历史证据，不是新版渲染。实拍逐点<1像素、全帧相机、物理材料与人体微动仍未通过。新版使用自己的原生观测与完整输出K合同，不继承旧相机/几何/高斯。查看器保留119历史模型，不宣称重影已修复或00已部署。',
 '证据：robust-portrait-122/pixel-convention-bridge/result.json、review.json；face-material-support/visual-review.json和19页原生裁片；video-update-20261027/content-review.json、portrait-version-mapping/result.json。更新既有body-smplx.html#robust122-new-input和同一工作日志。',previous
].join('\n\n');
const candidatePath=build+'/material122-candidate.pptx';
const finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_122材料核验与新版视频.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
await fs.writeFile(build+'/material122-deck-build.json',JSON.stringify({source,source_sha256:expected,candidatePath,finalPath},null,2));
execFileSync(python,[build+'/preserve_deck_material122.py'],{stdio:'inherit'});
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath:build+'/material122-exact-candidate.pptx',finalPath,
 pythonExecutable:python,integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
 explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
 fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},
 verifyArtifactToolImport:true,receiptPath:build+'/material122-final-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/material122-final-52.png',bytes);
 else if(h!==prior.renders[i].render_sha256)throw Error('Unexpected render change on slide'+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/material122-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],first51_render_hashes_identical:true,canonical_replaced:false}));
