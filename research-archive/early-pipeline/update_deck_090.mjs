import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/Administrator/Documents/New project',build=work+'/shuolee-slide-update';
const base='C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee';
const run=base+'/current/joint-calibration-model-audit-090',source=base+'/reports/02_有限视角重建优化路线.pptx';
const expected='5da8884796a5c79c6390f272b92b71da6ea86dd5fe7032110272ec7a1864e696';
const skill='C:/Users/Administrator/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11348/skills/presentations';
const digest=b=>crypto.createHash('sha256').update(b).digest('hex'),hash=async p=>digest(await fs.readFile(p));
const read=async p=>JSON.parse((await fs.readFile(p,'utf8')).replace(/^\uFEFF/,''));
if(await hash(source)!==expected)throw Error('Canonical report changed');
const d=await read(run+'/decision.json'),visual=await read(run+'/visual-review.json'),previous=await read(build+'/human087-deck-receipt.json');
if(!visual.actually_inspected)throw Error('Actual image review required');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snap=(await p.inspect({select:'$.slides[*]',include:['id','elements.id','elements.kind','elements.text','notes.text']})).records;
if(snap.length!==52)throw Error('Unexpected slide count');
const slide=snap[51],old=slide.elements.find(e=>e.text?.startsWith('鼻口位姿补偿未通过独立纹理'));
if(!old)throw Error('Expected conclusion missing');
p.resolve(old.id).text.replace(old.text,'观测替换与相机扩展均未消除鼻口错位。\n继续核验原生投影链路，当前高斯仍未通过。');
p.resolve(slide.id).speakerNotes.text=[
 '当前结论：'+d.summary,d.observation,d.fan,d.focal,d.joint,d.independent,d.limits,
 '本页图像仍为083原生高斯的084冻结贡献分析。左侧为原始第100帧实拍，右侧为相同相机的渲染及重影贡献。红色对应初始中央面区域相关高斯的空间贡献，青色为其他区域。图像不是088至090的训练效果，不提供点的完整生成历史。当前没有新训练或新PLY。',
 '为何核验观测：图像检测给出语义关键点，局部纹理匹配试图跟随相似图像结构，二者并非必然对应固定的同一物理点。侧脸可见性、鼻翼轮廓、闭唇轮廓及眼镜反光均会影响这一关系。088采用相邻帧、锚帧直接匹配与逐帧组合匹配核查，先用前后向一致性、纹理相关性和结构张量判断可观测性，再报告语义位置差异。不使用相机残差选择好点。不能以匹配通过单独证明检测错误或真实头部运动。',
 '088固定条件：175张原图，342个帧对，每对51个人脸点。匹配窗口21像素、金字塔最大4层，前后向误差不超过0.75原图像素，15像素纹理窗口相关系数至少0.90，直接与组合匹配差不超过1像素。相隔10帧鼻翼只有123/330、唇缘322/1320个查询通过可观测性检查，未隐去其余失败查询。相邻帧直接与组合相同，不能视作两项独立证据。',
 'FAN参考：VHAP人脸观测模块使用face-alignment。官方face-alignment1.4.1的TWO_HALF_D、4层FAN、SFD与翻转推理，本轮使用既有官方预训练权重，对全部175张原图重新推理，未读取旧人物拟合。SFD仅用整数裁剪定位目标框，框坐标还原到原图后运行FAN。23帧未检测到有效目标，未用旧结果填补。共同查询的稳定性比较不是解剖真值。参考：https://github.com/1adrianb/face-alignment 。',
 '逐帧焦距假设：089在已有外参六自由度诊断之外只增加一个焦距标量，横纵焦距保持一致，变化限制在正负3%，主点与径向畸变固定。使用完全相同的支持点、验证点、图像空间特征簇及鲁棒损失。486次拟合全部收敛且无触边，仍未降低下脸误差。这个局部固定结构检验不能排除全局标定与结构耦合。',
 '090模型比较：SIMPLE_RADIAL含共享焦距、主点与一项径向畸变。OPENCV含横纵焦距、主点、两项径向畸变与两项切向畸变。候选从与基线完全相同的初始投影开始，旧三维点清空。两组使用相同的原图、SIFT对应和已注册相机初值，分别重新三角化后进行最多100次全局束调整。三角化阶段相机未变，最终联合优化相机和结构。多个内参作为一组模型变更，不能把结果归因于某个单独参数。参考COLMAP3.11.1官方实现：https://github.com/colmap/colmap 。',
 '独立验证：使用既有080图像纹理轨迹及固定查询划分，每个候选相机都重新计算15351个三维验证点，仅用轨迹外侧时间段三角化，中间时间段用于验证。所有159023个既定查询保持有效，区域同帧比较使用105708个。前述411是下脸图像区域查询数，不是411个独立人物或411个经过人工标注的鼻口点。',
 '原生求解记录：SIMPLE_RADIAL在31次迭代、OPENCV在42次迭代收敛。最终原始SIFT观测的平均重投影误差约1.126和1.122像素。此为训练观测统计，不能替代独立下脸验证。'+d.native_audit,
 '当前实现的结构限制：083从头20万点初始化，其中1772个中央面点投到本轮重新配准的中性FLAME表面，其他头发、颈部和衣物保留新初始化。训练后所有点自由，没有网格面片绑定或模板距离项。18000步输出300220点，格式兼容检查通过，但鼻口重影和未知视角几何未通过。初始化在一个表面上不保证优化后仍然保持单层表面。',
 '下一项核验：'+d.next_bounded_work.scope,d.next_bounded_work.gate,
 '本轮源数据与协议：'+run+'；观测核验：'+base+'/current/named-image-correspondence-audit-088；焦距核验：'+base+'/current/variable-focal-camera-audit-089。',
 '静态主线保持不变。相机来源是整张原图定位，人体仅辅助初始化。最新高斯083，最新诊断090，几何质量仍未通过。'
].join('\n\n');
const candidatePath=build+'/human090-candidate.pptx',finalPath=work+'/shuolee-slide-output/02_有限视角重建优化路线_相机观测核验.pptx';
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:work,candidatePath,finalPath,pythonExecutable:'C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:52,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'reference',families:['微软雅黑'],referencePath:source,referenceSha256:expected},verifyArtifactToolImport:true,receiptPath:build+'/human090-validation.json'});
const q=await PresentationFile.importPptx(await FileBlob.load(finalPath)),renders=[];
for(let i=0;i<52;i++){
 const bytes=new Uint8Array(await(await q.export({slide:q.slides.items[i],format:'png',scale:1})).arrayBuffer()),h=digest(bytes);
 if(i===51)await fs.writeFile(build+'/human090-final-52.png',bytes);
 else if(h!==previous.renders[i].render_sha256)throw Error('Unexpected slide change '+(i+1));
 renders.push({slide:i+1,render_sha256:h,changed:i===51});
}
await fs.writeFile(build+'/human090-deck-receipt.json',JSON.stringify({source,source_sha256:expected,finalPath,final_sha256:await hash(finalPath),changed_slides:[52],renders,canonical_replaced:false},null,2));
console.log(JSON.stringify({finalPath,changed_slides:[52],canonical_replaced:false}));
