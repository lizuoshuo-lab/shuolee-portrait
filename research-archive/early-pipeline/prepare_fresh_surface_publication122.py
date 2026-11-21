from pathlib import Path
import json,hashlib,shutil
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
V=T/'viewer';S=Path('D:/00/工作/查看器');L=Path('D:/00/工作日志/2026-10-27/内参文件解析')
for p in [V/'body-smplx.html',S/'progress.html',L/'内参文件解析与待核实项.html']:
 s=p.read_text(encoding='utf-8-sig');begin=s.index('<!-- ROBUST122_T4_BEGIN -->');end=s.index('<!-- ROBUST122_T4_END -->')+len('<!-- ROBUST122_T4_END -->');part=s[begin:end]
 assert '<style>.t4-result-table' not in part
 part=part.replace('<table class="t4-result-table">','<style>.t4-result-table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:15px;line-height:1.6}.t4-result-table td,.t4-result-table th{border:1px solid #3e5147;padding:8px 10px;text-align:left;vertical-align:top}.t4-result-table th{background:#1c3026}</style><table class="t4-result-table">')
 p.write_text(s[:begin]+part+s[end:],encoding='utf-8')
for dest in [V/'new-capture-118/calibration-note',S/'new-capture-118/calibration-note']:shutil.copy2(L/'内参文件解析与待核实项.html',dest/'01_内参文件解析与待核实项.html')
s=(H/'update_deck_nativeEntry_122.mjs').read_text(encoding='utf-8')
s=s.replace('nativeEntry-122','freshPrep-122').replace('native-training-entry','native-fresh-preparation').replace("'/report-table.png'","'/known-mask-centers.png'")
s=s.replace("'/nativeUIV2-122-deck-receipt.json'","'/nativeEntry-122-deck-receipt.json'")
s=s.replace("if(review.checks!==46||review.actual_exit_code!==3)","if(review.checks!==26||review.GPU_used!==false)")
start=s.index("replace('原生界面的");end=s.index('const images=',start)
s=s[:start]+'''replace('训练前的相机与材料验收','遮挡观测与衣物表面初始化');
replace('实际输入检查：原图、相机、遮罩、材料轨迹。','合成轮廓（上）与20万点中心投影（下）。');
replace('46项检查通过；当前视频在高斯初始化前停止。','6个前景、2个未知视角，支持率由75%改为100%。');
replace('面部最大误差仍13.049px，未启动新人物训练。','26项控制通过。实拍面部仍未对齐，训练未启动。');
'''+s[end:]
s=s.replace('frame.height*741/474','frame.height*576/256').replace('实际训练入口报告表；截图裁剪无重采样，没有在照片上添加文字','已知合成几何的四相机轮廓与实际20万点中心投影；图内无文字，保留比例')
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index(".concat([previous]).join('\\n\\n');",start)+len(".concat([previous]).join('\\n\\n');")
notes=[
 '2026-10-28 T4从头准备接口与遮挡观测分母。数据、代码和预算先登记，仅执行一次23.284秒CPU验证，26项控制通过。未运行真实人物公开先验、SMPL-X拟合、高斯优化或GPU。',
 '源代码发现：历史119的clothed_surface.py把前景票数除以全部视角数，没有观测有效mask。受控八视角已知箱体中，同一点6个明确前景观测、2个被遮挡的未知观测，历史分母给6/8=.75，新分母给6/6=1。阈值固定.97；未知退出投票，背景继续否定，至少三视角有效。该例能证明算法将未知视为反对的缺陷，不证明历史脸部重影的唯一原因。',
 'native_surface122.py从全新拟合仅获取粗提议范围和世界变换。已知合成衣物包络生成20万个自由世界点，全部位于内箱范围以外，距离内箱最大.214846模型米。这个量不是皮肤距离或衣服真实厚度。中心、正尺度、wxyz四元数、透明度和三阶SH完整，没有裸SMPL三角面配额、吸引项、动态形变或旧人物点云。',
 '本页图来自实际known-mask-centers.png，四列相机0、2、4、6，上排解析轮廓，下排200000中心投影。不是高斯渲染，也不是人物。固定2像素轮廓膨胀、.012模型米网格与最近像素采样使包络更宽。不能据此声称准确衣物深度、凹陷、皮肤贴合、亚像素精度或未知背面恢复。图内无文字，不拉伸。',
 '控制还验证实际背景拒绝、完全未知不产生证据、世界刚体变换保持投票、动态每帧几何拒绝、正尺度、提议范围截断拒绝、表面接触未知支持区拒绝。没有自动降低阈值或生成假闭合表面。',
 'prepare_native_person122.py已连接输入验收、新公开先验与静态共享SMPL-X拟合、实际拟合审阅、新自由衣物表面、实际表面审阅与T3训练入口。人体解剖检测原始NPZ字节、时间、帧身份、K整数中心到角点+.5桥接和原外参逐项控制通过；这只是接口验证。原拟合器对外部遮挡未作显式观测有效建模，粗拟合必须同相机审阅。成功实际推理/拟合/GPU分派尚未执行，不能把接口接通称为完整端到端完成。',
 '实际白衣clip03新fit命令返回3，在先验、拟合、表面和GPU前停止。102条原始材料观测与13条脸部数值均保持不变，最大13.04875764像素，物理准入仍0。完整训练帧计划和原生alpha、颜色有效、遮挡有效mask仍缺；464为编码帧总数，7个为诊断相机。',
 '用户最新说明继续以第一组YAML的4000×3000为主要输出假设，再下采样居中裁剪。标定文件width/height直接可读，但实际去畸变输出K是否同值尚未独立证实。N5/N6/N12g已经采用4000基准，不是本轮改掉4096就能解释现有大误差。保留整数缩放裁剪与半像素约定，不重新去畸变或重新估K。',
 '下一步独立准备明确训练帧选择和原生监督数据，先核对原视频精确帧时钟并登记选择，不能插值补相机或重复失败BA。相机、材料、静止性和面部逐点小于1像素验收不变。历史119点云、00部署和原113安装入口不改。',
 '证据：robust-portrait-122/native-fresh-preparation/plan.json、result.json、review.json、checks.json、known-scene/known-geometry.json、surface-positive/result.json、actual-person-denied/admission.json。以下是同一报告此前阶段记录，当前验收以上述范围为准。']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_nativeEntry_122.py','preserve_deck_freshPrep_122.py').replace('02_有限视角重建优化路线_122训练入口.pptx','02_有限视角重建优化路线_122衣物表面初始化.pptx')
(H/'update_deck_freshPrep_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_nativeEntry_122.py').read_text(encoding='utf-8').replace('nativeEntry-122','freshPrep-122')
(H/'preserve_deck_freshPrep_122.py').write_text(p,encoding='utf-8')
print('T4_TABLE_AND_DECK_BUILDER_READY')
