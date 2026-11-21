"""Prepare a single-slide edit using the existing Artifact Tool workflow."""
from pathlib import Path
import json
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/native-supervision')
r=json.loads((R/'review.json').read_text(encoding='utf-8'))
assert r['actual_review_complete'] and r['all_silhouettes_coarsely_reviewed']==464
s=(H/'update_deck_freshPrep_122.mjs').read_text(encoding='utf-8')
s=s.replace('freshPrep-122','nativeSupervision-122').replace('native-fresh-preparation','native-supervision')
s=s.replace("root+'/known-mask-centers.png'","root+'/review-inputs/report-comparison.png'")
s=s.replace("'/nativeEntry-122-deck-receipt.json'","'/freshPrep-122-deck-receipt.json'")
s=s.replace("if(review.checks!==26||review.GPU_used!==false)","if(review.all_silhouettes_coarsely_reviewed!==464||review.actual_review_complete!==true)")
start=s.index("replace('训练前的相机");end=s.index('const images=',start)
s=s[:start]+"""replace('遮挡观测与衣物表面初始化','白衣视频的原生图与分割草稿');
replace('合成轮廓（上）与20万点中心投影（下）。','第0、232、463帧，每组左为原图、右为分割。');
replace('6个前景、2个未知视角，支持率由75%改为100%。','464帧图像与时间已固定，完整相机仍待注册。');
replace('26项控制通过。实拍面部仍未对齐，训练未启动。','边界与遮挡仍待验收。面部误差未改善，尚未训练。');
"""+s[end:]
s=s.replace('width=frame.height*576/256','width=1130')
s=s.replace('已知合成几何的四相机轮廓与实际20万点中心投影；图内无文字，保留比例','第0、232、463帧的原图与分割草稿；无图中文字，保留比例')
s=s.replace('image.frame={left:frame.left+(frame.width-width)/2,top:frame.top,width,height:frame.height};','image.frame={left:75,top:190,width,height:width*541/1824};')
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index(".concat([previous]).join('\\n\\n');",start)+len(".concat([previous]).join('\\n\\n');")
notes=[
 '2026-10-28 T5白衣视频完整原生帧和分割草稿。执行前登记全部464帧，不复用旧人物几何、点云或优化器。来源视频SHA256 0a4d31cfcf31036ef2244b2709daaf29b425c1921c5516183302670728d030de，显示原生1520×2704。',
 '逐帧ffprobe实读实际PTS，首帧0、末帧19.267744秒，帧间隔最小41.601、中位41.616、最大41.626毫秒。保持全段时序，不用24fps标签重建时间轴，不插入或删除帧。',
 'FFmpeg RGB48 PNG提取不旋转、不缩放、不再次去畸变。464张原生RGB逐一记录尺寸和SHA，7张与既有诊断PNG逐字节一致。PIL/模型读取使用8位RGB视图，PNG保留16位RGB存储；这属于解码颜色表示说明，不是几何重采样。',
 '人物分割复用本机固定SHA的YOLOv8n和SAM2.1tiny，无新增下载。跟踪主人物检测框身份，以框与中心提示分割，每帧保留所有检测框、选中身份、alpha、文件SHA和原始时间。检测/SAM内部缩放不等于亚像素边界定位，输出alpha为1520×2704草稿。',
 '实际审阅全部15页共464个轮廓缩略图，以及第0、116、232、348、463帧原图/分割和原生头脸、手部衣物、脚部裁片。粗轮廓检查不能替代全464帧逐像素边界验收。',
 *r['findings_zh'],
 '未自动创建所有像素可观察的mask。外部遮挡、边缘混色、颜色有效区域和物理对应需要分开核验，当前alpha未作为通过验收的训练监督。',
 '完整464帧相机仍未实际注册。当前7个诊断相机不能插值补齐。K、外参、材料轨迹与留出划分均未改变，13条面部留出最大仍13.04875764原生像素，没有新增真实人物拟合或高斯训练。',
 '第一组YAML的4000×3000经下采样、居中裁剪仍为主要条件输出假设。尺寸字段本身不独立证实去畸变的K_rect，旧相机已使用4000基准；本轮完整解码不构成新的标定证明。',
 '图为实际第0、232、463帧，每组左原图、右分割，展示缩小但未拉伸，图内无文字。原生RGB/alpha存D:/GS-SLAM-Data/robust122-whiteT-native-supervision，D:/00仅同步审阅图和证据，不同步大体积中间帧。',
 '证据：robust-portrait-122/native-supervision/plan.json、frame-plan.json、frame-clock.json、decode-receipt.json、result.json、review.json、review-inputs/index.json。后续需处理审阅中具体监督缺陷，并在新的可辨识证据下登记相机诊断，不重复失败的七帧BA。以下保留同一报告此前阶段记录。']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');"+s[end:]
s=s.replace('preserve_deck_freshPrep_122.py','preserve_deck_nativeSupervision_122.py').replace('02_有限视角重建优化路线_122衣物表面初始化.pptx','02_有限视角重建优化路线_122原生监督.pptx')
(H/'update_deck_nativeSupervision_122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_freshPrep_122.py').read_text(encoding='utf-8').replace('freshPrep-122','nativeSupervision-122')
(H/'preserve_deck_nativeSupervision_122.py').write_text(p,encoding='utf-8')
print('T5_DECK_BUILDER_READY')
