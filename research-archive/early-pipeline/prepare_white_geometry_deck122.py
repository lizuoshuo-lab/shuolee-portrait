"""Prepare a native Artifact Tool edit of the existing slide52 only."""
from pathlib import Path
import json
H=Path(__file__).parent
assert not (H/'update_deck_whiteGeometry122.mjs').exists()
s=(H/'update_deck_whiteGrid122.mjs').read_text(encoding='utf-8-sig').replace('whiteGrid122','whiteGeometry122')
s=s.replace("build+'/whiteAllocation122-deck-receipt.json'","build+'/whiteGrid122-deck-receipt.json'")
s=s.replace("../clip03-grid-coverage/array-cache-repair/material_evidence.png","../clip03-fresh-geometry/face_feet_evidence.png")
start=s.index("replace('固定物体角点")
end=s.index('const images=',start)
s=s[:start]+"""replace('均衡特征分配后的固定物体对应','固定背景相机下的面部错位');
replace('三排为屋顶拟合族','三排为面部原图、面部投影与鞋面投影；三列为200、232、280帧。');
replace('屋顶循环轨迹从','背景留出中位数0.33px；面部留出中位数3.89px。');
replace('非地面固定物体留出族','面部最大13.05px，未通过逐点1px要求，暂不训练。');
"""+s[end:]
s=s.replace("alt:'屋顶拟合族屋顶留出族与玻璃疑点的原生128像素裁片'","alt:'原生面部及鞋面留出投影，蓝圈观测，粉色十字预测'")
start=s.index('p.resolve(slide.id).speakerNotes.text=');end=s.index('const candidatePath=',start)
notes=[
'2026-10-28 N12e/f/g。继续固定第一组YAML4000×3000参考K，按原N4半像素缩放和理想居中裁剪到1520×2704。没有换4096、重估K或二次去畸变。YAML尺寸证据与真实输出K的独立确认保持区分。',
'N12e只测定位：原83行的前三个观测，41×41原生局部窗口，双向仿射配准及三视角闭环，共498方向，477测量、10边界不足、8不收敛、3超出预定范围。面部SIFT和局部仿射位置最多相差3.254px，这是估计器差异而非已知真值误差。玻璃疑点也有小闭环，不能因此升级为固定物理点。结果未用于替换本轮几何观测。',
'N12f对既有全部31个屋顶族、全部观测的原生128裁片和256上下文完成审阅。3组轮廓或玻璃支持仍不明确，保留证据但不作为固定点。合并此前其他物体/地面后40FIT与15heldout，其中非地面25/6。没有重新提特征、换分组或按残差删点。此前2个非地面留出仅指有限已审阅样本，不是整份数据上限。',
'N12g第一条白T固定K相机候选：以满足预登记支持的最早邻对216/232从头PoseLib相对位姿，依次PnP注册200、184、248、264、280，原始SIFT观测与40背景FIT族进入固定K共享三维优化，未加载旧优化相机或几何。40族均有正深度和预定视差，无结构排除。7帧为464帧的诊断子集。',
'独立目标排除三角化统计：背景留出86个目标，中位0.331884px，最大1.544858px，81/86严格小于1。非地面24目标中位0.247845，最大1.046321。面部heldout2族13目标，中位3.891467px，最大13.051620px，2/13小于1。全部6面部族41目标中位4.609864，最大13.325810，2/41小于1。鞋袜7族47目标中位0.395835，最大2.827080，36/47小于1。',
'背景优化前后中位0.337299降到0.331884，但面部heldout中位3.584912升到3.891467。没有相机候选择优或更改内参。背景低中位数不能说明全部像素或人脸已对齐。',
'实际审阅全部41面部、47鞋袜目标的原图与投影。眉眼存在开合，晚期面部投影出现共同竖直偏移，鞋面多数贴合。当前没有唯一证据区分人物微动、测量偏差或相机在人体深度的弱约束。背景早期184/200无非地面留出，屋顶主要集中一栋远楼。206人物诊断目标最大168.43px异常全部保留，未宣称整体正确。',
'图中三列200、232、280帧；第一排face row59/family800 heldout原图，第二排同族预测叠加，第三排feet row77/family1454 heldout预测叠加。蓝圈是原始SIFT像素，粉十字是未使用目标观测三角化的3D预测。每格128×128原生像素，无二维形变或图中文字。',
'下一步只读分析背景FIT相机弱约束是否足以解释脸部共同残差，再判断低自由度头部微动诊断可辨识性。冻结本轮，不扫K/参数，不再次提特征或重跑相机。面部逐点1px未通过，未训练或交付新人物点云。119历史模型保持。',
'来源：robust-portrait-122/video-update-20261027/clip03-localization/result.json与visual-review.json；clip03-roof-review/visual-review.json；clip03-fresh-geometry/result.json、validation.json、visual-review.json、figure-index.json；各自执行前plan。以下保留历史诊断以便溯源：'
]
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[end:]
s=s.replace('02_有限视角重建优化路线_122均衡特征分配对照.pptx','02_有限视角重建优化路线_122固定背景与面部误差.pptx')
(H/'update_deck_whiteGeometry122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_whiteGrid122.py').read_text(encoding='utf-8-sig').replace('whiteGrid122','whiteGeometry122')
(H/'preserve_deck_whiteGeometry122.py').write_text(p,encoding='utf-8')
a=(H/'accept_white_grid122.py').read_text(encoding='utf-8-sig').replace('whiteGrid122','whiteGeometry122').replace('clip03-grid-coverage/array-cache-repair','clip03-fresh-geometry').replace('robust122-n12d','robust122-n12g').replace('N12d','N12g')
a=a.replace("'N12g heading,4000basis,equal-budget countsand3nativecrop rows actuallyvisible'","'N12g heading,4000basis,fresh7view geometry result,background/face/feet error table and3nativecrop rows actuallyvisible'")
a=a.replace('N12g equal-budget grid selection comparison+83materialrows+16contexts,actualUI andsamePPTslide52 completed. Roofcoverage improved;7FIT/2heldout nonground fails6/3;no newcamera/training.','N12e localphotometric diagnosis,N12f31roof review,N12g fresh7view fixedKcamera candidate and41face/47feet review completed. Backgroundmedian.332px butfaceheldout3.891px/max13.052 fails. ActualUI andsamePPT52completed;no camera/modelpromotion ortraining.')
a=a.replace('latest_white_grid_PPT','latest_white_geometry_PPT').replace('Actual N12g controlled selection/material report','Actual N12g fixedKgeometry andface/feet evidence report').replace('new_camera=False','new_camera_candidate=True,camera_promoted=False').replace('new_camera=False','new_camera_candidate=True,camera_promoted=False')
(H/'accept_white_geometry122.py').write_text(a,encoding='utf-8')
print('WHITE_GEOMETRY_DECK_EDIT_PREPARED')
