"""Prepare a native edit of only the existing52ndslide; preserve its prior notes."""
from pathlib import Path
import json
H=Path(__file__).parent
assert not (H/'update_deck_materialGeometry122.mjs').exists()
s=(H/'update_deck_headSupport122.mjs').read_text(encoding='utf-8-sig').replace('headSupport122','materialGeometry122')
s=s.replace("build+'/whiteCamera122-deck-receipt.json'","build+'/headSupport122-deck-receipt.json'")
a=s.index("replace('背景相机");b=s.index('const images=',a)
s=s[:a]+"""replace('头部高度的固定点支持','对应证据接入三维评测');
replace('三列为248','沿用围栏反例，三列为248、264、280帧。4000基准K保持。');
replace('60组全部审阅','逐视角检查三角化支撑点，三次立柱观察均被拒绝。');
replace('该轨迹此前','已修复0.024像素的坐标依赖，面部误差仍达13.05像素。');
"""+s[b:]
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index('const candidatePath=',a)
notes=[
 '2026-10-28 N13。第一YAML K以4000×3000为当前首选输出假设，经下采样和居中裁剪得到视频K，N5/N6/N12g已经这样计算。该尺寸字段自身证明标定参考尺寸，未独立证明实际K_rect。当前1520×2704的fx1494.442548、fy1493.094180、cx761.316834、cy1357.202690保持原值。原视频不再去畸变。',
 '真实调用链核查：旧N3 Observation合同此前主要由诊断及报告使用。新增material_geometry122.py固定相机几何评测器复用同一数据类，检查目标及每个三角化支撑观察，同族任一反证传播到所有目标。保留每个原观测数值、冻结划分和所有失败。没有创建另一套观测格式。',
 '一项工程组件，正式122训练入口尚未实现。旧113 admission.py/cli.py/training.py内容哈希保持不变，不改变另一00对话的部署项目。相机文件绑定证据并不等于物理标定真值。',
 '一次有界CPU运行初轮在旧数值一致性检查失败后停止。failure.json和初始源码保存。诊断显示重设世界原点后99个目标的DLT预测最大变化0.023671011899px，误差值最大变化0.019660799455px。相同固定三维点的坐标变换投影差仅9.0949e-13px；恢复原216坐标后旧误差值一致到2.2867e-12px。不是相机矩阵转置错误的证据。',
 '必要修复只改新评测器：每个留出目标用最早支撑相机作局部坐标，最大支撑基线作尺度，做齐次代数DLT后转回世界坐标。待测目标从三角化与规范选取中排除。没有更新K、相机或原始像素。此规范消除输入世界Sim3依赖，但不将代数DLT变成几何重投影最优估计。',
 '修复后28项CPU组件检查全部通过：已知合成几何、完整偏心主点、目标与支撑物理身份、同族跨轨迹反证、FIT/留出泄漏、缺相机、未知静止性或测量不确定性、错误源视频、重复身份、不合法位姿以及世界旋转平移尺度变换。实际102个目标全部保留，99个历史目标恢复原规范核对。输入和旧入口哈希不变。',
 '新局部规范下输入世界坐标变化的预测最大差2.2737e-12px；规范改变导致误差数值相对旧评测最大变化0.055904140079px，如实保留。13个面部留出目标仅2个数值<1px，最大13.048757643629px，物理准入0个。0.024px的数值问题不足以解释13px偏差，不能宣称重影根因已解决。',
 '实物反例仍是N12i的族4936，248/264为横幅文字下立柱，280变为卡通图案下的另一实例。该族三次观察都因整族反证拒绝，原数值误差412.211、195.908、363.890px保留。该族从未用于N12g相机拟合，不能据此归因N12g面部误差。',
 '本页图片是上一轮已审阅的原生上下文，非本轮新增图像或训练结果。维持原像素比例，图内无新增文字。左至右248、264、280帧，上排远楼，下排围栏。',
 '下一步只读检查现有native_portrait_renderer122和119自由衣物表面训练实现，登记必要的合成工程接入验证。当前仅7诊断帧，全训练帧尚未注册，物理材料/静止性/输出K均有缺口，不开始人物训练。合成通过不等于实拍亚像素或多人物泛化通过。',
 '来源：robust-portrait-122/video-update-20261027/material-geometry-consumer-plan.json；material-geometry-consumer/failure.json、parity-diagnosis.json、repair-plan.json、review.json与gauge-consistent-repair/result.json/evaluated-observations.json。以下保留历史诊断：'
]
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[b:]
s=s.replace('02_有限视角重建优化路线_122头部固定点支持.pptx','02_有限视角重建优化路线_122对应证据接入.pptx')
(H/'update_deck_materialGeometry122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_headSupport122.py').read_text(encoding='utf-8-sig').replace('headSupport122','materialGeometry122')
p=p.replace('replacements[oldasset]=value;allowed.add(oldasset)','replacements[oldasset]=value\n                if a.read(oldasset)!=value: allowed.add(oldasset)')
(H/'preserve_deck_materialGeometry122.py').write_text(p,encoding='utf-8')
a=(H/'accept_head_support122.py').read_text(encoding='utf-8-sig').replace('headSupport122','materialGeometry122').replace('clip03-head-level-support','material-geometry-consumer').replace('robust122-n12i','robust122-n13').replace('N12i','N13')
a=a.replace('N13 heading,4000basis,60family support review andfar-facade/repetitivepost nativecontexts actuallyvisible','N13heading,4000basisKtable,triangulationsupportconsumer,0.023671pxgaugeissue and13.048758pxfacefailure actuallyvisible')
a=a.replace('N1360nativebackgroundfamilies all reviewed; only provisionalnear-depth4936switchespostinstance at280, notusedN12g. ExistingN3contractrejects. ActualUI andsamePPT52completed;no camera/modelmutation ortraining.','N13actualfixedcamera consumer uses existingN3support/familyevidence,28controls/102targets;gauge inconsistency repaired butface13.05px remains. ActualUI/samePPT52complete;4000Kand119model unchanged,noformal122trainingentry ortraining.')
a=a.replace('latest_head_static_support_PPT','latest_material_geometry_PPT').replace('Actual N13 headlevelsupport andpostidentity report','Actual N13conditional4000K andmaterialgeometryconsumer report')
(H/'accept_material_geometry122.py').write_text(a,encoding='utf-8')
print('N13_DECK_PREPARED')
