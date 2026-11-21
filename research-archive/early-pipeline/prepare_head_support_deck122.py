"""Prepare native slide52 edit, retaining the existing deck and its history."""
from pathlib import Path
import json
H=Path(__file__).parent
assert not (H/'update_deck_headSupport122.mjs').exists()
s=(H/'update_deck_whiteCamera122.mjs').read_text(encoding='utf-8-sig').replace('whiteCamera122','headSupport122')
s=s.replace("build+'/whiteGeometry122-deck-receipt.json'","build+'/whiteCamera122-deck-receipt.json'")
s=s.replace('../clip03-camera-observability/background_face_feet.png','../clip03-head-level-support/facade_post_context.png')
a=s.index("replace('固定背景"); b=s.index('const images=',a)
s=s[:a]+"""replace('背景相机约束对面部的影响','头部高度的固定点支持');
replace('三排为背景','三列为248、264、280帧，上排远楼，下排重复围栏。');
replace('假设背景观测噪声','60组全部审阅，唯一暂估近景候选在末帧跳到另一根立柱。');
replace('这是局部敏感度','该轨迹此前未参与相机拟合，不能据此解释现有面部重影。');
"""+s[b:]
s=s.replace("alt:'背景、面部、鞋面原生留出投影，蓝圈观测，粉色十字预测'","alt:'远楼与重复围栏立柱的三帧原生上下文，图片无新增文字'")
s=s.replace('left:frame.left+(frame.width-frame.height)/2,top:frame.top,width:frame.height,height:frame.height','left:frame.left+(frame.width-frame.height*1.5)/2,top:frame.top,width:frame.height*1.5,height:frame.height')
a=s.index('p.resolve(slide.id).speakerNotes.text=');b=s.index('const candidatePath=',a)
notes=[
 '2026-10-28 N12i。保持第一组YAML的4000×3000参考尺寸、下采样和居中裁剪的工作假设。最近的N12g已使用此K。本轮不改K、相机、三维点或原始观测，也不新增检测匹配。',
 '预登记选择：7帧中6个冻结面部材料族的y范围加减128原生像素，横向取全图。收录任一既有成员触及范围的全部可用背景族，保持先前代表轨迹或最长轨迹加固定哈希排序。选择不依赖残差或深度。60组全部审阅，37FIT、23heldout，原划分不变。',
 '实际审阅15张256×256上下文页，包含全部60组中心128×128原生裁片及所有可用观测；另看4936的128像素裁片、232原图及先前审阅。未声称另行打开全部10张128裁片页。结果31组远楼重复窗带、18组植被或叶片天空边缘、5组建筑围栏植被混合、3组横幅围栏、1组台阶植被交界、1组树干叶片交界、1组重复立柱换实例。',
 '暂估几何仅用于深度上下文。候选相机为已完成N12g，没有优化。面部深度参考采用4个FIT标签面部族，它们未进入相机FIT。60组均为正深度，只有4936的深度/脸深度中位比0.833落在描述性区间0.5至2内。其他59组约5至14。这里的区间不是物理验收阈值，深度不是米制真值。',
 '4936是heldout族，track2196。248、264帧中心立柱对应横幅文字下方，280帧中心立柱上方变成卡通图案，说明实例发生切换。128裁片只见相似立柱、网格与道路白线，256上下文才更容易揭示错误。它在N12d已因围栏/道路深度交界判为不确定，从未进入N12g拟合。因此不能将它归为N12g面部错位原因。',
 '保留全部原坐标、视频SHA、族标签和观察身份，复用已有material_observations122.py准入模块。280帧身份记为contradicted，早两帧仍unknown，未人为确认；3个观察全部被拒绝拟合，整族留出一致。此为只读模块应用，尚未声称整套训练调用链已经验证。',
 '本图使用原生context_048_family_4863与context_058_family_4936中248、264、280帧的256像素裁片，组成768×512，无重采样、几何变形或新增图中文字。上排表示远楼支持，下排展示重复结构身份反例。图例用于解释审阅证据，不用于挑选准确率。',
 '结论仅限现有7帧轨迹：没有建立新的可信近人物、头部高度固定材料支持。灯具可见但现有循环轨迹为零；画面存在物体不等于它已提供可靠测量。不能据此断言整个视频没有固定点，也不能排除相机模型偏差或证明人物微动。',
 '下一步复核现有相机与训练准入消费者，确保正深度、循环一致、低残差不能覆盖物理身份反证与支持不足。复用已有证据格式，使用这一实际反例。当前候选冻结，不重复求解。新实验需新的可辨识变量或材料证据。未生成新点云，未启动训练。',
 '来源：robust-portrait-122/video-update-20261027/clip03-head-level-support-plan.json，clip03-head-level-support/result.json、visual-index.json、visual-review.json、observation-contract-check.json、figure-index.json。以下保留历史诊断：'
]
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[b:]
s=s.replace('02_有限视角重建优化路线_122相机空间敏感度.pptx','02_有限视角重建优化路线_122头部固定点支持.pptx')
(H/'update_deck_headSupport122.mjs').write_text(s,encoding='utf-8')
(H/'preserve_deck_headSupport122.py').write_text((H/'preserve_deck_whiteCamera122.py').read_text(encoding='utf-8-sig').replace('whiteCamera122','headSupport122'),encoding='utf-8')
a=(H/'accept_white_camera122.py').read_text(encoding='utf-8-sig').replace('whiteCamera122','headSupport122').replace('clip03-camera-observability','clip03-head-level-support').replace('robust122-n12h','robust122-n12i').replace('N12h','N12i')
a=a.replace('N12i heading,4000basis,conditional unit-noise sensitivity table and3nativecrop rows actuallyvisible','N12i heading,4000basis,60family support review andfar-facade/repetitivepost nativecontexts actuallyvisible')
a=a.replace('N12i once-only background information and174target transfer sensitivities reviewed. Face/feet unit-noise gain ratio7.079; no actual uncertainty or unique-cause claim. ActualUI andsamePPT52completed;no camera/modelmutation ortraining.','N12i60nativebackgroundfamilies all reviewed; only provisionalnear-depth4936switchespostinstance at280, notusedN12g. ExistingN3contractrejects. ActualUI andsamePPT52completed;no camera/modelmutation ortraining.')
a=a.replace('latest_white_camera_sensitivity_PPT','latest_head_static_support_PPT').replace('Actual N12i local camera sensitivity report','Actual N12i headlevelsupport andpostidentity report')
(H/'accept_head_support122.py').write_text(a,encoding='utf-8')
print('N12I_DECK_PREPARED')
