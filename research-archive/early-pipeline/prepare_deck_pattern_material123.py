from pathlib import Path
import json
H=Path(__file__).parent;tag='pattern-material123'
s=(H/'update_deck_pattern_coverage123.mjs').read_text(encoding='utf-8')
s=s.replace("tag='pattern-coverage123'","tag='pattern-material123'")
s=s.replace("b+'/video10-native-patch123-deck-receipt.json'","b+'/pattern-coverage123-deck-receipt.json'")
s=s.replace("pattern-preflight/coverage/visual-review/review.json","pattern-preflight/native-material/visual-review/review.json")
s=s.replace('!review.actual_saved_native_crop_review||!review.all99_context_and_face_tiles_reviewed','!review.actual_saved_pixels_reviewed||!review.actual_all10_false_track_pixels_reviewed')
a=s.index("replace('局部细化");b=s.index('const figure=',a)
s=s[:a]+"""replace('脚部预测落在桌面和上衣','跨视角匹配把衣服连到了眼睛');
replace('99帧核验','原生纹理双向匹配把衣服与眼部合并为同一轨迹。');
replace('人体mask内','桥接误连已隔离，剩余闭环尚未通过物理对应与亚像素验收。');
replace('第120、245、370帧','第0、120、125帧，同一轨迹的原生192px裁片。查询在中心附近。');
"""+s[b:]
s=s.replace('pattern-preflight/coverage/visual-review/invalid_foot_locations_native.png','pattern-preflight/native-material/visual-review/row_07_native.png')
a=s.index("alt:'");b=s.index("'});",a)
s=s[:a]+"alt:'同一跨视角轨迹的第0、120、125帧原生192px裁片。第一个点在衣服图案，后两个点在眼部。图片未加文字或扭曲。'"+s[b+1:]
notes=[
'2026-10-29。只更新第52页，前51页保留。video4.mp4花纹上衣人物，99个原生2704x1520帧，stride5，从491帧中预先选定。输入已去畸变，不二次去畸变。',
'一次SIFT原生区域提取和464对双向L2 ratio0.75匹配，共396237候选与359252匹配边。1.5px去重、3px近重复族划分均在几何之前冻结。15684条未隔离轨迹共95886个观测。',
'实际审阅预先选定的30组三视角原生裁片，以及错误轨迹全部10个原生裁片。第7组先在衣服图案，随后在眼部，确认不是同一物理点。不能把其他29组的视觉相似当成物理认证。',
'错误桥接为原帧5/key421到85/key9。前向描述子比值0.7238906，反向0.7474796，均通过0.75阈值。二维距离614.466px不是重投影误差，筛选没有使用这个距离。该边没有三角闭环且为桥；整条轨迹其他部分有5个三角闭环，因此仅查整条轨迹有闭环会漏掉这一错误。',
'本页显示同一轨迹的原帧0、120、125，三个192x192原生裁片，无图内文字或几何扭曲。查询位于各裁片中心附近。实际错误边5到85的完整逐边证据保存在false-track-trace-reviewed.json。',
'同一固定观测做一次拓扑控制：隔离26925条桥接边，保留9661个至少三视角的段、62938个观测。32948个短段观测保留为待核验。桥接边并非全部错误，保留闭环仍可能是重复图案的错误对应。未重新提取或匹配、未修改K或二维位置。',
'FIT保留7699段，HELD1962段，其中1946段尚未被当前人工查看。所有原始节点、边和近重复家族及FIT/HELD不变。自写Tarjan和独立NetworkX在全部15684个处理图的桥与连通分段上完全一致。合成控制包含两环经桥相接、完整三角形、链及输入顺序不变。实现正确不等于物理正确。',
'固定30组查询中14组仍共同落在有闭环的段内，16组失去完整支持。这是覆盖变化，不是准确率。例7被拆为3个单点及一个7视角眼部段，衣服和眼睛不再合并。',
'明确反例是HELD，从未进入新相机或SMPL-X。它说明身份合并弱点，不证明旧拟合唯一根因。30个已审阅家族全部作为开发证据，保持原FIT/HELD，不用其验收新规则的独立精度。排除这些家族后原帧490未查看的HELD头部候选为0，缺口保留。',
'上一轮确认脚点误标到桌面、上衣和袖子。关节语义、跨视角物理身份、相机和人物微动应分别核验。当前pattern没有相机、SMPL-X或高斯训练。独立物理可信面部留出0，不声称1px达标。',
'下一步在固定段上检查193个头部家族的物理身份，单次预算最多160，剩余显式未审；背景最多40家族。没有新特征/权重/匹配/相机。材料支持不足停止该人物候选，不自动放宽门槛或补点。',
'证据：local-clothed-surface-123/pattern-preflight/native-material/decision.json、visual-review/review.json、false-track-trace-reviewed.json、bridge-control/result.json、bridge-control/physical-review-plan.json。右侧仅更新诊断，历史PLY不变，未验证00部署或WPS云往返。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123遮挡关键点核验.pptx','02_有限视角重建优化路线_123跨视角身份核验.pptx')
s=s.replace('preserve_deck_pattern_coverage123.py','preserve_deck_pattern_material123.py')
(H/'update_deck_pattern_material123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_pattern_coverage123_deck.py').read_text(encoding='utf-8').replace('pattern-coverage123',tag)
(H/'commit_pattern_material123_deck.py').write_text(c,encoding='utf-8')
print('Prepared single slide52 edit; canonical deck not changed.')
