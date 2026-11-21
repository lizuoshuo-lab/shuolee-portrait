from pathlib import Path
import json
H=Path(__file__).parent
s=(H/'update_deck_pattern_retained123.mjs').read_text(encoding='utf-8')
s=s.replace("tag='pattern-retained123'","tag='pattern-consumer123'")
s=s.replace("b+'/pattern-material123-deck-receipt.json'","b+'/pattern-retained123-deck-receipt.json'")
s=s.replace("pattern-preflight/native-material/bridge-control/physical-review/review.json","pattern-preflight/native-material/bridge-control/physical-review/consumer-audit/result.json")
s=s.replace("if(!review.actual_saved_pixels_reviewed||review.confirmed_contradictions_FIT!==3)throw Error('Review missing');", "if(review.state!=='stopped_at_registered_consumer_gap'||review.direct_consumer_integration_completed!==false||review.family_conflicts!==3)throw Error('Consumer audit missing');")
a=s.index("replace('跨视角匹配");b=s.index('const figure=',a)
s=s[:a]+"""replace('闭环匹配仍可能连接不同物体','审阅结果尚未接入相机入口');
replace('脸部、椅架','旧相机脚本会重新提特征，不读取已经审阅的轨迹。');
replace('发现3条错误FIT段','严格准入已阻止新拟合；错误对应尚未在消费端完成修复。');
"""+s[b:]
notes=[
'输入仍是video4花纹上衣人物。上轮实际审阅160头部家族和40背景家族，确认3个FIT段物理身份矛盾。图为源帧335、375、415的192px原生裁片，完整三角闭环仍把脸、椅架、门把手连接。',
'本轮按既有consumer-evidence-plan只读检查真实消费接口，不新增检测、匹配、相机、人体先验、SMPL-X或高斯训练。原始审阅、轨迹和特征清单SHA保持不变。',
'3个矛盾家族在原始分段表中关联10段、35个观测，逐条保留原source frame、slot、keypoint、图像SHA、特征SHA和浮点native_xy。家庭内存在矛盾不等于每个观测都是错点，故仅记录family conflict，点级身份仍保留unknown。',
'实际material_observations122接口只允许fit_background、heldout_background和heldout_human。新的原始划分为FIT/HELD，混有人物与场景候选；ROI矩形不是物理语义。对三条真实反例保留原始FIT后调用接口，均因Frozen split/family required而拒绝。这是格式/语义不匹配拒绝，不是接口已经读懂物理矛盾。',
'实际local_surface123_camera.py中的camera函数会重新执行SIFT检测和双向匹配，不读取本次已保存segments或physical-review，也不调用N3物理证据准入。因此不能宣称桥接隔离或错误家族已经进入生产相机求解。该旧脚本本轮未运行。',
'实际检查pattern运行根仍不存在；调用现有require_camera_admission在CPU上明确拒绝camera_alignment_receipt_missing。没有创建运行根、导入torch或调用SMPL-X。这一保护已存在，本轮没有将它冒充新修复。',
'依照预登记停止条件，停止这次接口适配，保存audit-only family-evidence.json和result.json，不改FIT/HELD、不把未知人物点贴成静态背景、不新增训练输入。完成的是接口缺口定位，不是对齐改善或pipeline验收。',
'原有123七人物队列还包含未启动的田地深色上衣field，对应内容SHA a028323a751e69ab3236bb3879b636159780906cf34acf19dc2edc2bebfb30a8，现名video7.mp4。继续按已授权队列做一次原生输入核验，固定177帧及5个展示时刻；没有调用相机或人体训练。具体完成状态以field-preflight/result.json与实际图像审阅为准。',
'当前强鲁棒性目标仍未完成。精确输出K/残余相机误差、物理身份、人物微动仍需分开；独立有效面部留出逐点严格小于1像素尚无通过证据。右侧仅更新诊断，历史点云与camera保持，未验证00部署或WPS云往返。',
'证据：local-clothed-surface-123/pattern-preflight/native-material/bridge-control/physical-review/consumer-audit/result.json、family-evidence.json、execution.json；上一轮review.json与contradiction-trace.json。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123闭环物理核验.pptx','02_有限视角重建优化路线_123消费接口核验.pptx')
s=s.replace('preserve_deck_pattern_retained123.py','preserve_deck_pattern_consumer123.py')
(H/'update_deck_pattern_consumer123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_pattern_retained123_deck.py').read_text(encoding='utf-8').replace('pattern-retained123','pattern-consumer123')
(H/'commit_pattern_consumer123_deck.py').write_text(c,encoding='utf-8')
print('Prepared last-slide edit. Canonical deck unchanged.')
