from pathlib import Path
import json
H=Path(__file__).parent
s=(H/'update_deck_pattern_consumer123.mjs').read_text(encoding='utf-8')
s=s.replace("tag='pattern-consumer123'","tag='camera-entry123'")
s=s.replace("b+'/pattern-retained123-deck-receipt.json'","b+'/pattern-consumer123-deck-receipt.json'")
s=s.replace('pattern-preflight/native-material/bridge-control/physical-review/consumer-audit/result.json','camera-observation-entry/result-completed.json')
s=s.replace("if(review.state!=='stopped_at_registered_consumer_gap'||review.direct_consumer_integration_completed!==false||review.family_conflicts!==3)throw Error('Consumer audit missing');", "if(review.total_checks!==26||review.geometric_solver_executed!==false||review.quality_accepted!==false)throw Error('Entry regression evidence missing');")
a=s.index("replace('闭环匹配");b=s.index('const figure=',a)
s=s[:a]+"""replace('审阅结果尚未接入相机入口','错误对应已在相机入口被拒绝');
replace('旧相机脚本会重新提特征','原始像素和FIT／HELD保留，身份未知的点继续保留为未知。');
replace('严格准入已阻止新拟合','26项工程检查通过。尚无实拍相机或人体对齐通过证据。');
"""+s[b:]
notes=[
'本轮修复实际local_surface123.prepare调用路径。新增冻结观测读取器和单独的reviewed相机入口，先验证源文件及角色证据再创建运行根。旧相机文件保持原字节，新几何部分与旧seed_pairs之后的计算完全一致。没有重新检测、匹配、优化相机、拟合SMPL-X或训练高斯。',
'真实video4数据共99原生帧、42609个原始段、95886个观测。读取时保留原始slot、源帧、图像SHA、特征SHA、keypoint编号、native_xy、原始FIT/HELD与development标志，不从ROI推断物理身份。',
'三条已确认物理矛盾的FIT轨迹族关联10段和35个观测。新入口实际记录family_physical_identity_conflict并在求解前拒绝。原8527条HELD段全部留出。未知角色没有改为静态背景，当前可准入静态背景FIT数仍为0。',
'初始18项测试覆盖真实错误族/原始像素划分、实际相机入口拒绝、真实prepare缺省/真实输入拒绝、合成prepare到新相机接口的传输、源帧/像素/划分/文件篡改等。合成正例使用几何求解器spy，实际几何求解调用数为0，不构成物理标定或训练验收。',
'代码复核随后补齐证据范围检查，另8项合成检查通过。支持文件必须绑定同一视频、轨迹族、分段、全部原始节点、源帧、特征与图像字节、原生像素，并有可见性及测量不确定性声明。检查通过只说明证据引用范围一致，不能由解析器证明声明的物理真伪。',
'图沿用已经实际审阅的335、375、415源帧裁片。完整三角闭环也会连接脸、椅架与门把手。图片无图内文字或扭曲。本轮是消费接口修复，不把这条候选错误当作旧SMPL-X全部错位的唯一原因。',
'下一步按已授权field队列做一次177帧已有YOLO/RTMW区域覆盖核验，计划已登记但未执行。只用现有官方权重定位区域，后续材料对应必须独立核验，不以相同面部编号充当同一皮肤点。',
'当前各人物仍未通过独立面部逐点严格小于1像素及全身保护。精确输出K、外参、物理对应与人物微动仍需分开诊断。此次没有发布新点云或改变历史PLY/camera，没有00部署或WPS云同步验证。',
'证据：local-clothed-surface-123/camera-observation-entry/result.json、result-completed.json、real-source-audit.json、actual-camera-denial.json、code-manifest.json、implementation.json，及field-preflight/coverage-plan.json。原PPT只改第52页，前51页保留。'
]
a=s.index('notes.text=');b=s.index(".join('\\n\\n');",a)+len(".join('\\n\\n');")
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');"+s[b:]
s=s.replace('02_有限视角重建优化路线_123消费接口核验.pptx','02_有限视角重建优化路线_123冻结观测入口.pptx').replace('preserve_deck_pattern_consumer123.py','preserve_deck_camera_entry123.py')
(H/'update_deck_camera_entry123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_pattern_consumer123_deck.py').read_text(encoding='utf-8').replace('pattern-consumer123','camera-entry123')
(H/'commit_camera_entry123_deck.py').write_text(c,encoding='utf-8')
print('Prepared slide52 edit, canonical deck unchanged.')
