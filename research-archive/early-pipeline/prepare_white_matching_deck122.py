"""Prepare one-slide Artifact Tool edit, retaining all51 other slides."""
from pathlib import Path
import json
H=Path(__file__).parent
s=(H/'update_deck_support122.mjs').read_text(encoding='utf-8')
s=s.replace("const figure=evidence+'/scene-support-supplement/material_evidence.png';", "const figure=evidence+'/../clip03-native-matching/material_evidence.png';")
s=s.replace("build+'/quarantine122-deck-receipt.json'", "build+'/SOURCE_RECEIPT_TOKEN.json'",1)
s=s.replace('support122','whiteMatching122').replace('SOURCE_RECEIPT_TOKEN','support122-deck-receipt')
s=s.replace('122背景材料核验','122白T材料对应核验')
a=s.index("replace('跨视角错配与重影根因'");b=s.index('const images=',a)
s=s[:a]+"""replace('人物周围的背景材料核验','白T恤人物的跨视角材料对应');
replace('三排依次为湿泥','三排依次为面部、衣服和植被，各为200、216、232帧。');
replace('新增606个候选族','104组已审阅，头部候选16组中6脸、1发际线、9衣服。');
replace('材料核验未达预登记条件','固定建筑材料支持不足，本轮未重算相机或启动训练。');
"""+s[b:]
s=s.replace("alt:'三个视角下的湿泥水边植被和地面原生纹理裁片'","alt:'面部衣服植被在200216232帧的原生128像素裁片'")
notes=[
 '2026-10-28 N12b：对白T恤新视频预登记184、200、216、232、248、264、280七个原生1520×2704视角执行一次SIFT/RootSIFT双向及三视角循环匹配。第一组YAML以4000×3000为当前输出基准，经下采样和居中裁剪映射，K保持不变。没有再次去畸变、二维形变、K估计或新相机求解。',
 '每帧8000个特征，背景上中下各2000，头部候选区域600，其他人物1400。5004个三视角循环形成2740条轨迹，近重复整族合并后1584族。1292个背景候选族冻结为1016拟合、276留出。整族近邻隔离不能自动证明独立物理身份。',
 '实际审阅预先按空间分箱、拟合/留出和最长轨迹排序选出的104组全部128像素原生裁片，72背景、16头部候选、16其他人物。对其中16组有固定物体歧义的背景查看256像素原生上下文。没有用残差筛选或移动留出观测。',
 '头部候选16组中，6组在实际脸部，1组发际线，9组是白衣服。脸部为4拟合候选与2留出候选，尚未校准测量不确定性或证明逐点小于1像素。眨眼尤其在200帧可见，这是局部非静态图像外观证据，不能直接归因为全身微动或重影唯一原因。',
 '72组背景实际材料分类为30植被、5树干或枝条轮廓、29地面纹理、5围栏/植被/建筑深度混合、2布横幅、1可辨固定建筑细节。上下文中的灯杆、消防栓、台阶不等于中心特征属于固定物体。预登记需要6个拟合和3个留出固定非地面材料，现明确的粗候选仅1/0，故本轮不启动相机拟合。即使所有29个地面候选都可用，此项仍不满足。',
 '这只描述已抽检样本，并非认定其余1220个未审阅背景族错误或不存在。下一步应只读检查已有观测池的固定建筑材料分配，区分最长轨迹取样偏差与实际观测缺口，不重复检测或扫描阈值。',
 '本页原生图像三排分别取预选row72族833、row74族801、row21族1165，每排固定200、216、232帧。每格128×128，不缩放、无文字、无图像对齐或形变。点位于裁片中心，第三排中心是植被而非旁边台阶。',
 '7帧是诊断子集，不等于464帧全部注册。本轮没有新的数值重投影误差、相机、训练、PLY或00端验证。119模型保留，帽子视频历史失败仍未解决。4000基准在N5/N6已经使用，不能宣称本轮切换4000解决了既有偏差。',
 '来源：robust-portrait-122/video-update-20261027/clip03-native-matching/result.json、visual-review.json、visual-index.json、context-index.json、material-evidence-index.json，执行前方案clip03-native-matching-plan.json。',
 '以下历史记录保留溯源：'
]
a=s.index('p.resolve(slide.id).speakerNotes.text=[');b=s.index('const candidatePath=',a)
s=s[:a]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[b:]
(H/'update_deck_whiteMatching122.mjs').write_text(s,encoding='utf-8')
(H/'preserve_deck_whiteMatching122.py').write_text((H/'preserve_deck_support122.py').read_text(encoding='utf-8').replace('support122','whiteMatching122'),encoding='utf-8')
print('N12B_DECK_CODE_READY_SOURCE_UNCHANGED')
