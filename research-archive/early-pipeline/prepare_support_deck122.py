from pathlib import Path
import json
H=Path(__file__).parent
s=(H/'update_deck_quarantine122.mjs').read_text(encoding='utf-8')
s=s.replace("const figure=evidence+'/material-quarantine/same_native_before_after.png';","const figure=evidence+'/scene-support-supplement/material_evidence.png';")
s=s.replace("build+'/region122v2-deck-receipt.json'","build+'/quarantine122-deck-receipt.json'")
start=s.index("replace('人像与背景的跨视角偏差'");end=s.index('const images=',start)
s=s[:start]+"""replace('跨视角错配与重影根因','人物周围的背景材料核验');
replace('绿圈为观测','三排依次为湿泥、水边植被和地面，各为三个原生视角。');
replace('隔离已核实','新增606个候选族，预选样本中有12个粗静止FIT地面候选。');
replace('单条错配','材料核验未达预登记条件，本轮未重算相机或启动训练。');
"""+s[end:]
s=s.replace("alt:'隔离背景错配前后相同原生观测的脸部上衣地面对照'","alt:'三个视角下的湿泥水边植被和地面原生纹理裁片'")
s=s.replace("image.frame=frame;image.fit='contain';","image.frame={left:frame.left+(frame.width-frame.height)/2,top:frame.top,width:frame.height,height:frame.height};image.fit='contain';")
start=s.index('p.resolve(slide.id).speakerNotes.text=[');end=s.index('const candidatePath=',start)
notes=[
'2026-10-28 N11：一次人物周围原生背景补提完成，4000×3000基准K、原7帧和所有旧观测/留出划分不变。只在之前粗矩形排除但SAM2人物轮廓加32像素保护带之外的区域提取。仍使用SIFT6000、contrast0.02、edge10、sigma1.6、RootSIFT、互为最近邻ratio0.75、步长1/2及三视角循环。没有新去畸变、二维扭曲或K优化。',
'每帧新增1518、1530、1541、1561、1871、1912、1621个原生特征。旧特征和描述子原样复用，原1125条轨迹和分组哈希核验未变。共1603个含新节点循环形成1122条候选轨迹。199条与旧物理候选族同帧8像素内相连，完整隔离。剩余923条轨迹归入606个新候选族，冻结为475FIT族和131留出族，未把旧留出转入拟合。',
'按执行前登记的3×3空间分箱与FIT/留出分层、族哈希排序，每格最多3族，选出42族代表，其中21FIT和21留出。在任何几何求解前已实际查看全部42组三视角160像素原生裁片，并对10组有岸线/湿泥/水面疑问的中心额外查看320像素上下文。上下文只扩大显示范围，没有新增检测或调整对应。',
'12个FIT族和12个留出族在显示的原生视角中保留较清楚且一致的地面/小石块纹理，可作为粗静止材料候选。未宣称其亚像素测量、所有中间帧或完整同族观察均已认证。其余18族涉及近枝、远处植被、岸线、湿泥或水边植被，物理中心与静止性仍有不确定。灰色纹理未自动判为移动水面，也未宣称这些轨迹全为错配。',
'预登记的粗诊断门槛要求20个新审阅FIT族、8个留出族、至少3空间箱及每帧5个新FIT观测。现已审阅地面候选为12FIT、12留出，FIT覆盖4个箱，7帧观测分别6、6、8、7、9、8、5。因此20FIT要求未满足，本次没有启动新相机拟合。这个门槛是本轮预登记预算约束，不是证明整段视频无法重建的数学结论；其他未审阅候选仍保留。',
'本页三排分别是原预选第9、10、7行，对应族2136、1476、2194。每排为同一候选首/中/末三个视角。每格160×160原生像素，点位于裁片中心，无图中文字、图像拉齐或形变。三排组成480×480方形图等比显示。第一排重复湿泥纹理不能仅凭外观认定唯一材料；第二排水边植被身份仍不确定；第三排可保留为粗地面纹理候选。',
'本轮没有产生新的面部误差或质量提升数据。N10的脸部候选中位3.143952像素、所有57个有效候选仍不小于1像素，是先前隔离族399后的诊断，不属于N11改善。没有新训练、PLY、相机发布或远端00部署核验，119模型保留。',
'下一项是独立处理新clip03白T恤人物数据，画面含近处台阶、围栏和远建筑。保持相同4000基准映射，冻结原生轨迹与整族留出，实际审阅固定物体身份后再判断相机注册。跨视频结果用于流程验证，不能作为仅改变一个因素的根因证明。帽子人物的未解问题保留。',
'来源：robust-portrait-122/video-update-20261027/background-geometry/scene-support-supplement/result.json、visual-review.json、visual-index.json、context-index.json、material-evidence-index.json。执行前方案为同级scene-support-supplement-plan.json。',
'以下历史说明保留溯源：']
s=s[:start]+'p.resolve(slide.id).speakerNotes.text='+json.dumps(notes,ensure_ascii=False)+".concat([previous]).join('\\n\\n');\n"+s[end:]
# Rename all new build artifacts while retaining the source receipt reference.
s=s.replace("build+'/quarantine122-deck-receipt.json'","build+'/SOURCE_RECEIPT_TOKEN.json'",1)
s=s.replace('quarantine122','support122').replace('SOURCE_RECEIPT_TOKEN','quarantine122-deck-receipt')
s=s.replace('122材料隔离诊断','122背景材料核验')
(H/'update_deck_support122.mjs').write_text(s,encoding='utf-8')
p=(H/'preserve_deck_quarantine122.py').read_text(encoding='utf-8').replace('quarantine122','support122')
(H/'preserve_deck_support122.py').write_text(p,encoding='utf-8')
print('Prepared native Artifact Tool edit; source deck untouched')
