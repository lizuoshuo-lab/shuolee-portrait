from pathlib import Path
from PIL import Image
import json
H=Path(__file__).parent;Q=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/video10-correspondence-contrast')
tag='video10-matcher123'
Image.open(Q/'review/contexts/row_042.png').save(Q/'matcher-evidence.png')
s=(H/'update_deck_video10_support123.mjs').read_text(encoding='utf-8').replace("tag='video10-support123'",f"tag='{tag}'")
s=s.replace("/robust-portrait-122/video10-semantic-role'","/robust-portrait-122/video10-correspondence-contrast'")
s=s.replace("b+'/video10-semantic123-deck-receipt.json'","b+'/video10-support123-deck-receipt.json'")
s=s.replace("root+'/material-visual-review.json'","root+'/visual-review.json'")
s=s.replace("if(!review.actual_saved_visual_review||review.reviewed_rows!==113)","if(!review.reviewed_all72||review.rows.length!==72)")
a=s.index("replace('人体框误排背景'");z=s.index('const notes=',a)
s=s[:a]+"""replace('视频10的材料对应','遮挡误配也能通过往返检查');
replace('113族已审阅','冰箱背景点转到人物，往返误差仅0.08px。');
replace('连接计数最多','固定置信度门槛排除此例，物理对齐仍未通过。');
replace('空调局部','第644／698帧。左原点，中RootSIFT，右RoMa。裁片中心为查询位置。');
const figure=root+'/matcher-evidence.png';
const image=p.resolve(last.elements.find(e=>e.kind==='image').id);
image.replace({blob:new Uint8Array(await fs.readFile(figure)),contentType:'image/png',fit:'contain',alt:'同一原生查询点的384像素裁片。左原冰箱背景，中冻结RootSIFT对应，右RoMa对应。目标被人物遮挡，不能将小往返误差当真实物理点对齐。图内没有新增文字。'});
image.frame={left:66,top:180,width:1148,height:1148/3};image.crop={left:0,top:0,right:0,bottom:0};image.fit='contain';image.lockAspectRatio=true;
"""+s[z:]
a=s.index('notes.text=');z=s.index('const candidatePath=',a)
notes=['2026-10-29 视频10同一原生查询的匹配方法对照。',
'只改变对应算子。原图、SIFT查询位置、FIT/HELD族划分、K、相机、人体均保持不变。固定两个三帧片段484/510/537和644/671/698，以及806/859粗背景控制，共7对。复用本机官方RoMa Outdoor和DINOv2权重，不下载新模型。',
'7对仅一次推理，70.48秒，峰值显存5829209600字节。保存所有原始预测与置信度，双向和三视角循环检查只度量自洽。',
'72个预登记位置、12页原生裁片和16组扩大上下文已全部实际审阅。筛查固定为双向置信度>=0.9，往返<1原生px。15个通过，其中8个可辨认粗背景、1个粗人物、6个仍有反射或重复纹理歧义。12个原有粗背景控制仅4个通过这一门槛。未调整阈值。',
'图为第42行。644帧冰箱篮子/磁贴附近的背景在698帧被人遮住，目标落到脸颊或衣领附近。RoMa往返约0.08px，但置信度0.8375低于0.9，因此本轮固定门槛拒绝。这不证明通过该门槛的点均为正确物理身份。',
'头部矩形仍不能代替真实面部身份。其中一些查询实际落到柜门照片、飞镖盘或轮廓旁背景，不能用于人脸亚像素验收。后颈/衣领的粗匹配也不能当眼鼻嘴真值。',
'第58行眼镜/眼周粗局部为FIT，已实际审阅671/644/698三帧和返回671。两种方向循环均通过，闭合误差0.174177和0.074986原生px。这是有效的粗对应线索，不是独立物理重投影误差；镜片反光和精确表面点仍未认证。独立有效面部HELD为0。',
'全量预测中出现到相反划分旧族的邻近关联，均单独标记，未把HELD转为FIT，也未改旧族。这只是近邻隔离，不认证真实材料身份。',
'这一7对探查不能代表整视频相机已注册，也不能唯一解释旧灰衣SMPL-X重影。没有新相机、SMPL-X拟合或高斯训练。',
'新盘点发现636个原语义候选族中，已审阅代表208族；18个支持薄弱视角还涉及222个尚未完成全观测审阅的候选族（187FIT/35HELD）。下一步只审阅既有缓存，不重跑特征、RoMa或扫阈值。此前5/33是已审阅粗支持的计数范围，不能推广为所有原候选都不可用。',
'证据：'+str(Q/'visual-review.json')+'；'+str(Q/'decision.json')+'；'+str(Q/'run/result.json')]
s=s[:a]+'notes.text='+json.dumps(notes,ensure_ascii=False)+".join('\\n\\n');\n"+s[z:]
s=s.replace('02_有限视角重建优化路线_123视频10材料连接核验.pptx','02_有限视角重建优化路线_123视频10匹配对照.pptx')
s=s.replace("b+'/preserve_deck_video10_support123.py'","b+'/preserve_deck_video10_matcher123.py'")
(H/'update_deck_video10_matcher123.mjs').write_text(s,encoding='utf-8')
c=(H/'commit_video10_support123_deck.py').read_text(encoding='utf-8').replace('video10-support123',tag)
(H/'commit_video10_matcher123_deck.py').write_text(c,encoding='utf-8')
print('Prepared matcher slide52 edit; canonical deck unchanged.')
