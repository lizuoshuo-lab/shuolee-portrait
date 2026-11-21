from pathlib import Path
from PIL import Image
import hashlib,json,shutil
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-fixed-direction';new='surface123-face-observation'
source=R/'grayT-face-observation-cross-check/frame_079_native_face_and_skin_replay.png'
figure=H/(new+'-native-figure.png');shutil.copy2(source,figure);w,h=Image.open(source).size
(H/(new+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=w,height=h,sources=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_fixed_direction.mjs').read_text(encoding='utf-8').replace(old,new)
code=code.replace("read(build+'/surface123-scene-support-deck-receipt.json')","read(build+'/surface123-fixed-direction-deck-receipt.json')")
code=code.replace('grayT-fixed-direction-review/actual-review.json','grayT-face-observation-cross-check/actual-review.json')
a=code.index("replace('头部附近");b=code.index('const images=',a)
code=code[:a]+"""replace('单一栏杆方向无法验证全部外参','脸部观测改善了误差，但仍不是静态对齐');
replace('两组留出方向RMS约0.57和0.59像素，仍不能保证人脸对齐。','同一皮肤点35留出中位5.12→2.42像素，32个仍≥1像素。');
replace('数学控制：绕栏杆方向额外旋转7°，线方向预测保持不变。','当前眼部权重为0，结果含逐帧变化，不能当作静态验收。');
replace('原图／边缘测量／两套相机预测。最大线偏差2.35像素。','原图／RTMW有效点／FAN有效点／同一皮肤点回放。');
"""+code[b:]
code=code.replace('第80帧原生裁片，依次为原图、有效边缘测量、研究与00冻结相机方向预测。图内无文字，原图比例不变。预测线通过观测中心，不能检验绝对位置。','第79帧的4个原生320像素裁片：原图、RTMW有效鼻嘴点、FAN有效鼻嘴点、同一皮肤点观测与逐帧变形回放。绿圈观测、紫叉RTMW对照、橙叉FAN对照。图内无文字，未变形。')
a=code.index('p.resolve(slide.id).speakerNotes.text=');b=code.index('const candidatePath=',a)
d=json.loads((R/'face-observation-conclusion.json').read_text(encoding='utf-8'))
notes=[d['summary'],d['next'],d['baseline_note'],
 '这是00对话已有两次逐帧拟合的只读回放，没有新特征、推理、三维点拟合、相机优化或高斯训练。全部源哈希保持，95帧与原M0的6anchor/35HELD划分匹配。',
 '两组回放最大差分别0.00004334和0.00005796原生像素。FAN皮肤留出最小0.6784、中位2.4173、P90 3.5839、最大3.7972像素，32/35仍≥1像素。RTMW中位5.1193、最大9.5693、35/35≥1像素。',
 '全体2752个有效语义观测中，鼻部852、嘴部1900；眼部和眉部均0。FAN原生语义中位1.757、最大7.250像素，RTMW为2.299、13.636。FAN报告的1.039中位属于899×1600工作坐标，不是原生1520×2704像素。',
 '每帧19个颈部、头部、下颌、表情变量，共1805。40至80帧FAN颈部参数最大两两差20.77度、头部25.79度；这些包含模型和相机误差吸收，不能冒充真实微动测量。其他身体关节只出现1.2e-7数值差。',
 '原RTMW拟合未保留fit-script.py，FAN源码哈希7f1771d0fa962d9252f3ad9a479c9ceecfe9c2cb045a7b8545e79355d383785a与报告吻合。未称严格单因素实验。',
 '6个原生整数裁片41/49/57/65/73/79实际审阅；闭眼和侧脸遮挡存在。该皮肤标记重复用于前序验证，不是新封存材料，也不覆盖全脸或未见人物。',
 '证据：'+str(R/'face-observation-conclusion.json')]+d['observations']
code=code[:a]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[b:]
code=code.replace('123固定方向核查.pptx','123面部观测回放.pptx').replace('preserve_deck_surface123_fixed_direction.py','preserve_deck_surface123_face_observation.py')
(H/'update_deck_surface123_face_observation.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_fixed_direction.py','preserve_deck_surface123_face_observation.py'),('commit_surface123_fixed_direction_deck.py','commit_surface123_face_observation_deck.py')]:
 (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,new),encoding='utf-8')
print('Prepared slide52 edit; canonical unchanged.')
