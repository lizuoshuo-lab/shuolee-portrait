from pathlib import Path
import json,hashlib,shutil
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
prefix='surface123-articulation';old='surface123-simcc'
source=REC/'grayT-facial-articulation-review/frame_080_raw_observed_neutral_shared_expression_native.png'
figure=H/(prefix+'-native-figure.png');shutil.copy2(source,figure)
(H/(prefix+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=1280,height=320,sources=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_simcc.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-observation-deck-receipt.json')","read(build+'/surface123-simcc-deck-receipt.json')")
code=code.replace('grayT-simcc-decoder-review/actual-review.json','grayT-facial-articulation-review/actual-review.json')
start=code.index("replace('面部关键点精度与可见性不足'");end=code.index('const images=',start)
code=code[:start]+"""replace('连续解码未解决SMPL-X错位','增加表情仍未修复SMPL-X错位');
replace('6帧鼻点位置变化中位0.84像素，最大1.82像素。','固定相机对照：面部残差中位43.99→48.69像素。');
replace('逆仿射误差小于0.0001像素，遮挡与材料对应仍需验证。','同视频两轮内参一致，皮肤留出误差仍约19像素。');
replace('第80帧：原图、原解码青点、连续解码橙点。原拟合保持。','原图／观测青点／原模型紫点／共享表情候选橙点。');
"""+code[end:]
code=code.replace('第80帧三个原生160像素鼻部裁片，左原图、中原argmax青点、右连续解码橙点。没有图内文字，没有形变。','第80帧四个原生320像素裁片，依次原图、观测青点、原模型紫点、共享表情与下颌候选橙点。图内没有文字，没有形变。')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((REC/'facial-articulation-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],con['next'],
    '独立CPU单项对照，95帧、29个活跃鼻嘴语义点，2732个正权重观测。共享身体体型、根姿态、世界位置/尺度、K/E、所有观测与权重冻结。仅变化10维共享表情与3维共享下颌，用3*tanh和0.35*tanh限制；不是每帧动态模型，不是新的正式人物拟合。',
    '零参数回放与原输出的最大投影差0.000338710px，顶点差5.45e-7世界单位，13参数自动微分与中央差分的目标梯度最大差9.61e-11。仅验证计算，没有验证真实标定。',
    '48次目标计算，单次5.48秒CPU。加权鲁棒目标4.7750856→4.6460193（下降2.7029%）。脸部语义残差中位43.9922→48.6926px；鼻部45.8649→46.3916px；嘴部43.3939→50.2146px。不能用鲁棒损失下降代替逐点质量结论。',
    '9个表情维接近预先登记的边界，未放宽边界、未重新选点、未再次运行。6帧原生320px对照已实际审阅，侧脸鼻嘴仍落在脸颊，未发布候选模型，未开启新高斯训练。',
    '00对话独立复现路径：/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2。研究侧只读，没有改动它的源码或结果。该对话独立负责其18k训练、导出与质量审阅。',
    '两路源帧序号/时间一致，95帧。研究侧1520x2704，00为899x1600。严格半像素变换回原生坐标，有效K最大差2.2737e-13。第一帧外参均为单位矩阵；相对旋转差中位1.54359度、P90为6.62733度、最大6.88835度。这个差异不是相机真值误差，也不是人的运动角度。',
    '两路同语义观测差中位1.25164px、最大4.96115px；00面部语义拟合残差原生17.7061px，低于研究侧43.9921px。但输入分辨率、特征、遮罩和代码同时变化，不是单因素实验。',
    '复用既有M0皮肤候选的41个原生观测，仍用40/48/56/64/72/80六个ANCHOR，只在各自冻结相机下拟合一个静态3D点；35个HELD全部保留、不进入点拟合。研究侧中位18.8325px最大33.0977px，00中位19.1477px最大30.2174px。两边35/35均>=1px，不能称亚像素对齐。既有留出点被复用，是次级验证，不是新增独立真值。',
    '证据：'+str(REC/'facial-articulation-conclusion.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123连续解码核查.pptx','123表情与跨轮核查.pptx').replace('preserve_deck_surface123_simcc.py','preserve_deck_surface123_articulation.py')
(H/'update_deck_surface123_articulation.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_simcc.py','preserve_deck_surface123_articulation.py'),('commit_surface123_simcc_deck.py','commit_surface123_articulation_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared native slide52 edit; canonical unchanged.')
