from pathlib import Path
from PIL import Image
import hashlib,json,shutil
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-articulation';prefix='surface123-scene-support'
source=REC/'grayT-scene-support-cross-run-review/company00_FIT_02_d0326813da_actually_used_BA.png'
figure=H/(prefix+'-native-figure.png');shutil.copy2(source,figure)
w,h=Image.open(source).size
(H/(prefix+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=w,height=h,sources=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_articulation.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-simcc-deck-receipt.json')","read(build+'/surface123-articulation-deck-receipt.json')")
code=code.replace('grayT-facial-articulation-review/actual-review.json','grayT-scene-support-cross-run-review/actual-review.json')
start=code.index("replace('连续解码");end=code.index('const images=',start)
code=code[:start]+"""replace('增加表情仍未修复SMPL-X错位','头部附近场景验证仍不充分');
replace('固定相机对照：面部残差中位43.99→48.69像素。','两组已有留出残差中位1.19和1.50像素，材料身份待核实。');
replace('同视频两轮内参一致，皮肤留出误差仍约19像素。','局部区域实际BA支持为0条与1条，尚不能据此判定外参错误。');
replace('原图／观测青点／原模型紫点／共享表情候选橙点。','00复现的一条栏杆轨迹：54、58、61帧。原图保持比例。');
"""+code[end:]
code=code.replace('第80帧四个原生320像素裁片，依次原图、观测青点、原模型紫点、共享表情与下颌候选橙点。图内没有文字，没有形变。','00复现实际进入相机BA的同一栏杆候选轨迹，54、58、61帧的原生256像素方形裁片。青圈定位，没有图内文字，没有形变。材料身份未通过精细验证。')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((REC/'scene-support-cross-run-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],con['next'],con['native_threshold_caveat'],
 '只读审阅两轮全部95帧的原始轨迹、最终BA成员及既有留出，未提取新特征、未拟合三维点、未优化相机。全部源哈希保持。预登记头部附近图像ROI来自帧40至80，不代表物理深度。',
 '研究侧此区域实际BA为0条/0观测，00复现1条/7观测。已有留出分别284条/398观测与556条/839观测。原生像素误差中位1.1942364494和1.5033483312，P90为26.45491548和31.19300769，最大279.85318895和14045.83494935，至少1像素220/398和510/839。包含非正估计深度22/50个观测，未按误差筛除。',
 '非正深度或较大留出误差可能由错误材料对应、弱三角化或相机误差引起，不是单一外参错误的证明。局部支持少也不是相机不可估计的充分条件，其他区域固定点仍能约束外参。',
 '实际审阅14组三视角裁片（42个裁片位置），含11组原始分量、3组实际BA节点。原始FIT树木与路面轨迹的错误末端分别57/54帧未进入BA，不能将原始错误末端归为优化污染。实际树叶仍未证明静止，栏杆具有重复和平滑纹理歧义。',
 '00背景mask排除检测到的人物后，全部保存特征均处于允许区域，违规数0。背景不等于天然静止或材料身份确定。',
 '本页图像为00实际BA的d0326813da630086ad4eca3286b6246e569944c2f25a451b3a3609222779777e轨迹，全部7帧54/55/56/58/59/60/61保留，抽检54/58/61。图中支架固定结构看似相同，亚像素材料对应尚未认证。',
 '当前灰衣研究训练未准入，历史白T和蓝衣研究模型保留。00对话另有授权的独立实验，不作为本静态目标已通过。',
 '证据：'+str(REC/'scene-support-cross-run-conclusion.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123表情与跨轮核查.pptx','123场景支持核查.pptx').replace('preserve_deck_surface123_articulation.py','preserve_deck_surface123_scene_support.py')
(H/'update_deck_surface123_scene_support.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_articulation.py','preserve_deck_surface123_scene_support.py'),('commit_surface123_articulation_deck.py','commit_surface123_scene_support_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared slide52 edit; canonical unchanged.')
