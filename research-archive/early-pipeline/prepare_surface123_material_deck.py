"""One native edit of existing slide52. Preserve the other51slides and history."""
from pathlib import Path
import json,hashlib
from PIL import Image
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
prefix='surface123-material';old='surface123-camera-support'
paths=[R/f'grayT-skin-texture-probe/same_frame_{f:03d}_projection.png' for f in [58,60,62,65]]
image=Image.new('RGB',(1024,256))
for i,p in enumerate(paths):
    with Image.open(p) as v:assert v.size==(256,256);image.paste(v,(i*256,0))
figure=H/(prefix+'-native-figure.png');image.save(figure)
spec=dict(path=str(figure),width=1024,height=256,sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
(H/(prefix+'-figure.json')).write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_camera_support.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-alignment-deck-receipt.json')","read(build+'/surface123-camera-support-deck-receipt.json')")
code=code.replace("root+'/grayT-camera-support-review/actual-review.json'","root+'/grayT-skin-texture-probe/post-geometry-review.json'")
start=code.index("replace('SMPL-X错位与固定相机诊断'");end=code.index('const images=',start)
code=code[:start]+"""replace('SMPL-X错位与面部约束缺口','皮肤材料点仍有跨视角错位');
replace('灰衣95帧：相机优化中9个脸区观测，逐点审阅均为背景。','灰衣：35个留出视角中位18.9像素，最大32.3像素。');
replace('25条脸区候选轨迹未进入最终相机优化；人体拟合仍约44像素。','同帧地面观测0.16–0.96像素，相机残差与人物微动仍需分离。');
replace('鼻部候选轨迹的原生三视角。全图误差0.39像素不能证明面部准确。','青圈：皮肤点观测。橙十字：固定相机下同一静态三维点的投影。');
"""+code[end:]
code=code.replace('height=360,width=height*figureSpec.width/figureSpec.height','width=1148,height=width*figureSpec.height/figureSpec.width')
code=code.replace('同一鼻部候选轨迹的三个原生裁片和独立点位标记，图内没有文字','原帧58、60、62、65的皮肤材料点观测与冻结相机静态三维点投影，图内没有文字')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((R/'material-consistency-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],
 '本次单变量观察变更：由眼睑、嘴角、轮廓候选改为明确可见的皮肤深色标记。原生帧40–80共41帧，6个预登记锚点拟合一个静态三维点，35帧独立于该点拟合。原K/外参、原SIFT分组、原几何、SMPL-X和训练参数未修改。',
 '皮肤跟踪以31pxLK窗口、3层金字塔、固定阈值运行一次。往返最大0.451923px只反映算法重复性，原生像素检查显示仍可能有数像素中心漂移。不能声称物理定位已经小于1像素。',
 '预先6锚点的静态三维点拟合中位15.944px。35留出中位18.925px、P9027.613px、最大32.345px，全部大于10px。41个点深度均为正。点拟合仅3参数、13次评估，不作为训练初始化。',
 '同帧58/60/62/65皮肤残差17.119/13.431/11.743/13.687px，原地面轨迹留出0.962/0.159/0.444/0.686px。两侧检查点均未参与各自三维点估计。地面控制仅一个候选，在图像边缘，估计深度约为皮肤点2倍，不能据此证明全部相机精度。',
 '已实际检查25FIT与3HELD的全279节点、皮肤全41帧、4个皮肤投影以及地面全9节点。没有按误差删点或将留出转入拟合。',
 '手工初始像素坐标误写到邻近皮肤，在第一次LK前经64px原生裁片和最近邻放大纠正。原计划错误坐标文件保留，未发生多次跟踪或阈值搜索。',
 '灰衣高斯未开始，WhiteT及winterhat旧18k结果仍为质量未通过的研究结果。未验证00、独立SIBR或Spark。只有报告第52页更新，其他51页保留。',
 '证据：'+str(R/'grayT-skin-texture-probe/geometry-result.json'),
 '证据：'+str(R/'grayT-material-identity-review/actual-review.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123面部约束核查.pptx','123皮肤材料点核验.pptx')
code=code.replace('preserve_deck_surface123_camera_support.py','preserve_deck_surface123_material.py')
(H/'update_deck_surface123_material.mjs').write_text(code,encoding='utf-8')
for src,dst in [('preserve_deck_surface123_camera_support.py','preserve_deck_surface123_material.py'),('commit_surface123_camera_support_deck.py','commit_surface123_material_deck.py')]:
    (H/dst).write_text((H/src).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared one native52ndslide edit using canonical camera-support receipt.')
