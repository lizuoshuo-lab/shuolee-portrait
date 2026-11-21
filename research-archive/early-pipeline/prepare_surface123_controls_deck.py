"""Prepare a native edit of the existing52ndslide, retaining the first51."""
from pathlib import Path
import hashlib,json
from PIL import Image
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-material';prefix='surface123-controls'
paths=[R/f'grayT-skin-localization-review/same_frame_{f:03d}_projection.png' for f in [58,60,62,65]]
canvas=Image.new('RGB',(1024,256))
for i,path in enumerate(paths):
    with Image.open(path) as im:assert im.size==(256,256);canvas.paste(im,(256*i,0))
figure=H/(prefix+'-native-figure.png');canvas.save(figure)
(H/(prefix+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=1024,height=256,
    sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_material.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-camera-support-deck-receipt.json')","read(build+'/surface123-material-deck-receipt.json')")
code=code.replace("root+'/grayT-skin-texture-probe/post-geometry-review.json'","root+'/grayT-skin-localization-review/post-geometry-review.json'")
start=code.index("replace('SMPL-X错位与面部约束缺口'");end=code.index('const images=',start)
code=code[:start]+"""replace('皮肤材料点仍有跨视角错位','修正局部定位后，静态错位仍在');
replace('灰衣：35个留出视角中位18.9像素，最大32.3像素。','灰衣：35个留出视角中位18.83像素，最大33.10像素。');
replace('同帧地面观测0.16–0.96像素，相机残差与人物微动仍需分离。','定位最大修正2.91像素；近景路面18次检查仍有2次超过1像素。');
replace('青圈：皮肤点观测。橙十字：固定相机下同一静态三维点的投影。','青圈：亮度定位。橙十字：固定相机下静态点投影；相机与微动未分离。');
"""+code[end:]
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((R/'alignment-control-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],
    '本页为两项分开登记的控制结果。近景仅复用原HELD轨迹、原3D点和原K/E，无新提取或几何优化。按同帧覆盖、估计深度距离、ID排序，读取误差前审阅全110节点。六条中五条为路面候选，一条为树叶；树叶误差完整保留，但不能当静止负对照。',
    '路面在58/60/62/65共有18个检查，16个<1px，2个>=1px，中位0.686861、最大2.445777px。最接近旧皮肤点估计深度的路面比值1.138–1.177，三个检查0.706/1.850/2.446px。深度为估计，非物理测量；这些控制仍主要共面。',
    '观测定位单变量：13x13原生亮度斑点模型（负椭圆高斯+平面背景，8参数），中心以旧LK为初值且边界±3px，41帧一次运行，无调窗、重试或误差筛点。位置变化中位0.428064px、最大2.909808px。',
    '合成模型正确性：无噪声中心误差2.48e-16px，固定噪声样本误差0.006247px；这不验证真实材料亚像素精度。亮度Hessian条件95%半径中位0.247、最大0.352px，只在模型和噪声假设下有效。',
    '全部41帧原生64px及最近邻2倍定位图实际审阅，未见跨材料跳接。随后原6锚帧拟合一个静态3D点，35HELD均未用于点估计。留出中位18.832509、P9027.350449、最大33.097690px，35个均>=10px。旧LK对应中位18.925094px，局部观测修正没有消除大错位。',
    '原生同帧58/60/62/65亮度定位静态投影误差17.962/14.345/12.691/14.438px。照片裁片均256x256，未变形；图内无文字，青圈/橙十字仅为对照标记。',
    'SMPL-X帧映射与投影数值复核通过不等于拟合正确。灰衣面部检测残差中位43.99px，原粗身体门槛允许50–53px且未精细验脸。旧报告已明确quality=false；本轮不将它改称物理通过。',
    '无新相机、SMPL-X或高斯训练；当前WhiteT和winterhat仍是未通过质量验收的研究模型。未实测公司00、SIBRGUI或独立Spark页。仅更新同一52页PPT的第52页，前51页不变。',
    '下一步：在后续本地队列增加与粗初始化分开的对齐准入检查。几何原因仍缺近头部非共面固定点及多处独立刚性皮肤材料证据，不能声称全部由相机或头部运动引起。',
    '证据：'+str(R/'grayT-near-scene-support-review/result.json'),
    '证据：'+str(R/'grayT-skin-localization-review/geometry-result.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123皮肤材料点核验.pptx','123近景与定位控制.pptx')
code=code.replace('preserve_deck_surface123_material.py','preserve_deck_surface123_controls.py')
(H/'update_deck_surface123_controls.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_material.py','preserve_deck_surface123_controls.py'),('commit_surface123_material_deck.py','commit_surface123_controls_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared same52slide update; canonical not changed yet.')
