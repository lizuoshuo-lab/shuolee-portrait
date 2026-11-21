"""Prepare the existing slide52 edit from original native evidence."""
from pathlib import Path
from PIL import Image
import json, hashlib
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-head';prefix='surface123-observation'
sources=[REC/f'grayT-visibility-material-review/frame_{n:03d}_nose_native_mask_support.png' for n in [74,76]]
figure=H/(prefix+'-native-figure.png');im=Image.new('RGB',(640,160))
for i,src in enumerate(sources):
    with Image.open(src) as tile:
        assert tile.size==(320,160)
        im.paste(tile,(i*320,0))
im.save(figure)
(H/(prefix+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=640,height=160,
    sources=[dict(path=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()) for src in sources]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_head.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-controls-deck-receipt.json')","read(build+'/surface123-head-deck-receipt.json')")
code=code.replace("root+'/grayT-head-rigid-review/actual-review.json'","root+'/grayT-temple-validation-review/actual-review.json'")
start=code.index("replace('修正局部定位后，静态错位仍在'");end=code.index('const images=',start)
code=code[:start]+"""replace('局部姿态能减错，相机仍不能替换','面部关键点精度与可见性不足');
replace('灰衣：皮肤留出中位18.83→6.93像素，33/35仍超1像素。','网络中人脸约30像素高，鼻点约每1.41原图像素一档。');
replace('局部调整若套给场景相机，路面中位误差0.69→326.90像素。','第二皮肤标记9/41帧定位失败，停止新几何拟合。相机误差仍未排除。');
replace('青圈：皮肤观测；橙：静态；绿：头部逐帧。仍未达到亚像素验收。','74/76帧原图与鼻点。橙点在mask外，青点在mask内。');
"""+code[end:]
code=code.replace('原帧58、60、62、65的皮肤观测、静态与鼻部FIT代理逐帧刚体预测，原生裁片无文字',
    '原帧74、76的鼻部原生160px裁片，每组左原图右保存的检测点，图内无文字，无形变')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((REC/'observation-quality-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],
    '只读检查41个已存帧和全部369鼻部观测，未重跑检测或修改点权重。RTMW输入288×384，人物bbox加1.25padding并校正长宽比。模型SimCC argmax除2，原向和左右翻转坐标取平均。由真实bbox和解码解析得到的原生格点间隔中位1.412193726px，最大1.438488865px；369点距各自解析格点最大3.22e-13px。没有按观测拟合格点原点。',
    '单次解码的约2.82px间隔在翻转平均后变成约1.41px。浮点坐标不等于物理亚像素准确度。这能解释跳动的一部分，但不能证明静态皮肤18.83px误差全部来自量化。也不能将1.41px格点直接当作误差下界。',
    '脸部ROI在全身检测输入中位29.75px高。全部鼻点权重均为正，其中12点位于原SAM2 mask外0.786–3.660px，仍被4px边界余量保留。权重范围0.748–0.979。mask本身不是解剖真值，不能凭mask或残差直接删除远侧点。',
    '实际审阅全部41帧鼻部原图/点对照和41帧头部原生裁片。保守人工可见区域标签252可见鼻区、27远侧不确定、90边界或推断。这不是材料身份或解剖真值，不用于重加权。',
    '一次预登记第二材料M1测量：frame54颞部色素斑种子[792.5,965.5]，41原生帧，LK往返阈值0.5px，实际最大0.20004px。固定13×13负高斯+平面背景局部模型，40–48帧共9帧触及参数边界。实际复看原图后早段色素中心与相邻眉纹分离仍不清楚。不能用往返误差认证材料位置。',
    '严格执行预登记停止条件：没有把失败帧删掉，没有扩宽中心/宽度参数，没有缩短序列，没有运行新三维点或相机/姿态拟合。全部41测量保留。此前基于41头图的材料库存只表示可测候选，不能解读为全部41帧身份/中心已确认。',
    '此前头部代理逐帧调整在独立皮肤验证中仍33/35>=1px；不可用它替换整场景外参。当前独立面部与全身保护未通过，不启动灰衣高斯。历史白T/戴帽研究模型保持quality=false。',
    con['next'], '证据：'+str(REC/'observation-quality-conclusion.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123头部刚体诊断.pptx','123观测精度核查.pptx').replace('preserve_deck_surface123_head.py','preserve_deck_surface123_observation.py')
(H/'update_deck_surface123_observation.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_head.py','preserve_deck_surface123_observation.py'),('commit_surface123_head_deck.py','commit_surface123_observation_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared native evidence and slide52 edit; canonical still unchanged.')
