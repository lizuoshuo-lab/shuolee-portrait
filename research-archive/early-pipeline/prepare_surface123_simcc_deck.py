"""Prepare a focused edit of slide52, using an existing unwarped evidence image."""
from pathlib import Path
import json,hashlib,shutil
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-observation';prefix='surface123-simcc'
source=REC/'grayT-simcc-decoder-review/frame_080_raw_argmax_continuous_native.png'
figure=H/(prefix+'-native-figure.png');shutil.copy2(source,figure)
(H/(prefix+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=480,height=160,
    sources=[dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest())]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_observation.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-head-deck-receipt.json')","read(build+'/surface123-observation-deck-receipt.json')")
code=code.replace('26.1020.11348','26.1026.10807')
code=code.replace("root+'/grayT-temple-validation-review/actual-review.json'","root+'/grayT-simcc-decoder-review/actual-review.json'")
start=code.index("replace('局部姿态能减错");end=code.index('const images=',start)
code=code[:start]+"""replace('面部关键点精度与可见性不足','连续解码未解决SMPL-X错位');
replace('网络中人脸约30像素高，鼻点约每1.41原图像素一档。','6帧鼻点位置变化中位0.84像素，最大1.82像素。');
replace('第二皮肤标记9/41帧定位失败，停止新几何拟合。相机误差仍未排除。','逆仿射误差小于0.0001像素，遮挡与材料对应仍需验证。');
replace('74/76帧原图与鼻点。橙点在mask外，青点在mask内。','第80帧：原图、原解码青点、连续解码橙点。原拟合保持。');
"""+code[end:]
code=code.replace('原帧74、76的鼻部原生160px裁片，每组左原图右保存的检测点，图内无文字，无形变',
    '第80帧三个原生160像素鼻部裁片，左原图、中原argmax青点、右连续解码橙点。没有图内文字，没有形变。')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((REC/'simcc-decoder-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],
    '预登记6帧40、48、56、64、72、80，每帧原向与水平翻转各推理一次，共12次CPU推理，保存全部原始133点SimCC响应。没有改变RGB归一化、bbox、模型、相机、得分合并或原观测。仅对已有响应作连续峰位解码。',
    '旧argmax结果与已存观测逐点完全一致，置信度差0。独立cv2逆仿射回原图最大差9.52747e-5px。这个数是实现一致性，不是真实相机或材料定位精度。',
    '54个鼻点坐标变化中位0.842562px、P90为1.239838px、最大1.815771px。这是两种解码的差值，不是定位误差降低。全部鼻点的数值稳定检查通过，仍不能据此确定遮挡或物理身份。',
    '模型导出包pipeline.json解码参数sigma6/6.93、split2、use_dark=false；可选细化按官方MMPose的SimCC DARK公式，使用固定37/43高斯核。没有把官方默认关闭解释成已证实的配置缺陷。',
    '独立已知连续高斯峰控制最大误差2.24e-5bin，零、平坦、边界、相隔多峰和非有限响应均按预登记规则标无效。只是解码计算验证，不代表自然图像亚像素准确度。',
    con['metadata_correction'],con['serialization_issue'],
    '实际审阅全部6帧的原图/旧解码/新解码原生裁片。主鼻形布局几乎不变，后段侧脸轮廓和远侧推断仍存在。没有独立的物理精度改善证据，因此保持原拟合不变，不启动新高斯。',
    con['next'],
    '公式参考：https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/mmpose/codecs/utils/refinement.py',
    '公式参考：https://raw.githubusercontent.com/open-mmlab/mmpose/v1.3.2/mmpose/codecs/utils/post_processing.py',
    '证据：'+str(REC/'simcc-decoder-conclusion.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123观测精度核查.pptx','123连续解码核查.pptx').replace('preserve_deck_surface123_observation.py','preserve_deck_surface123_simcc.py')
(H/'update_deck_surface123_simcc.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_observation.py','preserve_deck_surface123_simcc.py'),('commit_surface123_observation_deck.py','commit_surface123_simcc_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared slide52 code and existing native figure; canonical not changed.')
