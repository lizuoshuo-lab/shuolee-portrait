"""Edit only slide52 of the existing52-page technical route deck."""
from pathlib import Path
import json,hashlib,shutil
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
old='surface123-controls';prefix='surface123-head'
src=R/'grayT-head-rigid-review/same_frame_comparison.png';figure=H/(prefix+'-native-figure.png');shutil.copy2(src,figure)
(H/(prefix+'-figure.json')).write_text(json.dumps(dict(path=str(figure),width=1024,height=256,
    sources=[dict(path=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest())]),ensure_ascii=False,indent=2),encoding='utf-8')
code=(H/'update_deck_surface123_controls.mjs').read_text(encoding='utf-8').replace(old,prefix)
code=code.replace("read(build+'/surface123-material-deck-receipt.json')","read(build+'/surface123-controls-deck-receipt.json')")
code=code.replace("root+'/grayT-skin-localization-review/post-geometry-review.json'","root+'/grayT-head-rigid-review/actual-review.json'")
start=code.index("replace('皮肤材料点仍有跨视角错位'");end=code.index('const images=',start)
code=code[:start]+"""replace('修正局部定位后，静态错位仍在','局部姿态能减错，相机仍不能替换');
replace('灰衣：35个留出视角中位18.83像素，最大33.10像素。','灰衣：皮肤留出中位18.83→6.93像素，33/35仍超1像素。');
replace('定位最大修正2.91像素；近景路面18次检查仍有2次超过1像素。','局部调整若套给场景相机，路面中位误差0.69→326.90像素。');
replace('青圈：亮度定位。橙十字：固定相机下静态点投影；相机与微动未分离。','青圈：皮肤观测；橙：静态；绿：头部逐帧。仍未达到亚像素验收。');
"""+code[end:]
code=code.replace('原帧58、60、62、65的皮肤材料点观测与冻结相机静态三维点投影，图内没有文字',
    '原帧58、60、62、65的皮肤观测、静态与鼻部FIT代理逐帧刚体预测，原生裁片无文字')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
con=json.loads((R/'head-rigid-conclusion.json').read_text(encoding='utf-8'))
notes=[con['summary'],
    '同一个预登记诊断：固定原相机K/E、SMPL-X头部9鼻部三维点、原权重和41帧。只改变所有帧共享6DoF与每帧6DoF。鼻部代理是FIT，独立皮肤定位从未进入姿态优化。此前35皮肤HELD已用于研究对照，不声称新未见测试集。',
    '先核对帧和IBUG27..35映射，独立投影最大差0.000327px；41帧9点都具数值Jacobian rank6，不能等同物理可辨识。合成SE3及参考帧控制通过后，一次CPU2.563秒完成1共享+41逐帧优化，无新相机/特征/模型训练。',
    '所有41帧原生鼻部裁片在拟合前实际审阅，侧面末段部分鼻梁检测点越过可见面部边界。没有按残差删点，也没有把这些语义代理称物理真值。',
    '共享刚体变换不会改善静态材料点多视角一致性；重估皮肤三维点后与旧静态投影最大差1.00e-6px。逐帧刚体后35HELD中位6.933834、P9010.810861、最大12.216955；只有2个<1px，33个>=1px。',
    '相对参考40的拟合头部旋转中位19.3037度，最大50.6867度。这是模型所需修正，不能报告为准确的真实人物微动。模板形状、遮挡代理与相机误差均可被局部自由度吸收。',
    '场景反事实只将E_i H_i inv(H_40)应用于原先保存的HELD背景三维点；不优化点或K。18次路面误差中位0.686861→326.902566，最大397.025960像素。只排斥这组头部修正作为整场景相机的解释，不能排除所有相机模型。',
    '工程修复：local123拟合前、表面初始化前、训练前和1000步复核均增加精细准入；27项合成边界合同和4调用入口检查通过，实际WSL三人物两阶段共6次拒绝缺失证据。没有用合成测试授权真实训练。',
    '当前灰衣仍未启动高斯，WhiteT/winterhat历史18k研究模型保留quality=false。未改原图/相机/拟合/PLY，未实测00、官方SIBR或独立SparkGUI。',
    '下一步只做原生可见性与多物理材料支持核查，不调整点权重、不复跑本次逐帧姿态。头部真实运动、精确标定及模板形状尚未唯一分离。',
    '结果：'+str(R/'grayT-head-rigid-review/result.json'),'工程证据：'+str(R/'alignment-admission-conclusion.json')]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123近景与定位控制.pptx','123头部刚体诊断.pptx').replace('preserve_deck_surface123_controls.py','preserve_deck_surface123_head.py')
(H/'update_deck_surface123_head.mjs').write_text(code,encoding='utf-8')
for a,b in [('preserve_deck_surface123_controls.py','preserve_deck_surface123_head.py'),('commit_surface123_controls_deck.py','commit_surface123_head_deck.py')]:
    (H/b).write_text((H/a).read_text(encoding='utf-8').replace(old,prefix),encoding='utf-8')
print('Prepared slide52 only. Canonical deck not changed.')
