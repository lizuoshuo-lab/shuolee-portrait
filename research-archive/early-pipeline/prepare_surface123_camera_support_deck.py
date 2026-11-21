"""Prepare one native edit of the same existing slide52; prior slides preserved."""
from pathlib import Path
import json,hashlib
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
prefix='surface123-camera-support'
figure=R/'grayT-track-attrition-review/FIT_whole_face_component_083ced88259e_locator.png'
spec=dict(path=str(figure),width=768,height=256,sources=[dict(path=str(figure),sha256=hashlib.sha256(figure.read_bytes()).hexdigest())])
(H/(prefix+'-figure.json')).write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
code=(H/'update_deck_surface123_alignment.mjs').read_text(encoding='utf-8')
code=code.replace('surface123-alignment',prefix)
code=code.replace("read(build+'/surface123-18000-deck-receipt.json')","read(build+'/surface123-alignment-deck-receipt.json')")
code=code.replace("root+'/grayT-smplx-audit-review/actual-review.json'","root+'/grayT-camera-support-review/actual-review.json'")
start=code.index("replace('自由衣物表面训练的18000步结果'")
end=code.index('const images=',start)
code=code[:start]+"""replace('SMPL-X错位与固定相机诊断','SMPL-X错位与面部约束缺口');
replace('灰衣人物95个视角的面部拟合中位残差约44像素。','灰衣95帧：相机优化中9个脸区观测，逐点审阅均为背景。');
replace('放开人体模板后仍约47像素，静态跨视角一致性尚未通过。','25条脸区候选轨迹未进入最终相机优化；人体拟合仍约44像素。');
replace('左为原图，中为参与拟合的检测点，右为SMPL-X投影。原视频第375帧。','鼻部候选轨迹的原生三视角。全图误差0.39像素不能证明面部准确。');
"""+code[end:]
code=code.replace('原生像素同裁剪的原图、检测鼻嘴点和SMPL-X投影，图内没有文字','同一鼻部候选轨迹的三个原生裁片和独立点位标记，图内没有文字')
start=code.index('p.resolve(slide.id).speakerNotes.text=');end=code.index('const candidatePath=',start)
notes=[
 '2026-10-28 灰衣相机支持审阅，未改变K、外参、点坐标、原始观测、拟合/留出划分或任何训练参数。灰衣高斯仍未开始。',
 json.loads((R/'camera-support-conclusion.json').read_text(encoding='utf-8'))['summary'],
 '53761个最终BA观测中52179个在人物mask外；脸ROI只有9个观测、7条轨迹、4/95帧。全部9个原生裁片和精确点位已查看：均为墙边植物或栏杆，不是脸部物理点。',
 '原生特征总数467584，脸ROI与mask交集556个。FIT分组含35条涉及该区域的组件，其中25条全部节点在该区域；最终BA的该区域观测数为0。25条只是候选，不能视为全部同名物理真值。',
 '三条未入选FIT全脸区候选按ID均匀选择后实际查看，点位位于鼻部、额侧/眼眉和眼周。图示为083ced88259e，完整原生图像和源帧在grayT-track-attrition-review/index.json。',
 '留出轨迹ee91c8b1f0c2从眼周变成背景栏杆，证实跨物体错配。嘴角轨迹344fc01b6ef5仍存在表情变化/语义身份歧义，其帧76检测残差27.884px不单独证明K或外参错误。',
 '所有高残差与负深度观察保留。背景外区域并非自动静态真值；某些抽样叶片可能运动。低全图FIT残差有原先几何筛选偏差，不能替代独立有效面部逐点<1px验收。',
 'SMPL-X灰衣活跃面部检测残差中位43.992px；冻结相机和检测观测，独立静态3D点诊断仍47.054px。该结果与本轮支持缺口尚不能唯一分离条件K、外参、检测语义和人物微动。',
 '保留同一份52页报告，其他51页及历史研究记录不改变。查看器当前WhiteT18k只是已发布研究结果，质量未通过。无新PLY、无公司00部署或SIBR GUI验收。'
]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123SMPL-X错位诊断.pptx','123面部约束核查.pptx')
(H/'update_deck_surface123_camera_support.mjs').write_text(code,encoding='utf-8')
for src,dst in [('preserve_deck_surface123_alignment.py','preserve_deck_surface123_camera_support.py'),('commit_surface123_alignment_deck.py','commit_surface123_camera_support_deck.py')]:
    text=(H/src).read_text(encoding='utf-8').replace('surface123-alignment',prefix)
    (H/dst).write_text(text,encoding='utf-8')
# The old module's helper basename is separate from its artifact prefix.
p=H/'update_deck_surface123_camera_support.mjs';text=p.read_text(encoding='utf-8').replace('preserve_deck_surface123_alignment.py','preserve_deck_surface123_camera_support.py');p.write_text(text,encoding='utf-8')
print('Prepared native Artifact Tool edit of slide52 only, using current canonical SHA.')
