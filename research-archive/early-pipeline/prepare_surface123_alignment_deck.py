from pathlib import Path
import json,hashlib
from PIL import Image
H=Path(__file__).parent
R=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
g=R/'grayT-smplx-audit-review';review=json.loads((g/'actual-review.json').read_text(encoding='utf-8'))
assert review['actual_visual_review']
figure=g/'frame_000076_native_face_raw_detector_model.png';im=Image.open(figure)
prefix='surface123-alignment'
(H/f'{prefix}-figure.json').write_text(json.dumps(dict(path=str(figure),width=im.width,height=im.height,
    sources=[dict(path=str(figure),sha256=hashlib.sha256(figure.read_bytes()).hexdigest())]),ensure_ascii=False),encoding='utf-8')
preserve=(H/'preserve_deck_surface123_18000.py').read_text(encoding='utf-8').replace('surface123-18000',prefix)
(H/'preserve_deck_surface123_alignment.py').write_text(preserve,encoding='utf-8')
code=(H/'update_deck_surface123_18000.mjs').read_text(encoding='utf-8').replace('surface123-18000',prefix).replace('preserve_deck_surface123_18000','preserve_deck_surface123_alignment')
code=code.replace("build+'/surface123-6000-deck-receipt.json'","build+'/surface123-18000-deck-receipt.json'")
code=code.replace('winterhat-train-018000-review','grayT-smplx-audit-review').replace('||review.steps!==18000','')
start=code.index("replace('自由衣物表面训练的6000步检查'");end=code.index('const images=',start)
code=code[:start]+"""replace('自由衣物表面训练的18000步结果','SMPL-X错位与固定相机诊断');
replace('29.4万高斯覆盖衣物，面部与鞋边重影仍未通过。','灰衣人物95个视角的面部拟合中位残差约44像素。');
replace('112个相机与点云配套上架，高斯和点云模式均已实测。','放开人体模板后仍约47像素，静态跨视角一致性尚未通过。');
replace('左为全身，右为两种面部视角。每组依次为原图、18000步渲染和实际点中心。','左为原图，中为参与拟合的检测点，右为SMPL-X投影。原视频第375帧。');
"""+code[end:]
code=code.replace('18000步同相机原图、自由表面高斯与实际中心分布，左全身、右上正面、右下侧脸。原图保持比例，图内无文字','原生像素同裁剪的原图、检测鼻嘴点和SMPL-X投影，图内没有文字')
start=code.index('p.resolve(slide.id).speakerNotes.text=[');end=code.index('const candidatePath=',start)
notes=[
    '2026-10-28 SMPL-X错位诊断。灰衣人物尚未开始高斯训练，原队列在已有表面审阅节点停止。强鲁棒性目标继续，保留此前两个人物和沙发119。',
    review['summary'],
    '三人物活跃面部观测的原生像素中位残差：戴帽蓝衣7.299，白T恤32.037，灰T恤43.992。固定相机、观测和权重，仅解除模板耦合后分别6.672、24.601、47.054。所有有效观察逐点低于1像素的目标均未通过。',
    '解除模板耦合只优化每个面部编号共享的静态三维点，保持原有加权鲁棒目标。灰衣目标4.775085降至4.413785，P90从92.031降至80.785，而中位数增加。优化目标下降不能冒充精准对应通过。有限局部优化不构成全局最优证明。',
    '数值投影差小于0.0004像素仅验证软件路径一致。条件K的物理正确性、相机外参、检测器解剖语义以及转头和表情变化仍未唯一分离。',
    '本轮缺少脸脚解剖可见性过滤，粗门槛只检查身体点占框高比例。当前拟合网格的自遮挡分析不是人体真值，不能按该结果或残差直接删观测。下一步先核实同一物理背景点，再做因果区分。',
    '白T恤独立从头训练已完成18000步，309523个自由高斯和93个相机。六幅同相机及36幅连续未知视角均审阅，面部复影、衣边拖尾和未拍背面破碎未解决。完成计算与点分布改变不代表人体或相机质量通过。',
    '证据目录：local-clothed-surface-123/smplx-mapping-audit、fixed-camera-face-relaxation、grayT-smplx-audit-review。所有诊断保留原观察和权重，未覆盖相机，未开始新的高斯训练。'
]
code=code[:start]+"p.resolve(slide.id).speakerNotes.text="+json.dumps(notes,ensure_ascii=False)+".concat(['图像证据：'+figureSpec.sources.map(x=>x.path+' sha256='+x.sha256).join('\\n'),'此前研究记录：',previous]).join('\\n\\n');\n"+code[end:]
code=code.replace('123自由表面18000步','123SMPL-X错位诊断')
(H/'update_deck_surface123_alignment.mjs').write_text(code,encoding='utf-8')
commit=(H/'commit_surface123_final_deck.py').read_text(encoding='utf-8').replace('surface123-18000',prefix)
(H/'commit_surface123_alignment_deck.py').write_text(commit,encoding='utf-8')
print('Prepared existing52slide final-page update only.')
