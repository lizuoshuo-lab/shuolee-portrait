"""Record the completed three-resolution review without changing any model."""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import json

ROOT=Path('D:/00/记录/2026-11-04_MSV2_原始_2倍_4倍对比')
summary=[
    '原始、Lanczos插值2倍、SwinIR超分4倍均完成同一透明MSV2的18,000步时钟，各自从相同未训练初始点开始。59帧、相机外参、RGB/Alpha损失和MSV2时序一致；两倍与四倍的内参按倍率缩放。',
    '统一原图相机下，原始尺寸版最贴近原图：人物SSIM为0.9412，插值2倍0.9356，SwinIR4倍0.9060；人物L1分别为0.02327、0.02433、0.02962。四倍显示较平滑，未证明得到更多可靠细节。',
    '全部59帧中，三组都出现面部外侧细丝、帽檐外点和手持物附近重影；侧脸后半段还有五官模糊与错位。三组背面环绕视角都有缺口和斑块，现有输入没有背面真值。单纯放大不能解决这些共同问题。',
    '因此按用户要求，保留MSV2训练模型，只验证mask惩罚；不裁剪或修改已有点云。相应mask对照独立记录。',
]
review=dict(created=datetime.now(timezone.utc).isoformat(),training_completed=True,
    reviewed_source_views=59,reviewed_face_views=59,reviewed_orbit_views=12,
    reviewed_files=[f'all_{k}_{i:02d}.png' for k in ['body','face'] for i in range(1,6)]+['all_orbit_000.png','all_orbit_180.png'],
    known_limitations=['Face-adjacent opacity noise in all three methods','Pose/expression mismatch in static face reconstruction','Hand-held-object and hand motion','Unobserved back geometry'],
    all_three_have_face_adjacent_noise=True,quality_accepted=False,models_modified=False,
    summary_paragraphs=summary)
(ROOT/'review-findings.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
html_path=ROOT/'查看对照.html';html=html_path.read_text(encoding='utf-8')
start=html.index('<section><h2>对比结论</h2>');end=html.index('</section>',start)+len('</section>')
html=html[:start]+'<section><h2>对比结论</h2>'+''.join('<p>'+escape(p)+'</p>' for p in summary)+'</section>'+html[end:]
html_path.write_text(html,encoding='utf-8')
print('THREE_WAY_REVIEW_RECORDED')
