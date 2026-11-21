"""Record the completed local visual review, without promoting quality."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib

OUT=Path('D:/00/记录/2026-11-04_SwinIR4倍_MSV2/demo')
def write(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

findings=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),quality_accepted=False,models_promoted=False,
    reviewed_source_views=59,reviewed_face_views=59,reviewed_orbit_views=12,
    review_files=[f'all_{kind}_{i:02d}.png' for kind in ['body','face'] for i in range(1,6)]+['all_orbit_000.png','all_orbit_180.png'],
    summary_paragraphs=[
        '本版已完成59帧SwinIR 4倍超分与全程4倍分辨率MSV2训练，18,000步时钟、17,999次实际更新、101,190个高斯。文件与相机兼容核验通过，重建质量未通过。',
        '全部59个源视角的全身与面部对照，以及12个环绕视角已审阅。SwinIR输入更锐利，但训练后面部仍模糊；第6—12帧表情变化和第49—59帧侧脸细节没有被共享静态模型准确保留。手持设备在不同帧的遮罩中取舍不一致，渲染出现缺失和碎片。',
        '以原始476×960图像为共同参考，人物L1从0.02958变为0.03433、SSIM从0.90813变为0.88243，脸部L1从0.05703变为0.06146；拟合SwinIR目标图的指标改善，不能据此说真实效果改善。',
        '实际导入通用点云与相机ZIP，确认101,190点和59台相机，并检查SH3高斯及点云模式。灰底显示头部、手臂和脚边的大量黑色不透明伪影；纯黑预览掩盖了这部分污染。',
        '环绕120—240度的背面存在大片缺口和拉伸；这些方向没有实拍真值，不能当作已完成背面重建。整体静态质量未通过，保留为透明背景修正前的必要对照。',
        '四倍版点云包已发给小润，QQ显示对方成功接收；已同时说明背景污染尚未解决。透明监督实验单独记录，不把本包标成修正版。'
    ])
write('review-findings.json',findings)
write('viewer-import-verification.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),
    url='http://127.0.0.1:8870/?sh=3',entry='打开文件',bundle='demo_SwinIR4倍_MSV2_18000步_点云与相机.zip',
    points=101190,cameras=59,gaussian_mode_visually_checked=True,point_mode_visually_checked=True,
    background='gray',sh_degree=3,standard_format_passed=True,quality_accepted=False,
    observed='Opaque dark spikes around head, hands and feet; same pollution visible in native gray-background rendering.',
    sidebar_note='The existing latest-result sidebar refers to an unrelated experiment; selected imported file and point count identify this review.'))
print('SWINIR4X_REVIEW_RECORDED',OUT)
