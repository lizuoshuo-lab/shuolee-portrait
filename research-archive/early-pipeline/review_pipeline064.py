"""Record actual inspection of completed static RGB, coverage and point renders."""
from pathlib import Path
import json,datetime,hashlib,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
R=B/'current/person-static-pgsr-fresh-064'
files=['gaussian-run/review-current.jpg','gaussian-run/export-audit/same-camera-062-064.jpg',
       'gaussian-run/export-audit/constant-color-comparison.jpg','gaussian-run/export-audit/center-comparison.jpg']
d=dict(created=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),actually_inspected=True,
    evidence={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in files},
    summary_zh='同相机比较中，064的左右侧面袖口、衣摆边缘较062更连续，高斯中心图能看出更清楚的外套和裤腿结构。但鞋底部分方向的缺口更明显，仰视躯干仍有重影，未获得全面改善。',
    limitations_zh='整体质量尚未通过，064保留为表面约束对照，不替代062成为整体最优。隐藏表面没有真实三维测量；静态假设也无法解释实拍中的真实微动。颜色拟合良好或点中心看起来整齐，都不足以证明几何正确。',
    next_zh='继续静态主线。下一轮仍从头训练，固定相同分辨率与增密方式，分离最小尺度压缩和多视图表面约束的影响，优先查清鞋底与仰视缺口的来源；不恢复逐帧人体形变，也不把生成视角当作真值。',
    quality_accepted=False,geometry_accepted=False,promoted_overall_best=False,
    same_camera_comparison=True,independent_gt=False,view_directions=7,view_count_is_not_a_robustness_benchmark=True,
    export_roundoff='15 pixels among8 rendered images exceeded strict1-level thresholds. Exhaustive analytic compositing explained depth ties/alpha cutoff; local geometry unchanged. Strict pixel parity remains false in receipt.')
(R/'visual-review.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
g=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/repos/GaussianAvatars_SMPLX-shuolee')
out=R/'source-final/export';out.mkdir(exist_ok=True)
for name in ['audit_pgsr_static_delivery.py','compatible_gaussian_export.py','export_viewer_upright.py','diagnose_upright_order.py']:
    shutil.copy2(g/name,out/name)
print('Actual final renders reviewed; overall quality not accepted')
