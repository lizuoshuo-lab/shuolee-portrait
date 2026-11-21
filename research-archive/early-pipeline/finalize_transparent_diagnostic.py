"""Record the visible pollution reduction separately from the strict opacity gate."""
from pathlib import Path
from datetime import datetime,timezone
import json

OUT=Path('D:/00/记录/2026-11-04_透明背景_MSV2/demo')
data=dict(reviewed_at=datetime.now(timezone.utc).isoformat(),quality_accepted=False,source_views_reviewed=59,
    same_step_body_views_reviewed=59,face_views_reviewed=59,background_views_reviewed=59,orbit_views_reviewed=12,
    summary='全59帧与12个环绕视角已审阅。黑色污染高斯明显减少，用户也认可这一视觉改善；当前3,000步候选予以保留。手部、设备及袖口仍有漂浮碎片和半透明外溢，侧脸仍模糊，背面裤腿存在孔洞。1%轮廓外不透明度门槛尚未达到，整体重建质量尚未验收。',
    black_pollution_gaussians_visibly_reduced=True,
    user_visual_feedback=dict(aspect='黑色污染高斯减少',positive=True,
        quotes=['这个效果这不是挺好的，没有了黑色的噪点高斯','污染高斯'],
        scope='用户认可可见的黑色污染高斯改善；不据此改写历史数值门槛或整体质量验收'),
    gate_interpretation='1%是本次试验预设的所有轮廓外不透明度门槛，包含浅色半透明残影，并非黑色污染高斯数量指标。',
    region_findings=[
        dict(region='head_and_hat',observation='Hat and face coverage remain; facial detail and expression still mismatch in side views.'),
        dict(region='hands_device_sleeves',observation='Dominant remaining wisps and overlapping fragments across most views; device silhouette is inconsistently included by the source masks.'),
        dict(region='torso_clothes',observation='Major dark halos decrease at equal step count; clothing remains visible, with residual boundary fuzz.'),
        dict(region='legs_feet',observation='Source-view coverage retained and dark outer blobs reduced; novel back-view pants remain perforated.'),
    ],
    demonstrated=['Outside RGB cannot affect foreground photo loss or gradients','Background depth-resampling probability is zero in the probe',
                  'True straight RGBA PNGs from rendered alpha; no output-mask clipping','Same 3000-step mean opacity leakage reduced approximately30percent'],
    unresolved=['One-percent outside-opacity threshold failed','Source mask semantic inconsistency around device','Static fit cannot reproduce observed expression changes exactly','Uncaptured back views lack ground truth'],
    training_running=False,additional_sweep_started=False,production_deployed=False,
    important_limit='Do not compare 3000-step image sharpness or point count directly with the sent 18000-step model as a controlled improvement.')
(OUT/'visual-review.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status=Path('C:/Users/Administrator/Documents/New project/shuolee-coordination/company00-status.json')
s=json.loads(status.read_text(encoding='utf-8-sig'))
s['msv2_transparent_demo'].update(state='stopped_at_3000_quality_gate_failed_reviewed',own_gpu_jobs_running=False,pending_gpu_phases=[],
    updated=datetime.now(timezone.utc).isoformat(),quality_accepted=False,training_completed=False,actual_iteration_label=3000,optimizer_updates=2999,
    report=str(OUT/'查看对照.html'),black_pollution_gaussians_visibly_reduced=True,user_positive_visual_feedback=True,
    remaining='Retained candidate with visibly reduced black pollution; remaining hand wisps and incomplete unseen geometry')
status.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
verification=dict(verified_at=datetime.now(timezone.utc).isoformat(),passed=True,
    viewer='http://127.0.0.1:8870/?sh=3',entry='打开文件',
    bundle='demo_SwinIR4倍_透明试验_3000步_未通过_点云与相机.zip',
    points=1397706,sh_degree=0,paired_cameras=59,camera='frame_000001',
    gaussian_mode_visually_checked=True,point_mode_visually_checked=True,background='gray',
    gaussian_scale=1.0,point_size_px=2,
    scope='Local viewer compatibility; separate from reconstruction quality and numeric opacity gate',
    observation='Body, hat and feet visible in Gaussian mode; dark outer blobs reduced; floating fragments remain near hands. Point mode renders the same imported model.')
(OUT/'viewer-import-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('TRANSPARENT_REVIEW_RECORDED')
