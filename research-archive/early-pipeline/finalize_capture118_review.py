from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil

TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
O=TOP/'shuolee/current/new-capture-118'
L=Path('D:/00/工作日志/2026-10-26/新视频118')
R=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-new-capture-118')
D=R/'data/alignment-diagnostics'
s=json.loads((D/'summary.json').read_text(encoding='utf-8'))
q=json.loads((O/'queue-state.json').read_text(encoding='utf-8'))
now=datetime.now(timezone.utc).isoformat()
review=dict(updated=now,state='preflight_reviewed_training_not_admitted',
    registered_frames=89,decoded_preflight_frames=89,raw_video_frames=884,
    fit_reprojection_mean_px=0.8706338446180925,
    measured_tracks=s['measured_tracks'],total_tracks=s['total_tracks'],regions=s['regions'],
    source_measurements_sha256=hashlib.sha256((D/'measurements.json').read_bytes()).hexdigest(),
    inspected_projection_frames=['frame_000001','frame_000023','frame_000045','frame_000067','frame_000089'],
    inspected_material_sheets=13,all_face_candidate_sheets_reviewed=8,
    source_material_findings=[
        dict(track_id='presfm_094e99a4d58a2b05978bf5df1b6b6fd7934a1216aa0bbc2749aaa9e71efd7841',
            finding='confirmed_physical_identity_jump',first_frames='000073-000080: small circular feature on the door',
            later_frames='000082-000085: ceiling downlight',maximum_held_residual_px=413.4978330033731,
            evidence='对应点核验/background_outlier_diagnosis_01_094e99a4.png'),
        dict(finding='Other four inspected background outliers lie on repeated railings or curtains; exact material identity remains ambiguous.'),
        dict(finding='Eight face candidates show no cross-object jump in inspected patches, but fine skin/beard identity and all relevant nose/mouth points are not established.'),
        dict(finding='Face candidates cover 32 of 89 frames, including separate early and late track groups; this does not establish full-orbit head alignment or staticity.'),
        dict(finding='Several full frames crop or occlude feet/lower legs. Back-of-head regions must not count as visible facial validation.')],
    interpretation='At least one large residual has a confirmed correspondence cause. It cannot be assigned to calibration alone, nor used to certify all camera parameters.',
    next_action='Audit existing appearance graph links and physical ambiguity without changing the frozen 256-track metrics or current cameras; measure full-orbit material/staticity support before admission.',
    residual_based_metric_filtering=False,cameras_changed=False,observations_removed=False,
    official_fisheye8_mapping_verified=False,physical_camera_calibration_verified=False,
    subject_staticity_verified=False,training_started=False,quality_accepted=False,
    old_point_cloud_replaced=False,company00_verified=False)
for folder in (O,L):
    (folder/'camera-material-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
q.update(state=review['state'],updated=now,review='camera-material-review.json',training_started=False,quality_accepted=False)
for folder in (O,L):(folder/'queue-state.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plan=TOP/'shuolee/long-term-plan.md'
text=plan.read_text(encoding='utf-8')
marker='2026-10-26 新视频118实际预检查：'
line=marker+'新室内视频89/89采样整图注册；固定留出面部中位0.768px、身体0.432px。实际检查发现一条背景轨迹从门上圆点跳到天花板灯，最大误差413.498px，不能解释成统一相机偏差。8条面部轨迹未见跨物体跳转，但细纹理身份和整圈静止性仍未证实，多帧腿脚被裁切/遮挡；不删除失败验证点降分，不启动高斯或替换历史点云。下一步只读审查已有匹配图的错误链接及整圈材料支持。\n'
if marker not in text:
    text=text.replace('\n','\n\n'+line,1)
    plan.write_text(text,encoding='utf-8')
print(json.dumps(dict(state=q['state'],training_started=False,quality_accepted=False)))
