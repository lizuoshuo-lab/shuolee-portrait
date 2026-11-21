"""Record the actual six-camera and 36-angle review performed in this turn."""
from pathlib import Path
import json, hashlib, datetime
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/clothed-surface-119'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-clothed-surface-119/portrait')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=REC/'portrait-train-018000-review'
orbit=REC/'portrait-final-review'
validation=read(ROOT/'export/validation.json')
assert validation['format_passed'] and sha(ROOT/'export/point_cloud.ply')==validation['sha256']
d=dict(updated=datetime.datetime.now().astimezone().isoformat(),subject='portrait',steps=18000,
    source_sha256=validation['sha256'],raw_world_sha256=sha(ROOT/'gaussian-run/step-018000/world.ply'),
    actual_same_camera_review=True,actual_center_distribution_review=True,
    reviewed_frames=['frame_000001','frame_000023','frame_000045','frame_000067','frame_000089','frame_000112'],
    continuous_novel_views_reviewed=36,novel_review_method='36 ordered 10-degree renders inspected in three consecutive contact sheets; not a video or ground truth',
    observed_clothing_center_support_improved=True,quality_accepted=False,
    summary='119自由点已覆盖裙摆、外套、挎包和帽子；1.8万步完成。正面与采集视角衣物更清楚，面部重影及侧后方破碎仍未通过质量验收。',
    findings=[
      'All six raw/render/one-pixel-center crops were inspected without geometric warping. Skirt and bag are supported by centers rather than only oversized splats.',
      'Six face crops: front face is clearer; right oblique views still show double eyebrows/eyes, nose and mouth edges. The silhouette representation does not solve camera/motion ambiguity.',
      '36 novel angles: front and captured-side clothes remain recognizable; unseen rear hat, shoulder, back, bag and skirt show severe tearing/streaks. Foot/hem intersections and thin edge streaks remain.',
      'Baseline portrait had only 1000 steps. Final 18000-vs-1000 quality comparison is not equal-budget causality; the independently reviewed 1000-vs-1000 pair is the matched-step evidence.',
      'Distance to naked mesh vertices and projected center coverage show changed support, not true skin/cloth depth. Visual-hull concavities and unobserved backs are unverified.'
    ],
    evidence=[dict(path=str(p),sha256=sha(p)) for p in [review/'body-contact.png',review/'face-contact.png',*[orbit/f'orbit-{i:02d}.png' for i in range(3)]]],
    format_passed=True,physical_calibration_accepted=False,pc00_opened_this_round=False)
save(REC/'portrait-quality-review.json',d)
save(review/'actual-review.json',d)
save(orbit/'actual-review.json',d)
print(d['summary'])
