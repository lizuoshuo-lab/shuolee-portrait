"""Persist the actual final six-camera, 36-angle and FOV diagnostic review."""
from pathlib import Path
import json, hashlib, datetime
TOP=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=TOP/'shuolee/current/clothed-surface-119'
ROOT=Path('//wsl.localhost/Ubuntu-20.04/opt/gs-slam-lab/shuolee-clothed-surface-119/sofa')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review=REC/'sofa-train-018000-review';orbit=REC/'sofa-final-review'
validation=read(ROOT/'export/validation.json')
assert validation['format_passed'] and sha(ROOT/'export/point_cloud.ply')==validation['sha256']
assert not (REC/'sofa-quality-review.json').exists()
frames=[x['frame'] for x in read(review/'index.json')['images'] if x['region']=='body']
d=dict(updated=datetime.datetime.now().astimezone().isoformat(),subject='sofa',steps=18000,
    source_sha256=validation['sha256'],raw_world_sha256=sha(ROOT/'gaussian-run/step-018000/world.ply'),
    actual_same_camera_review=True,actual_center_distribution_review=True,reviewed_frames=frames,
    continuous_novel_views_reviewed=36,novel_review_method='36 ordered 10-degree images actually inspected in three contact sheets; no novel-view GT',
    observed_clothing_center_support_improved=True,quality_accepted=False,
    summary='沙发人物1.8万步完成，中心已支持衬衫、裤子与头发；同18k采集视角的衣物/头发边缘更稳，鞋尖拖影、背部破碎和未观测侧的质量仍不通过。',
    findings=[
      'Six body comparisons at identical18k budgets: shirt/pants/hair exterior has actual center support. Captured-side head silhouette is cleaner; sampled views are visually close to raw images.',
      'Six face crops: nose, mouth and glasses appear broadly aligned in captured views. Beard detail remains smoothed and eye/lens detail is not exact; no claim all-view ghosting solved.',
      'All36novel angles: frontal and captured-side appearance is coherent; unseen rear head/back/seat show severe streaking, tearing and approximate envelope geometry. Shoes have translucent trails and missing toe support.',
      'Format inverse/native-render checks passed, but nearest-axis export retains42.41degree body-up tilt. This is not measured gravity or an extra pose deformation.',
      'Combined initialization/representation change, not single-factor attribution. Camera applicability and subject motion still unresolved.'
    ],
    evidence=[dict(path=str(p),sha256=sha(p)) for p in [review/'body-contact.png',review/'face-contact.png',*[orbit/f'orbit-{i:02d}.png' for i in range(3)]]],
    format_passed=True,physical_calibration_accepted=False,pc00_opened_this_round=False)
save(REC/'sofa-quality-review.json',d);save(review/'actual-review.json',d);save(orbit/'actual-review.json',d)
fov=REC/'sofa-fov-diagnostic';v=read(fov/'result.json')
v.update(actual_visual_review_pending=False,actual_visual_review=True,
    reviewed_images=[dict(path=str(p),sha256=sha(p)) for p in sorted(fov.glob('*.png'))],
    actual_conclusion='把图像外体素从负票改为未观测后，60748变61794，恢复1046体素；六视角对照主要恢复部分鞋尖/足部。证明当前初始化有出画面负票偏差，不能证明相机正确或成品脚部已修复。',
    corrective_training_performed=False,scope='CPU-only visibility-accounting counterfactual; no changed camera, mask, learned result or extra training')
save(fov/'result.json',v)
print(d['summary'])
