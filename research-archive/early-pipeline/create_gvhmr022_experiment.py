from pathlib import Path
from datetime import datetime
import hashlib,json

lab=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
repo=lab/'repos/GVHMR-shuolee/shuolee'
base=lab/'outputs/mini-splatting2-mip-depth-001/shuolee'
phase=base/'current/smplx-gvhmr-surface-contact-fresh-022'
source=repo/'refine_fresh.py'
target=repo/'refine_surface_contact_fresh.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='8b2ac5e53bb45e2a55f7ce0343dc7c327f79648c72b93cecad9137727448b72f'
assert not phase.exists() and not target.exists()
audit=json.loads((base/'current/smplx-gvhmr-articulation-fresh-021/motion-review/motion-audit.json').read_text('utf-8'))
assert audit['stable_shoe_tracks']==66
assert audit['statistics']['refine']['stable_shoe_feature_transfer_error_px']['median']>30
s=source.read_text('utf-8-sig')
edits=[
    ("smplx-gvhmr-articulation-fresh-021","smplx-gvhmr-surface-contact-fresh-022"),
    ("        center=patch.mean(1)\n            stable += .5*robust((center-center.mean(0))/(st['s']*.025)).mean()", "        stable += .5*robust((patch-patch.mean(0,keepdim=True))/(st['s']*.025)).mean()"),
    ("        all59_real_cameras_fixed=True,official_coco17_regression=True,relative_rotations_composed_on_SO3=True,", "        sole_stability='Pointwise sole patch instead of patch centroid; same0.025m normalization and0.4 weight',\n        diagnostic_texture_tracks_in_training=False,\n        matched_reference='021 independently fitted raw-prediction control; not an initialization',\n        all59_real_cameras_fixed=True,official_coco17_regression=True,relative_rotations_composed_on_SO3=True,"),
]
# Match the indentation explicitly to avoid changing unrelated losses.
edits[1]=("            center=patch.mean(1)\n            stable += .5*robust((center-center.mean(0))/(st['s']*.025)).mean()",edits[1][1].replace('        stable','            stable'))
for before,after in edits:
    assert s.count(before)==1,before
    s=s.replace(before,after,1)
target.write_text(s,'utf-8')
phase.mkdir()
receipt=dict(created=datetime.now().astimezone().isoformat(timespec='seconds'),state='prepared_not_started',
    hypothesis='Centroid stability leaves per-foot rotations free. Pointwise stability also constrains orientation while retaining the same coefficient and length scale.',
    source_control=str(source),source_control_sha256=sha(source),candidate_source=str(target),candidate_sha256=sha(target),
    only_numerical_change='Replace sole centroid dispersion with corresponding sole-vertex dispersion.',
    new_initialization_from_raw_official_predictions=True,previous_personalized_checkpoint_used=False,
    source_frames=59,steps=1200,gaussian_updates=0,
    diagnostic_tracks='66 original stable shoe texture tracks stay out of this loss.',
    review_requirements=['Real image overlays','Same fixed-world cameras','Shoe texture transfer consistency','Knee/body/hand regression checks'],
    accepted=False)
(phase/'experiment-design.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),'utf-8')
status_path=base/'work-status.json';status=json.loads(status_path.read_text('utf-8-sig'))
status.update(updated=receipt['created'],current_phase='gvhmr_surface_contact_fresh_022',
    current_decision='021的66条稳定鞋面纹理诊断显示模型传递中位残差33.03px，静态重投影0.63px。022从头只替换脚底稳定项：中心稳定改为对应顶点稳定，权重和尺度保持一致；不把纹理轨迹加入拟合损失。')
status['latest_local_execution']=dict(kind='gvhmr_surface_contact_fresh_022',state='prepared_not_started',running=False,
    directory=str(phase),planned_steps=1200,new_body_fit_updates=0,gaussian_updates=0,old_person_fit_loaded=False,accepted=False)
status_path.write_text(json.dumps(status,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(receipt,ensure_ascii=False))
