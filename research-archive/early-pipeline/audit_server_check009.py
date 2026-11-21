from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
import numpy as np
from PIL import Image

P=Path('D:/00/工作/人体重建')
WSL=Path('//wsl.localhost/Ubuntu-20.04')
R=WSL/'opt/gs-slam-lab/shuolee-portrait-server-check-009'
LOG=Path('D:/00/工作日志/2026-10-26/新视频工程接口验证009')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
status=read(R/'status.json')
assert status['state']=='diagnostic_stage_complete'
assert status['completed']==['extract','sfm','observations','prior','fit','fit-gate']
assert not (R/'gaussian-run').exists()
plan=read(P/'provenance/server-interface-check-009-plan.json')
for name,digest in plan['source_hashes'].items():assert sha(P/name)==digest, f'Running source changed: {name}'
run=read(R/'run.json')
assert run['video_sha256']==plan['raw_video_sha256'] and run['from_scratch'] and not run['old_subject_loaded']
rows=read(R/'data/inputs.json')['records']
maps=read(R/'inference/crop-mapping.json')
assert len(rows)==len(maps['records'])==23
assert [r['frame'] for r in rows]==[r['frame'] for r in maps['records']]
assert not maps['training_image_warped'] and not maps['physical_camera_accuracy_proven']
checks=[]
for row,m in zip(rows,maps['records']):
    image=WSL/row['image'].lstrip('/')
    assert sha(image)==row['source_sha256']==m['source_sha256']
    assert list(Image.open(image).size)==row['wh']==m['source_wh']
    assert m['crop_wh']==[256,256]
    assert np.allclose(m['bbox_xyxy'],row['initial_box_xyxy'],atol=0,rtol=0)
    forward=np.array(m['source_to_crop']);inverse=np.array(m['crop_to_source'])
    assert np.isfinite(forward).all() and np.isfinite(inverse).all()
    identity_error=float(np.max(np.abs(forward@inverse-np.eye(3))))
    cx,cy,size=m['center_size']
    source=np.array([[cx-size/2,cy-size/2,1],[cx+size/2,cy+size/2,1],[cx,cy,1]])
    endpoints=(source@forward.T)[:,:2]
    endpoint_error=float(np.max(np.abs(endpoints-np.array([[0,0],[255,255],[127.5,127.5]]))))
    assert identity_error<1e-8 and endpoint_error<1e-3
    assert not row['camera_registration']['interpolation_used']
    checks.append(dict(frame=row['frame'],source_frame_index=row['source_frame_index'],source_sha256=row['source_sha256'],identity_error_max=identity_error,endpoint_error_crop_pixels_max=endpoint_error))
result=dict(updated=datetime.now(timezone.utc).isoformat(),state='post_crop_fix_fresh_pipeline_numeric_check_passed',
    frames=len(rows),completed=status['completed'],source_unchanged=True,
    original_video_sha256_verified=True,input_images_sha256_verified=True,
    all_crop_maps_match_current_images_and_boxes=True,
    max_crop_endpoint_error_pixels=max(r['endpoint_error_crop_pixels_max'] for r in checks),
    max_inverse_composition_error=max(r['identity_error_max'] for r in checks),
    records=checks,coarse_fit=read(R/'fit/geometry-gate.json'),
    new_person_test=False,gaussian_training_started=False,new_ply_created=False,
    physical_calibration_accepted=False,quality_accepted=False,ghosting_resolved=False,
    visual_review_complete=False,
    scope='Fresh pipeline integration only. New SfM may differ, so this is not a single-variable image-quality comparison with008.')
save(R/'crop-integration-audit.json',result)
LOG.mkdir(parents=True,exist_ok=True)
for image in sorted((R/'reviews/fit').glob('*.png')):shutil.copy2(image,LOG/image.name)
for src,name in [(R/'crop-integration-audit.json','裁剪与全新流程核验.json'),(R/'inference/crop-mapping.json','逐帧裁剪正逆映射.json'),(R/'status.json','执行状态.json'),(R/'fit/geometry-gate.json','粗拟合检查.json'),(R/'inference/receipt.json','时序先验说明.json')]:shutil.copy2(src,LOG/name)
save(LOG/'帧映射.json',[{k:r[k] for k in ['frame','source_frame_index','time_seconds','wh','K','w2c']} for r in rows])
print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False))
