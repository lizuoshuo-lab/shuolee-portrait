"""Register the first field ROI inventory; do not run models here."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
H=Path(__file__).parent
REC=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
F=REC/'field-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (F/'coverage-plan.json').exists() and not (F/'coverage').exists()
result=read(REC/'camera-observation-entry/result-completed.json');review=read(F/'visual-review.json')
assert result['total_checks']==26 and result['quality_accepted'] is False
assert review['actual_full_frames_reviewed']==5
script=(H/'pattern_coverage123.py').read_text(encoding='utf-8')
script=script.replace("PRE=REC/'pattern-preflight'","PRE=REC/'field-preflight'")
script=script.replace('preflight/pattern','preflight/field').replace('==99','==177').replace('frames=99','frames=177').replace('total=99','total=177').replace('done=99','done=177').replace('/99 selection','/177 selection')
script=script.replace('progress: local matching branch rejected with evidence, all99patternframes decoded and reviewed in5samples','progress: frozen-observation caller repaired with26engineeringchecks; field177nativeframes decoded and5fullframes+10nativecrops reviewed')
path=H/'field_coverage123.py';assert not path.exists();ast.parse(script);path.write_text(script,encoding='utf-8',newline='\n')
plan=dict(registered=datetime.now(timezone.utc).isoformat(),subject='field',
    input_sha256=review['source_sha256'],manifest_sha256=sha(F/'decoded.json'),
    source_frame_count=882,selected_frame_count=177,selected_source_indices=list(range(0,882,5)),
    purpose='First local field person/region visibility inventory using frozen existing YOLO/RTMW. No semantic keypoint number is a material match.',
    reason='The previously unprocessed local field subject has front/side/back and feet in five original samples. Full registered-frame ROI coverage is needed before its first native texture inventory.',
    source_review=str(F/'visual-review.json'),source_review_sha256=sha(F/'visual-review.json'),
    method='Reuse the exact prior coverage decoder and detector settings; change only the subject/frame count. Native coordinates are inverse-mapped from detector ROI; no image warp is used for camera fitting.',
    script=str(path),script_sha256=sha(path),historical_pattern_source_sha256=sha(H/'pattern_coverage123.py'),
    budget=dict(frames=177,passes=1,max_wall_seconds=1200,GPU_jobs=1,new_downloads=0,feature_matching=0,cameras=0,SMPLX=0,training=0),
    review_slots=[0,44,88,132,176],
    stop='If current PID exists inspect it, never start twice. Stop on source/hash/raster/mapping failure or timeout. Ambiguous detections remain explicit unavailable rows. Review all support counts and the five fixed native contexts before registering a material pass. No fallback weights or threshold sweep.',
    unchanged='Existing models, fixed conditional K, old historical geometry and original videos. This inventory supplies regions only; no staticity, physical correspondence or subpixel claim.')
(F/'coverage-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(plan=str(F/'coverage-plan.json'),executed=False,training=False),ensure_ascii=False))
