"""Persist completed input review and register one existing-model ROI pass."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'indoor-preflight'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
stamp=datetime.now(timezone.utc).isoformat()
assert not (PRE/'visual-review.json').exists() and not (PRE/'coverage-plan.json').exists()
r=read(PRE/'result.json');d=read(PRE/'decoder-comparison/result.json')
assert r['selected_frames']==177 and len(d['pairs'])==4 and d['training_started'] is False
notes=[
 '源帧0000：正面，眼鼻嘴可见；下腿与脚被画面下边截断。置物架及墙面温控器只作固定结构候选。',
 '源帧0220：右向侧面，远侧眼嘴存在自遮挡；桌面与图像下边挡住部分腿脚。',
 '源帧0440：后脑勺，眼鼻嘴不可见。楼梯与柜体可调查；发光环、重复栏杆与植物不自动算物理对应。脚被家具和下边界遮挡。',
 '源帧0660：左向侧面，近侧脸可见；门框、楼梯接头与开关可作候选。地上猫有运动可能，不能充当静止背景；腿脚部分遮挡或出画。',
 '源帧0880：回到正面，脸部可见且尺度较大；下腿脚出画。布局接近不等于静态人体或正确相机。']
review=dict(updated=stamp,actual_full_frames_reviewed=5,source_sha256=r['source_sha256'],full_native_resolution=True,
    rows=[dict(**im,note=note) for im,note in zip(r['images'],notes)],
    physical_landmark_labels_provided=False,face_visibility_qualitative_only=True,
    full177_visibility_not_yet_reviewed=True,physical_staticity_certified=False,training_started=False,quality_accepted=False)
save(PRE/'visual-review.json',review)
interpretation=dict(updated=stamp,compared_pairs=4,fixed_tiles=36,
    maximum_phase_correlation_shift_px=max(x['all9_max_shift_norm_px'] for x in d['pairs']),
    conclusion='Saved-pair comparison does not support a gross whole-image translation. Interior RGB differences and stronger bottom-border differences are preserved. HEVC10bit BT709 conversion/padding is a candidate explanation, not a proven decoder cause.',
    limitations='Phase correlation is not a physical camera or 2D-landmark accuracy measurement. No phase-correlation displacement is applied to an image or K. Original121 coordinates are not imported.',
    action='Use one current FFmpeg RGB raster consistently for images, freshly computed observations and later fitting. Preserve all original pixels and out-of-image/occlusion flags.',
    geometric_warp_applied=False,physical_camera_accuracy_passed=False)
save(PRE/'decoder-comparison/interpretation.json',interpretation)
r.update(state='native177_reviewed_single_decoder_domain',actual_full_frames_reviewed=5,
    visual_review=str(PRE/'visual-review.json'),decoder_comparison=str(PRE/'decoder-comparison/result.json'),updated=stamp)
save(PRE/'result.json',r)
src=H/'field_coverage123.py';script=src.read_text(encoding='utf-8')
assert "PRE=REC/'field-preflight'" in script and 'preflight/field' in script
script=script.replace("PRE=REC/'field-preflight'","PRE=REC/'indoor-preflight'").replace('preflight/field','preflight/indoor')
script=script.replace('progress: frozen-observation caller repaired with26engineeringchecks; field177nativeframes decoded and5fullframes+10nativecrops reviewed',
    'progress: indoor177nativeframes prepared; five native originals reviewed; four historical decoder pairs checked without changing pixels')
path=H/'indoor_coverage123.py';assert not path.exists();ast.parse(script);path.write_text(script,encoding='utf-8',newline='\n')
plan=dict(registered=stamp,subject='indoor',input_sha256=r['source_sha256'],manifest_sha256=sha(PRE/'decoded.json'),
    source_frame_count=884,selected_frame_count=177,selected_source_indices=list(range(0,884,5)),
    purpose='First123 indoor177 person/region inventory on current native RGB raster. Supply 2D observation locations and unknown visibility for later fixed-topology SMPL-X binding; region detection does not certify material correspondence.',
    new_information='173 predeclared source indices were absent from121 diagnostic24. No old camera, individual geometry, optimized points or historical RGB observations are initialization.',
    source_review=str(PRE/'visual-review.json'),source_review_sha256=sha(PRE/'visual-review.json'),
    decoder_review=str(PRE/'decoder-comparison/interpretation.json'),
    method='Reuse existing frozen YOLOv8n/RTMW-L settings and inverse ROI mapping. Change only subject paths. Every selected frame and ambiguous detection retained.',
    script=str(path),script_sha256=sha(path),reused_source_sha256=sha(src),
    budget=dict(frames=177,passes=1,max_wall_seconds=1200,GPU_jobs=1,new_downloads=0,feature_matching=0,cameras=0,SMPLX=0,training=0),
    review_slots=[0,44,88,132,176],
    stop='No duplicate run. Source/weights/raster/mapping failure, ambiguous person selection or timeout stays explicit. Review fixed five native crops and out-of-frame counts. Never assign anatomical visibility from detector confidence. No threshold sweep.',
    unchanged='Native pixels, conditional K, frozen anatomical topology mapping and all older models. This pass does not start fit or training.')
save(PRE/'coverage-plan.json',plan)
print(json.dumps(dict(input_review_saved=True,registered_frames=177,new_run=str(path),training=False),ensure_ascii=False))
