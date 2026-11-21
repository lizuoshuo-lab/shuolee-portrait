from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,ast
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
REC=T/'shuolee/current/local-clothed-surface-123';PRE=REC/'indoor-preflight';C=PRE/'coverage'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (C/'visual-review.json').exists()
stamp=datetime.now(timezone.utc).isoformat();inv=read(C/'review-inventory.json');r=read(C/'result.json')
notes=[
 dict(source_frame=0,face='正面八个面部部位可见；预测点落在相应区域附近，精确像素定位尚未测量。',feet='脚踝和双脚被沙发靠背遮挡，预测点落在裤腿和沙发表面，不能标成可见脚点。'),
 dict(source_frame=220,face='朝画面右侧，人物右眼与右嘴角为近侧；远侧点在轮廓附近仍有预测，但不可当作可见观测。',feet='远侧一只脚局部可见，近侧脚在画面下方/被截；需要逐个部位区分，不能把六个脚点一并启用。'),
 dict(source_frame=440,face='后脑勺，没有任何可见眼鼻嘴。检测器仍输出全部68个面部点；中位原始响应3.7756867并非概率。',feet='双脚及部分脚踝被沙发挡住；预测点落在靠背和腿边。'),
 dict(source_frame=660,face='朝画面左侧，人物左眼与左嘴角为近侧；远侧眼角、嘴角被遮挡。',feet='一只脚部分可见，另一只接近画面下边并与沙发重叠；预测踝点可落在腿上，尚未逐点核定。'),
 dict(source_frame=880,face='近正面八个部位可见，鼻底胡须区域的精确语义位置仍需单独检查。',feet='裁片只有手、衣摆和裤腿，脚已出画；仍有脚部预测落在裤腿靠近图像边缘。')]
review=dict(updated=stamp,actual_native_head_crops_reviewed=5,actual_native_foot_crops_reviewed=5,
    displayed_images={n:sha(C/n) for n in ['five-native-head-predictions.png','five-native-feet-predictions.png']},
    displayed_markers='RTMW predictions, not SMPL-X projections or measured truth',observations=notes,
    scope='Five registered contexts only; all177 detector outputs retained. Remaining172 image visibility is unknown.',
    resolved_subject_frames=177,frames_with_body_outside_raster=33,frames_with_foot_outside_raster=25,
    observation_mapping_numeric_parity_px=0.,physical_projection_accuracy_verified=False,
    foot_visibility_correction='The closer native crops show couch occlusion in source0 and440; a raster-boundary test alone is insufficient.',
    anatomical_family_binding='IBUG17..67 -> RTMW40..90 ->51 fixed SMPL-X triangles/barycentric anchors, repeated consistently across177frames',
    binding_count=9027,active_fit_points=0,SMPLX_fitted=False,training_started=False,quality_accepted=False)
save(C/'visual-review.json',review)
src=H/'field_semantic_face123.py';target=H/'indoor_semantic_face123.py';assert not target.exists()
text=src.read_text(encoding='utf-8').replace("PRE=REC/'field-preflight'","PRE=REC/'indoor-preflight'").replace('preflight/field','preflight/indoor')
text=text.replace('progress: direct 2D-to-SMPL-X objective implemented and known-geometry optimization tested; real-person result remains unaccepted',
    'progress: indoor177native observations retained; actual5head/5foot review found invisible predictions; only existing model ROI binding is authorized here')
ast.parse(text);target.write_text(text,encoding='utf-8',newline='\n')
save(PRE/'semantic-source-reuse.json',dict(updated=stamp,parent=str(src),parent_sha256=sha(src),candidate=str(target),candidate_sha256=sha(target),
    change='Only subject paths and provenance note. Existing FAN crop/weights/decoder unchanged.',
    purpose='Five native frames verify more detailed anatomical locations for the actual2D-to-SMPL-X bindings before full-sequence face observations.',
    already_known_rounding_fix='Reuse continuous inverse; no repeat of the synthetic SMPL-X test and no claim of independently measured subpixel accuracy.'))
print(json.dumps(dict(review_saved=True,next_five_face_images=5,source_reused=True,training=False)))
