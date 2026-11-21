from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
H=Path(__file__).parent
T=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001')
F=T/'shuolee/current/local-clothed-surface-123/field-preflight';C=F/'coverage'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert not (C/'visual-review.json').exists()
inventory=read(C/'review-inventory.json');assert len(inventory['items'])==5
review=dict(updated=datetime.now(timezone.utc).isoformat(),state='five_fixed_native_contexts_reviewed',
    coverage_frames=177,actual_source_frames_reviewed=[0,220,440,660,880],actual_subject_native_crops_reviewed=5,
    actual_head_native_crops_reviewed=5,actual_feet_native_crops_reviewed=5,
    sources={n:sha(C/n) for n in ['review-inventory.json','five-native-person-crops.png','result.json','observations.npz']},
    observations=[
        dict(source_frame=0,view='front',face='Visible face and eyes/nose/mouth, no localization accuracy claimed.',feet='Both shoes visible. Trouser hems occlude ankles and portions of the shoes.',
             roi='Person and face region overlap the intended subject. Bag and clothes are separate surfaces from the body.'),
        dict(source_frame=220,view='side',face='One side is visible, far eye and facial surface are self-occluded. Whole 68-point face is not valid observed geometry.',
             feet='Two shoes partly visible at different depths, ankles and heels partly hidden by trousers.',roi='Correct person. Far hand is occluded; detected hand coordinates do not establish visibility.'),
        dict(source_frame=440,view='back',face='No visible facial surface. All 68 semantic face outputs remain predictions, not measured facial observations.',
             feet='Back of trousers covers most of both shoe heels. Predicted six foot coordinates are not six independently visible physical points.',
             roi='Face ROI [1324.409,663.731,1353.866,697.031] lies on rear head in the original image.',
             raw_face_response_min=3.094046115875244,raw_face_response_median=3.6205220222473145,
             diagnosis='A finite in-image output with a positive response does not imply a visible anatomical feature.'),
        dict(source_frame=660,view='opposite_side',face='One facial side visible; the far eye and far skin surface are occluded.',
             feet='Both shoes partly visible, ankles hidden by trouser hems.',roi='Correct subject. Far hand and portions of the bag are occluded.'),
        dict(source_frame=880,view='front',face='Face visible. Similar first/last appearance does not prove no movement.',
             feet='Both shoes visible with ankles/heel partly obscured by trousers.',roi='Correct subject and face region.')],
    accepted_scope='Detector identity and broad regional coverage in the five preregistered contexts only. Other 172 frame pixels have not been visually audited here.',
    confirmed_failure_mode='Back-head and self-occluded semantic face/foot predictions cannot be treated as measured visible material correspondences.',
    raw_scores_are_probabilities=False,raw_scores_are_localization_uncertainty=False,
    physical_face_correspondences_verified=0,global_staticity_verified=False,
    camera_solved=False,SMPLX_fitted=False,training_started=False,quality_accepted=False,
    next='Separate broad ROI use from fitting observations. Preserve all predictions and explicitly retain unknown visibility. Before any fitting, physically review materials and use visible observations only without transferring HELD points to FIT or filtering by residual. No error-threshold or detector sweep.')
save(C/'visual-review.json',review)
# The first presentation draft predates this coverage run. Update its notes before publication.
p=H/'update_deck_camera_entry123.mjs';s=p.read_text(encoding='utf-8')
old='下一步按已授权field队列做一次177帧已有YOLO/RTMW区域覆盖核验，计划已登记但未执行。只用现有官方权重定位区域，后续材料对应必须独立核验，不以相同面部编号充当同一皮肤点。'
new='田地人物首次177帧区域核验已完成，耗时87秒，每帧均找到所跟踪人物，零越界仅证明坐标在画面内。实际查看五个固定原生人物、头和脚裁片。第440帧只有后脑勺，检测器仍输出68个面部点，原始响应中位数3.62并非概率。侧脸的远侧五官、背面裤腿下的鞋后跟也不可直接作可见观测。这次只定位区域，尚未核验物理材料对应、求相机或拟合。'
assert old in s;s=s.replace(old,new).replace('及field-preflight/coverage-plan.json。','及field-preflight/coverage-plan.json、coverage/result.json和coverage/visual-review.json。')
p.write_text(s,encoding='utf-8')
print(json.dumps(dict(review=str(C/'visual-review.json'),PPT_notes_corrected_before_publication=True),ensure_ascii=False))
