from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil
BASE=Path(r'C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/new-capture-118')
ROOT=Path(r'\\wsl.localhost\Ubuntu-20.04\opt\gs-slam-lab\shuolee-new-capture-118-native-fixedk')
PROJECT=Path('D:/GS-SLAM-Data/shuolee-portrait-new-capture-118')
OUT=Path('D:/00/工作日志/2026-10-27/新人物118')
OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
measure=read(BASE/'native-fit-review/measurements.json')
review=dict(created=datetime.now(timezone.utc).isoformat(),selected_arm='reference',
    reviewed_frames=measure['selected_frames'],fit_sha256=measure['fit_sha256']['reference'],
    actual_review=True,formal_geometry_accepted=False,quality_accepted=False,
    images=['native-fit-review/'+x for x in ['body_original_reference_surface-visibility.jpg','face_original_reference_surface-visibility.jpg','feet_original_reference_surface-visibility.jpg']],
    findings=[
        'Reviewed original/reference/visibility-variant same-crop images at all six fixed cameras. Front and side head/torso broadly overlap, with remaining profile/neck/arm differences.',
        'Both meshes still have foot/ankle placement problems in visible side views; some observations are on calf or furniture/occlusion boundaries. Self-occlusion alone does not resolve these.',
        'On identical fixed detector rows, face median 4.277 to 4.298px and foot median 18.435 to 26.244px; visibility variant is not selected.',
        'Visibility suppresses 969/2041 supported face and 96/110 supported foot rows. Lower masked loss does not establish improvement.',
        'Camera candidate still fails independent whole-body/background protection. Physical crop/rectification and FISHEYE_8 remain unverified.'
    ],next_action='One preregistered fresh fixed-K research run using the reference fit; inspect at step1000 before spending the remaining18000-step budget. Stop if gross nose/foot artifacts persist. No formal admission or old checkpoint loading.')
save(BASE/'native-fit-visual-review.json',review)
save(ROOT/'fit/visual-review.json',review)
plan=dict(created=datetime.now(timezone.utc).isoformat(),kind='finite_research_training_with_early_review',
    max_steps=18000,initial_points=200000,initial_optimizer_entries=0,frames=89,training_wh=[1600,899],
    fit_sha256=review['fit_sha256'],camera_sha256=sha(ROOT/'data/training-inputs.json'),
    training_source_sha256=sha(PROJECT/'src/shuolee_portrait/training.py'),
    camera_estimator='Fixed supplied-K crop hypothesis; fresh OpenCV essential/PnP/bundle candidate, not COLMAP pose fitting',
    K_fixed=True,raw_distortion_coefficients_applied=False,formal_calibration_accepted=False,
    formal_training_admitted=False,quality_accepted=False,old_gaussian_loaded=False,old_optimizer_loaded=False,
    initialization='Fresh 1200-step reference SMPL-X fit from fresh video detector/prior; 200000 surface samples',
    early_review_step=1000,review_wait_seconds=900,
    stop_conditions=['Nonfinite computation, missing frames, failed CUDA projection or native PLY roundtrip',
        'At1000 actual original/render same-camera review shows gross duplicate nose or misplaced/floating feet; do not call it solved or automatically run to18000',
        'Review missing at15minute timeout; stop instead of unconditional continuation'],
    known_limitations=review['findings'],
    interpretation='Changed camera estimate AND regenerated raw-pixel body observations relative to old probe; this is a pipeline research iteration, not a single-factor proof of the ghosting cause.')
save(BASE/'fixedk-research-training-plan.json',plan)
for p in (BASE/'native-fit-review').iterdir():shutil.copy2(p,OUT/p.name)
shutil.copy2(BASE/'native-fit-visual-review.json',OUT/'拟合对照结论.json')
shutil.copy2(BASE/'fixedk-research-training-plan.json',OUT/'新内参训练计划.json')
protocol=read(BASE/'protocol.json');protocol['current_stage']='fresh_fixedK_research_training_with_1000_step_actual_review';protocol['current_plan']='fixedk-research-training-plan.json';protocol['formal_training_admitted']=False;protocol['quality_accepted']=False
save(BASE/'protocol.json',protocol)
print(json.dumps(dict(review_saved=True,formal_training_admitted=False,reference_fit_selected=True),ensure_ascii=False))
