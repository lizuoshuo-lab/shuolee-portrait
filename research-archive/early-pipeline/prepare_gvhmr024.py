from pathlib import Path
import json, hashlib
from datetime import datetime

BASE=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
ROOT=BASE/'current/smplx-gvhmr-body-motion-fresh-024'
CODE=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\repos\GVHMR-shuolee\shuolee')
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
now=datetime.now().astimezone().isoformat(timespec='seconds')
assert not (ROOT/'experiment-design.json').exists()
assert (ROOT/'diagnostic/audit.json').exists()
design=dict(created=now,state='prepared',previous_goal_turn='progress',
    previous_evidence='023 completed fresh fitting, actual review, publication and authorized duplicate cleanup.',
    hypothesis='Real stable torso/leg image features can constrain excessive predicted motion more specifically than forcing all body parts to share a static pose.',
    reason_for_refining_planned_test='284 raw-assigned feature trajectories reveal remaining pelvis/spine/leg motion mismatch. Use selected real observations to constrain shared motion only where supported; do not hard-freeze whole body.',
    arms=['control','body-texture'],updates_per_arm=1200,total_new_body_updates_planned=2400,
    new_gaussian_updates=0,old_person_state_initialization=False,
    initial_source='Regenerate independently from official raw GVHMR output, real image observations, cameras and licensed SMPL-X.',
    held_fixed=['59 real cameras','original image masks/keypoints','hand observations','foot texture split','optimizer/seed/schedule','all other loss weights'],
    intervention='Add0.3 ramped torso/leg texture transfer loss,10px normalization,100-300step ramp; identical no-additional-loss control.',
    gate=dict(diagnostic_median_reduction_fraction_min=.5,visible_body_median_increase_px_max=2.,
              no_obvious_fixed_side_or_limb_regression=True,continuous_body_acceptance_required=True),
    limits=['Raw single-bone assignment is approximate for loose clothing.', 'Static-consistent feature selection/source cameras reused; diagnostic split is not independent GT.',
            'A low relative-motion residual cannot establish absolute geometry at reference frame23.'],
    source_sha256={name:hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in ['refine_body_motion_fresh.py','body_texture_constraint.py','audit_body_texture_motion.py']},
    accepted=False,remote_deployed=False)
save(ROOT/'experiment-design.json',design)
state=read(BASE/'work-status.json');assert state['current_phase']=='gvhmr_shoe_motion_fresh_023'
state['updated']=now;state['current_phase']='gvhmr_body_motion_fresh_024'
state['current_decision']='已核对原始躯干/腿部纹理，准备两组各1200步从头对照：保持023足部约束，单独检验新增躯干/腿部纹理运动项。023仍为未验收候选，不启动高斯、不部署00。'
state['latest_local_execution']=dict(kind='gvhmr_body_motion_fresh_024',state='prepared',running=False,directory=str(ROOT),planned_steps_per_arm=1200,arms=design['arms'],new_body_fit_updates=0,gaussian_updates=0,accepted=False)
save(BASE/'work-status.json',state)
print(json.dumps({'prepared':str(ROOT),'planned_updates':2400}))
