from pathlib import Path
from datetime import datetime
import json,hashlib

lab=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab')
base=lab/'outputs/mini-splatting2-mip-depth-001/shuolee'
repo=lab/'repos/GVHMR-shuolee/shuolee'
phase=base/'current/smplx-gvhmr-shoe-motion-fresh-023'
source=repo/'refine_surface_contact_fresh.py';target=repo/'refine_shoe_motion_fresh.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)=='7b875643aaed175e8702eb830c3bc68018abc319b1eed9995f021247c025cfe7'
assert not phase.exists() and not target.exists()
s=source.read_text('utf-8-sig')
replacements=[
 ('smplx-gvhmr-surface-contact-fresh-022','smplx-gvhmr-shoe-motion-fresh-023'),
 ('import register_fresh as reg','import register_fresh as reg\nimport shoe_texture_constraint as shoe'),
 ("    assert not (OUT/'fitted.npz').exists()", "    assert not (OUT/'fitted.npz').exists()\n    shoe_data=shoe.prepare(data,OUT)"),
 ('local_j=localj,corr=corr,pelvis=pelvis)', 'local_j=localj,corr=corr,pelvis=pelvis,pose_rotations=poses)'),
 ("        total+=100*torch.relu(.05-st['cam'][...,2]).square().mean()", "        texture=shoe.objective(st,shoe_data,E,K)\n        texture_ramp=min(1.,max(0.,(step-100)/200))\n        total+=.3*texture_ramp*texture\n        total+=100*torch.relu(.05-st['cam'][...,2]).square().mean()"),
 ('contact=contact,foot_stability=stable,smoothing=smooth)', 'contact=contact,foot_stability=stable,smoothing=smooth,shoe_texture=texture)'),
 ('diagnostic_texture_tracks_in_training=False', 'diagnostic_texture_tracks_in_training=True,shoe_feature_protocol="shoe-protocol.json"'),
 ("matched_reference='021 independently fitted raw-prediction control; not an initialization'", "matched_reference='022 independently fitted pointwise contact control; never an initialization'"),
 ('    started=time.monotonic();history=[]', '    shoe.check_contract(forward(),shoe_data,E,K,OUT)\n    started=time.monotonic();history=[]'),
]
for before,after in replacements:
    assert s.count(before)==1,before
    s=s.replace(before,after,1)
target.write_text(s,'utf-8');phase.mkdir()
stamp=datetime.now().astimezone().isoformat(timespec='seconds')
design=dict(created=stamp,state='prepared_not_started',new_body_updates=0,gaussian_updates=0,
    hypothesis='Stable original shoe tracks constrain relative ankle motion that 2D landmarks and contact alone leave inconsistent.',
    source_sha256=sha(target),constraint_sha256=sha(repo/'shoe_texture_constraint.py'),
    control='022 pointwise sole stability, independently trained',old_personalized_state_used=False,
    source_frames=59,steps=1200,fresh_official_prediction_initialization=True,
    changed_loss='Add0.3 shoe texture transfer loss,10px normalization,ramp100-300; same other losses and1200 budget.',
    tentative_component_gate='Unoptimized diagnostic median transfer error reduced by at least50%, visible body median increases at most2px, then inspect actual geometry.',
    full_acceptance_requires='Source overlays, fixed world multi-view and temporal shape, no full-geometry claim from the component gate.',
    statistical_limit='Both subsets were already inspected, and cameras used source features; not a blind test.',accepted=False)
(phase/'experiment-design.json').write_text(json.dumps(design,ensure_ascii=False,indent=2),'utf-8')
# Close the already-finished022 execution before introducing the new candidate.
p22=base/'current/smplx-gvhmr-surface-contact-fresh-022'
old=json.loads((p22/'run-state.json').read_text('utf-8-sig'))
assert old['step']==1200 and not old['running']
old.update(state='fitted_component_gain_insufficient',accepted=False,
    decision='Pointwise contact barely changes observed shoe-transfer consistency; old landmark-frame metric supplemented by verified kinematic-frame audit.')
(p22/'run-state.json').write_text(json.dumps(old,ensure_ascii=False,indent=2),'utf-8')
path=base/'work-status.json';status=json.loads(path.read_text('utf-8-sig'))
status.update(updated=stamp,current_phase='gvhmr_shoe_motion_fresh_023',local_training_running=False,
    current_decision='022从头1200步完成，整体未通过。核对官方运动链后，鞋面传递残差仍约30.83px。023从原始预测重新拟合，加入原视频的鞋面特征运动约束；保留另一半特征作未优化诊断。')
status['latest_local_execution']=dict(kind='gvhmr_shoe_motion_fresh_023',state='prepared_not_started',running=False,
    directory=str(phase),planned_steps=1200,new_body_fit_updates=0,gaussian_updates=0,old_person_fit_loaded=False,accepted=False)
ex=status['current_experiment'];ex.update(latest_fitting_trial='022',latest_fitting_steps=1200,
    latest_fitting_gaussian_steps=0,current_candidate=str(p22/'fitted.npz'),candidate_kind='SMPL-X fitting, not a Gaussian checkpoint',
    status='fitting_reviewed_not_accepted',candidate_accepted=False)
ex['previous_gaussian_candidate_steps']=ex.pop('candidate_steps',None)
ex['previous_gaussian_candidate_points']=ex.pop('candidate_points',None)
path.write_text(json.dumps(status,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(design,ensure_ascii=False))
