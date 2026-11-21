from pathlib import Path
import json,hashlib,datetime,shutil
B=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\outputs\mini-splatting2-mip-depth-001\shuolee')
R=B/'current/person-static-human-surface-init-fresh-083'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=read(R/'clothing-init/audit.json');assert a['initialization_revision']==2 and a['replaced_face_points']==1772 and a['all_other_points_bitwise_identical']
assert a['point_sha256']==sha(R/'clothing-init/points.npz')
images=['initialization-review.jpg']+[f'patch-review-{i:06d}.jpg' for i in [34,70,84,136]]
save(R/'initialization-review.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),allow_training=True,
    initialization_revision=2,actual_visual_review=True,images=[dict(path='clothing-init/'+f,sha256=sha(R/'clothing-init'/f)) for f in images],
    observation='All6native initialization silhouettes preserve the clothed person, hands and feet. Four source-image face overlays show the central face samples on a single neutral surface. Peripheral unsupported samples stay at fresh hull positions. No full-head or neck seam is replaced.',
    initialization_only=True,quality_accepted=False,head_shape_identity_accepted=False,
    altered_initial_points=1772,unchanged_points=198228,initial_points=200000,
    fresh_recalculation=True,no_old_personalized_model_loaded=True,
    limitation='The mean face is approximate, and unchanged peripheral points can still carry incorrect geometry. Mask support is an initialization precondition, not held-out accuracy evidence. Improvement after training remains unknown.',
    stop=['nonfinite values','input mismatch','catastrophic visible loss of person','18000updates maximum']))
for repo,name in [('PGSR-shuolee','shuolee_static.py'),('PGSR-shuolee','camera_contract.py'),('PGSR-shuolee','observed_correspondence.py'),
                 ('GVHMR-shuolee','shuolee/fit_fresh_sequence.py'),('GaussianAvatars_SMPLX-shuolee','initialize_static_clothing_cloud.py')]:
    src=Path(r'C:\Users\Administrator\Desktop\GS-SLAM-Lab\repos')/repo/name
    dest=R/'source-final'/repo/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
s=read(B/'work-status.json');s.update(updated=datetime.datetime.now().astimezone().isoformat(),current_phase='083 human-surface initialization passed for bounded training',local_training_running=False)
s['latest_local_execution'].update(state='initialization_reviewed',running=False,gaussian_training=False,steps=18000)
s['next_bounded_work']['new_training_ready']=True
s['iteration_policy']['new_training_ready']=True;s['iteration_policy']['required_next_protocol']=str(R/'analysis-before-training.json')
s['latest_diagnostic']=dict(round='082',status='completed_reviewed',path=str(B/'current/human-head-surface-initialization-082'),candidate_adopted=False,gaussian_updates=0)
save(B/'work-status.json',s)
print(json.dumps(dict(allowed_bounded_training=True,points=200000,changed=1772)))
