from pathlib import Path
import json,hashlib,datetime,shutil,sys
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
arm=sys.argv[1];assert arm in ['control','no_ncc'];R=B/'current/static-ncc-paired-fresh-093'/arm
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=read(R/'clothing-init/audit.json');assert a['fresh_array_equal083'] and a['replaced_face_points']==1772 and a['point_sha256']==sha(R/'clothing-init/points.npz')
images=['initialization-review.jpg']+[f'patch-review-{i:06d}.jpg' for i in [34,70,84,136]]
save(R/'initialization-review.json',dict(updated=datetime.datetime.now().astimezone().isoformat(),arm=arm,allow_training=True,actual_visual_review=True,
    images=[dict(path='clothing-init/'+f,sha256=sha(R/'clothing-init'/f)) for f in images],
    observation='Six native silhouettes preserve clothed torso, hands and feet. Four source-image overlays show approximate single neutral central surface; unsupported peripheral samples retain fresh hull positions. Neutral identity and fine nose/eyes alignment are not accepted.',
    initialization_only=True,quality_accepted=False,initial_points=200000,altered_initial_points=1772,unchanged_points=198228,fresh_recalculation=True,old_person_model_loaded=False,
    limitation='Mask-supported neutral template is an initializer, not true personal anatomy. The paired experiment deliberately keeps this same approximate initializer.' ))
# Seal the final executable sources after adding trace RNG guards; before any optimizer step.
sources=read(R/'source-snapshot.json')
for item in sources:
    p=Path(item['path'].replace('/mnt/c/','C:/'));shutil.copy2(p,R/'source-final'/p.name);item['sha256']=sha(p)
save(R/'source-snapshot.json',sources)
state=read(B/'work-status.json');state.update(updated=datetime.datetime.now().astimezone().isoformat(),local_training_running=False,current_phase=f'093 {arm}: fresh initialization visually reviewed')
state['latest_local_execution']=dict(round='093',arm=arm,run=str(R),state='initialization_reviewed',running=False,gaussian_training=False,steps=18000)
state['next_bounded_work']['new_training_ready']=True
save(B/'work-status.json',state)
print(json.dumps(dict(arm=arm,training_allowed=True,quality_accepted=False)))
