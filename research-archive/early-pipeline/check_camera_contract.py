from pathlib import Path
import sys,json,copy
L=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab');B=L/'outputs/mini-splatting2-mip-depth-001/shuolee'
sys.path.insert(0,str(L/'repos/PGSR-shuolee'))
from camera_contract import require_registered_cameras
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
candidate=read(B/'current/static-camera-registration-071/training-inputs-registered.json')
report={'actual_candidate_review_accepted':candidate['accepted_for_training'] is True}
def rejected(doc):
    try:require_registered_cameras(doc)
    except AssertionError:return True
    return False
pending=copy.deepcopy(candidate);pending['accepted_for_training']=False
assert rejected(pending);report['pending_rejected']=True
old=read(B/'current/person-static-mask-foreground-fresh-070/data/training-inputs.json')
old.update(camera_registration_complete=True,accepted_for_training=True)
assert rejected(old);report['old_interpolation_rejected']=True
valid=copy.deepcopy(candidate);valid['accepted_for_training']=True
report['validated_structure']=require_registered_cameras(valid)
for case in ['wrong_frame','invalid_rotation','duplicate_frame','fake_registered']:
    bad=copy.deepcopy(valid)
    if case=='wrong_frame':bad['records'][1]['camera_registration']['raw_source_frame']+=1
    if case=='invalid_rotation':bad['records'][1]['w2c'][0][0]+=0.1
    if case=='duplicate_frame':bad['records'][1]['source_frame_index']=bad['records'][0]['source_frame_index']
    if case=='fake_registered':bad['records'][1]['camera_registration']['valid']=False
    assert rejected(bad),case;report[case+'_rejected']=True
(B/'current/static-camera-registration-071/training-gate-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report))
