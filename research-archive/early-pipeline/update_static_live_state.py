from pathlib import Path
import json,datetime,hashlib
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');R=B/'current/person-static-no-scale-fresh-065';S=B/'current/person-static-no-multiview-fresh-066'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
review=read(R/'review/browser-observation.json');review['created']=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
review['proof_sha256']={n:hashlib.sha256((R/'review'/n).read_bytes()).hexdigest() for n in ['standalone-gaussian.png','standalone-centers.png']}
save(R/'review/viewer-validation.json',review)
for p in [R/'phase-conclusion.json',B.parent/'viewer/models/whole-body-static-no-scale/delivery.json']:
    x=read(p);x['local_viewer_validation']=review;x['local_viewer_passed']=True;save(p,x)
registry=read(B/'gaussian-deliveries.json');registry['rounds']['065']['local_viewer_validation']=review;save(B/'gaussian-deliveries.json',registry)
st=read(B/'work-status.json');progress=read(S/'gaussian-run/progress.json')
st['latest_local_execution']=dict(kind='static_ablation_066',state='fresh_static_pgsr_training',running=progress['running'],directory=str(S),
    steps=progress['step'],target_steps=18000,from_scratch=True,representation='shared_static_world',quality_accepted=False)
st['local_training_running']=progress['running'];st['updated']=review['created'];save(B/'work-status.json',st)
print(json.dumps(st['latest_local_execution']))
