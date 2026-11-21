from pathlib import Path
import json, datetime, hashlib, urllib.request, argparse
p=argparse.ArgumentParser();p.add_argument('--screenshot',default='C:/Users/Administrator/Documents/New project/shuolee-slide-update/camera116-review.png');args=p.parse_args()
O=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/portrait-alignment-open-source-116')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
q=read(O/'queue-state.json');q.update(updated=now,automation_id='automation',automation_name='人像相机与跨视角对齐',automation_status='ACTIVE',automation_interval_minutes=30)
save(O/'queue-state.json',q)
save(O/'automation-receipt.json',{'automation_id':'automation','name':'人像相机与跨视角对齐','status':'ACTIVE','kind':'heartbeat','interval_minutes':30,'created_via':'mcp__codex_app__automation_update','create_result':'Created automation in the app; status ACTIVE.','view_result':'Rendered automation card in the app.','same_chat':True,'requires_app_running':True,'updated':now})
shot=Path(args.screenshot);assert shot.exists()
checks=[]
for suffix in ['body-smplx.html','body-camera116/capture-provenance.json','body-camera116/hovr-frame-timing-check.json','body-camera116/candidates.json','body-camera116/protocol.json','body-camera116/sfm-track-support.json','body-camera116/evidence-ledger.json']:
    with urllib.request.urlopen('http://127.0.0.1:8870/'+suffix,timeout=10) as response:
        data=response.read();checks.append({'resource':suffix,'status':response.status,'bytes':len(data)})
        if suffix=='body-smplx.html':assert '原始相机信息与跨视角人像对齐' in data.decode('utf8')
        else:json.loads(data)
save(O/'viewer-ui-review.json',{'updated':now,'scope':'Existing browser report page, not Gaussian or point-cloud validation','url':'http://127.0.0.1:8870/body-smplx.html#alignment116','actual_browser_opened':True,'visible_text_reviewed':True,'screenshot':str(shot),'screenshot_sha256':hashlib.sha256(shot.read_bytes()).hexdigest(),'http_resources':checks,'new_ply_created':False,'new_training_started':False,'ghosting_fixed':False,'computer00_verified':False})
p=read(O/'publication.json');p.update(visual_review_pending=False,visual_review='viewer-ui-review.json',automation_id='automation');save(O/'publication.json',p)
print(json.dumps({'automation':'ACTIVE','review_passed':True,'resources_checked':len(checks),'root_cause_resolved':False,'new_training_started':False},ensure_ascii=False))
