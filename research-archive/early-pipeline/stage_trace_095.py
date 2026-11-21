from pathlib import Path
import json,hashlib,datetime,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer';R=B/'current/whole-body-early-trace-fresh-095'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
run=R/'gaussian-run';result=read(run/'result.json');d=read(run/'viewer-export/delivery.json')
assert result['steps']==1800 and result['diagnostic_stop'] and result['frames_used']==175 and d['compatibility_passed']
source=run/'viewer-export/point_cloud.ply';assert sha(source)==d['sha256']
base=read(V/'model-whole-body-static-ncc-control.json');old=read(V/'models/whole-body-static-ncc-control/delivery.json')
assert old['world_to_viewer']==d['world_to_viewer']
assert read(R/'data/training-inputs.json')['records']==read(B/'current/static-ncc-paired-fresh-093/control/data/training-inputs.json')['records']
modelid='whole-body-early-trace';dest=V/'models'/modelid;dest.mkdir(exist_ok=True);name='shuolee_095_s01800_f0100.ply'
shutil.copy2(source,dest/name);assert sha(dest/name)==d['sha256']
status='095早期诊断模型：175帧、20万点从头运行至1,800步。全身扩散和重影仍在，尚未通过质量验收。'
receipt={**d,'round':'095','download_path':str(dest/name),'download_url':f'http://127.0.0.1:8870/models/{modelid}/{name}?download=1',
    'viewer_url':f'http://127.0.0.1:8870/?model={modelid}','local_standalone_open_verified':False,'pc00_opened_this_round':False,'quality_accepted':False}
save(dest/'delivery.json',receipt)
meta={**base,'label':'095 早期训练追踪 · 1,800步诊断','model':f'models/{modelid}/{name}','sha256':d['sha256'],'gaussianCount':result['points'],
    'steps':1800,'trained':True,'fromScratch':True,'oldCheckpointLoaded':False,'qualityAccepted':False,'deliveryAccepted':False,'plyCompatibilityPassed':True,
    'status':status,'latestDiagnostic':'095','report':'body-smplx.html#whole-body-alignment','deliveryReceipt':f'models/{modelid}/delivery.json','registeredCameras':175,'diagnosticOnly':True}
save(V/f'model-{modelid}.json',meta)
js=V/'viewer.js';code=js.read_text(encoding='utf-8')
if json.dumps(modelid) not in code:code=code.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(modelid)+', ',1);js.write_text(code,encoding='utf-8')
latest=dict(round='095',modelId=modelid,sha256=d['sha256'],points=result['points'],compatibilityPassed=True,qualityAccepted=False,
    updated=datetime.datetime.now().astimezone().isoformat(),viewer_url='http://127.0.0.1:8870/?model=latest',download_url=receipt['download_url'],
    report='body-smplx.html#whole-body-alignment',latestDiagnostic='095',candidateNotBest=True,diagnosticOnly=True,status=status)
latest['revision']=hashlib.sha256(json.dumps(latest,sort_keys=True).encode()).hexdigest();save(V/'latest-result.json',latest)
save(R/'staged-delivery.json',receipt)
print(json.dumps(dict(modelId=modelid,points=result['points'],sha256=d['sha256'],download_url=receipt['download_url'])))
