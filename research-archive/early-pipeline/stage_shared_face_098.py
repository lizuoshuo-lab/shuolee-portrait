from pathlib import Path
import json,hashlib,shutil
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer'
R=B/'current/person-static-shared-face-fresh-098'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');t.replace(p)
result=read(R/'gaussian-run/result.json');config=read(R/'gaussian-run/config.json');delivery=read(R/'gaussian-run/viewer-export/delivery.json')
assert result['steps']==18000 and result['frames_used']==175 and config['shared_face_world']['enabled'] and delivery['compatibility_passed']
base=read(V/'model-whole-body-static-ncc-control.json');old=read(V/'models/whole-body-static-ncc-control/delivery.json')
assert old['world_to_viewer']==delivery['world_to_viewer']
assert read(R/'data/training-inputs.json')['records']==read(B/'current/static-ncc-paired-fresh-093/control/data/training-inputs.json')['records']
modelid='whole-body-shared-world';dest=V/'models'/modelid;dest.mkdir(exist_ok=True)
name='shuolee_098_s18000_f0100.ply';src=R/'gaussian-run/viewer-export/point_cloud.ply'
assert sha(src)==delivery['sha256'];shutil.copy2(src,dest/name);assert sha(dest/name)==delivery['sha256']
status='098共享空间点约束：175帧、20万点从头训练18,000步。导出兼容通过，重影与全身效果待最终审阅。'
receipt={**delivery,'round':'098','download_path':str(dest/name),'download_url':f'http://127.0.0.1:8870/models/{modelid}/{name}?download=1',
         'viewer_url':f'http://127.0.0.1:8870/?model={modelid}','local_standalone_open_verified':False,'pc00_opened_this_round':False,'quality_accepted':False}
save(dest/'delivery.json',receipt)
meta={**base,'label':'098 共享三维特征点 · 全新18,000步','model':f'models/{modelid}/{name}','sha256':delivery['sha256'],
      'gaussianCount':result['points'],'status':status,'latestDiagnostic':'098','report':'body-smplx.html#shared-world-points',
      'deliveryReceipt':f'models/{modelid}/delivery.json','sharedFacialWorldPoints':50,'independentTest':False,
      'qualityAccepted':False,'deliveryAccepted':False,'plyCompatibilityPassed':True,'diagnosticOnly':False}
save(V/f'model-{modelid}.json',meta)
js=V/'viewer.js';code=js.read_text(encoding='utf-8')
if json.dumps(modelid) not in code:code=code.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(modelid)+', ',1);js.write_text(code,encoding='utf-8')
save(R/'staged-delivery.json',receipt)
print(json.dumps(dict(modelid=modelid,points=result['points'],download_url=receipt['download_url'],latest_not_changed_before_UI_check=True)))
