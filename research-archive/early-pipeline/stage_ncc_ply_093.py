"""Publish only byte-verified standard+Y files; quality acceptance is separate."""
from pathlib import Path
import json,hashlib,shutil,datetime,numpy as np,argparse
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer';P=B/'current/static-ncc-paired-fresh-093'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');q.replace(p)
def copy(src,dst):
    q=dst.with_suffix(dst.suffix+'.tmp');shutil.copy2(src,q);q.replace(dst)
def main(selected_arm='all'):
    published=[];base=read(V/'model-whole-body-static-human-init.json');old=read(V/'models/whole-body-static-human-init/delivery.json')
    for arm,suffix,label in [('control','control','保留NCC'),('no_ncc','off','关闭NCC')]:
        if selected_arm!='all' and selected_arm!=arm:continue
        root=P/arm;run=root/'gaussian-run';r=read(run/'result.json');delivery=read(run/'viewer-export/delivery.json');source=run/'viewer-export/point_cloud.ply'
        assert r['steps']==18000 and r['from_scratch'] and r['frames_used']==175 and delivery['compatibility_passed'] and sha(source)==delivery['sha256']
        # Reusing camera metadata is valid only for the identical global rigid transform and source matrices.
        assert delivery['world_to_viewer']==old['world_to_viewer']
        assert read(root/'data/training-inputs.json')['records']==read(B/'current/person-static-human-surface-init-fresh-083/data/training-inputs.json')['records']
        modelid='whole-body-static-ncc-'+suffix;dest=V/'models'/modelid;dest.mkdir(exist_ok=True)
        name=f'shuolee_093_{arm}_s18000_f0100.ply';copy(source,dest/name)
        assert sha(dest/name)==delivery['sha256']
        receipt={**delivery,'round':'093','arm':arm,'download_path':str(dest/name),'download_url':f'http://127.0.0.1:8870/models/{modelid}/{name}?download=1','viewer_url':f'http://127.0.0.1:8870/?model={modelid}','quality_accepted':False,'local_standalone_open_verified':False,'pc00_opened_this_round':False}
        old_receipt=read(dest/'delivery.json') if (dest/'delivery.json').exists() else {}
        if old_receipt.get('sha256')==delivery['sha256'] and old_receipt.get('local_standalone_open_verified'):
            receipt.update(local_standalone_open_verified=True,local_viewer_validation=old_receipt['local_viewer_validation'])
        save(dest/'delivery.json',receipt)
        meta={**base,'label':f'093 {label} · 全新静态训练','model':f'models/{modelid}/{name}','sha256':delivery['sha256'],'gaussianCount':r['points'],
            'steps':18000,'trained':True,'fromScratch':True,'oldCheckpointLoaded':False,'qualityAccepted':False,'deliveryAccepted':False,'plyCompatibilityPassed':True,
            'status':f'093 {label}对照：175帧、20万点从头训练完成，尚待错位与细节质量审阅。','latestDiagnostic':'093','report':'body-smplx.html#static-surface-fit','deliveryReceipt':f'models/{modelid}/delivery.json','registeredCameras':175}
        save(V/f'model-{modelid}.json',meta)
        published.append(dict(arm=arm,modelId=modelid,points=r['points'],sha256=delivery['sha256'],file=str(dest/name),download_url=receipt['download_url'],viewer_url=receipt['viewer_url']))
    js=V/'viewer.js';code=js.read_text(encoding='utf8')
    for record in reversed(published):
        if json.dumps(record['modelId']) not in code:code=code.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(record['modelId'])+', ',1)
    js.write_text(code,encoding='utf8')
    item=published[-1];latest=dict(round='093',modelId=item['modelId'],sha256=item['sha256'],points=item['points'],compatibilityPassed=True,qualityAccepted=False,
        updated=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),viewer_url='http://127.0.0.1:8870/?model=latest',download_url=item['download_url'],report='body-smplx.html#static-surface-fit',latestDiagnostic='093',candidateNotBest=True)
    latest['revision']=hashlib.sha256(json.dumps(latest,sort_keys=True).encode()).hexdigest();save(V/'latest-result.json',latest)
    save(P/'staged-deliveries.json',dict(arms=published,actual_viewer_checked=False,quality_accepted=False))
    print(json.dumps(dict(published=published,quality_accepted=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['all','control','no_ncc'],default='all');main(p.parse_args().arm)
