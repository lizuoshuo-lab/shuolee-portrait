"""Stage verified models; latest promotion additionally requires actual UI proof."""
import argparse,json,hashlib,shutil,datetime
from pathlib import Path
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer';PAIR=B/'current/avatar-repository-pair-114'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');q.replace(p)
def main(arm,promote=False):
    root=PAIR/arm;run=root/'gaussian-run';result=read(run/'result.json');delivery=read(run/'viewer-export/delivery.json')
    assert result['steps']==18000 and result['from_scratch'] and delivery['compatibility_passed']
    base=read(V/'model-whole-body-native-align.json');old=read(V/'models/whole-body-native-align/delivery.json')
    assert old['world_to_viewer']==delivery['world_to_viewer']
    assert read(root/'data/training-inputs.json')['records']==read(B/'current/person-static-native-align-fresh-113/data/training-inputs.json')['records']
    tag='114a' if arm=='gaussianavatar' else '114b';modelid='whole-body-avatar114-'+('ga' if arm=='gaussianavatar' else 'smplx');name=f'shuolee_{tag}_s18000_f0100.ply'
    dest=V/'models'/modelid;dest.mkdir(exist_ok=True);src=run/'viewer-export/point_cloud.ply';assert sha(src)==delivery['sha256']
    if (dest/name).exists():assert sha(dest/name)==delivery['sha256']
    else:shutil.copy2(src,dest/name)
    assert sha(dest/name)==delivery['sha256']
    title='GaussianAvatar 全身' if arm=='gaussianavatar' else 'GaussianAvatars_SMPLX 本地适配'
    receipt={**delivery,'round':tag,'experiment':str(root),'download_path':str(dest/name),'local_path':str(src),'download_url':f'http://127.0.0.1:8870/models/{modelid}/{name}?download=1','viewer_url':f'http://127.0.0.1:8870/?model={modelid}','local_standalone_open_verified':False,'pc00_opened_this_round':False,'quality_accepted':False,'static_model':True}
    if promote:
        ui=read(root/'viewer-ui-review.json');assert ui['file_opened_through_UI'] and ui['format_verified'] and ui['source_sha256']==delivery['sha256'] and ui['observed_count']==result['points']
        assert set(ui['modes'])=={'Gaussian','points'} and all(sha(Path(q['path']))==q['sha256'] for q in ui['screenshots'])
        receipt.update(local_standalone_open_verified=True,local_viewer_validation=ui)
    save(dest/'delivery.json',receipt)
    meta={**base,'label':f'{tag} {title} · 从头18,000步','model':f'models/{modelid}/{name}','sha256':delivery['sha256'],'gaussianCount':result['points'],'status':f'{title}：训练完成，格式核验通过；重影改善尚待对照。','latestDiagnostic':tag,'report':'body-smplx.html#avatar114-pair','deliveryReceipt':f'models/{modelid}/delivery.json','qualityAccepted':False,'deliveryAccepted':promote,'plyCompatibilityPassed':True,'diagnosticOnly':False,'sharedFacialWorldPoints':None,'sharedMaterialWorldPoints':None}
    save(V/f'model-{modelid}.json',meta)
    js=V/'viewer.js';code=js.read_text(encoding='utf-8')
    if json.dumps(modelid) not in code:
        assert 'const BUILTINS = [' in code;code=code.replace('const BUILTINS = [','const BUILTINS = ['+json.dumps(modelid)+', ',1);js.write_text(code,encoding='utf-8')
    public=V/'body-avatar114'/arm;public.mkdir(parents=True,exist_ok=True)
    for p in (root/'evaluation').iterdir():
        if p.is_file() and p.suffix in ['.jpg','.mp4','.json']:shutil.copy2(p,public/p.name)
    save(root/'staged-delivery.json',receipt)
    if promote:
        latest=dict(round=tag,modelId=modelid,sha256=receipt['sha256'],points=result['points'],compatibilityPassed=True,qualityAccepted=False,updated=datetime.datetime.now().astimezone().isoformat(),viewer_url='http://127.0.0.1:8870/?model=latest',download_url=receipt['download_url'],report='body-smplx.html#avatar114-pair',candidateNotBest=True,diagnosticOnly=False,status=meta['status'],revision=receipt['sha256'])
        save(V/'latest-result.json',latest);ledger=read(B/'gaussian-deliveries.json');ledger['rounds'][tag]=receipt;ledger['latest_round']=tag;save(B/'gaussian-deliveries.json',ledger)
    print(json.dumps(dict(arm=arm,model=modelid,download=receipt['download_url'],promoted=promote),ensure_ascii=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['gaussianavatar','smplx'],required=True);p.add_argument('--promote-after-ui',action='store_true');q=p.parse_args();main(q.arm,q.promote_after_ui)
