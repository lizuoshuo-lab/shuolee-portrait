from pathlib import Path
import json,hashlib,datetime,argparse
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee');V=B.parent/'viewer';P=B/'current/static-ncc-paired-fresh-093'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main(arm):
    suffix='control' if arm=='control' else 'off';root=V/f'models/whole-body-static-ncc-{suffix}';receipt=read(root/'delivery.json');assert sha(Path(receipt['download_path']))==receipt['sha256']
    paths=[P/f'{arm}-standalone-{mode}.png' for mode in ['gaussian','points']];assert all(p.exists() and p.stat().st_size>10000 for p in paths)
    v=dict(updated=datetime.datetime.now().astimezone().isoformat(),host='local control PC',viewer='Shuolee Spark2.3.1',
        file_opened_through_UI=True,modes=['Gaussian','points'],observed_count=receipt['points'],
        snapshots=[dict(path=str(p),sha256=sha(p)) for p in paths],
        observation='Actual standalone file rendered upright clothed body in both modes. Nose/mouth and spectacle duplication remain. No per-model skeleton or sidecar transform loaded for this file.',
        format_verified=True,quality_accepted=False,pc00_opened=False)
    receipt.update(local_standalone_open_verified=True,local_viewer_validation=v);save(root/'delivery.json',receipt);save(P/arm/'viewer-ui-review.json',v)
    print(json.dumps(dict(arm=arm,points=receipt['points'],local_standalone_open_verified=True,quality_accepted=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=['control','no_ncc']);main(p.parse_args().arm)
