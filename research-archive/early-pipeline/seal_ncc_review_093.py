"""Close the bounded093 review, never the unsolved reconstruction goal."""
from pathlib import Path
import json, hashlib, datetime

B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
P=B/'current/static-ncc-paired-fresh-093';V=B.parent/'viewer'
W=Path('C:/Users/Administrator/Documents/New project/shuolee-slide-update')
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8');q.replace(p)

def main():
    d=read(P/'decision.json');assert d['training_completed'] and d['actual_visual_review_complete']
    st=read(P/'staged-deliveries.json');receipts=[]
    for item in st['arms']:
        r=read(V/f'models/{item["modelId"]}/delivery.json')
        ui=read(P/item['arm']/'viewer-ui-review.json')
        cam=read(P/item['arm']/'viewer-camera-audit.json')
        assert r['local_standalone_open_verified'] and ui['file_opened_through_UI']
        assert sha(Path(item['file']))==r['sha256']==item['sha256']
        assert all(sha(Path(s['path']))==s['sha256'] for s in ui['snapshots'])
        receipts.append(r)
    deck=read(W/'human093-deck-receipt.json');assert deck['canonical_replaced']
    assert sha(Path(deck['source']))==deck['final_sha256']
    proof=[P/'published-report.png',P/'published-viewer.png']
    assert all(p.exists() and p.stat().st_size>10000 for p in proof)
    st.update(actual_viewer_checked=True,quality_accepted=False);save(P/'staged-deliveries.json',st)
    publication=read(P/'publication.json')
    publication.update(ui_verified=True,paired_review_complete=True,arms=st['arms'],
        proof=[dict(path=str(p),sha256=sha(p)) for p in proof],ppt=deck['source'],ppt_sha256=deck['final_sha256'])
    save(P/'publication.json',publication)
    status=read(B/'work-status.json');status.update(updated=datetime.datetime.now().astimezone().isoformat(),local_training_running=False,goal_accepted=False)
    status['latest_local_execution'].update(state='paired_fresh_training_and_review_complete_quality_not_accepted',running=False,local_standalone_ply_verified=True,report_UI_verified=True,ppt_updated=True)
    save(B/'work-status.json',status)
    registry=read(B/'gaussian-deliveries.json');registry['rounds']['093'].update(arms=receipts,local_standalone_open_verified=True,quality_accepted=False,pc00_opened_this_round=False);save(B/'gaussian-deliveries.json',registry)
    print(json.dumps(dict(round='093',bounded_review_complete=True,quality_accepted=False,goal_complete=False)))

if __name__=='__main__':main()
