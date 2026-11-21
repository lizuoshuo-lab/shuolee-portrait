"""Persist an actual review of an existing snapshot; no training decision or mutation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, hashlib

p=argparse.ArgumentParser()
p.add_argument('--subject',required=True)
p.add_argument('--step',type=int,required=True)
p.add_argument('--summary',required=True)
a=p.parse_args()
rec=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
folder=rec/f'{a.subject}-train-{a.step:06d}-review'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
idx=read(folder/'index.json')
assert idx['subject']==a.subject and idx['step']==a.step
assert len([x for x in idx['images'] if x['region']=='body'])==6
for item in idx['images']:
    assert sha(folder/item['file'])==item['sha256']
    assert item['same_crop'] and item['aspect_preserved']
result=dict(updated=datetime.now(timezone.utc).isoformat(),subject=a.subject,step=a.step,
    actual_visual_review=True,index_sha256=sha(folder/'index.json'),
    reviewed_images=[dict(file=x['file'],sha256=x['sha256']) for x in idx['images']],
    quality_accepted=False,formal_calibration_accepted=False,
    training_or_camera_changed=False,summary=a.summary)
save(folder/'actual-review.json',result)
ledger=read(rec/'evidence-ledger.json')
kind=f'{a.subject}_actual_{a.step:06d}step_review'
assert not any(x.get('kind')==kind for x in ledger['entries']), 'Review already recorded; do not repeat.'
ledger['entries'].append(dict(kind=kind,date=result['updated'],summary=a.summary,
    quality_accepted=False,review=str(folder/'actual-review.json')))
save(rec/'evidence-ledger.json',ledger)
print(json.dumps(result,ensure_ascii=False))
