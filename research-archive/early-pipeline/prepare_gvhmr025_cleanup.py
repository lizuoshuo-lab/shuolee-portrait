from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np

build=Path(__file__).resolve().parent
base=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee')
r24=base/'current/smplx-gvhmr-body-motion-fresh-024'
r25=base/'current/smplx-gvhmr-shared-world-fresh-025'
out=build.parent/'shuolee-slide-output'
canonical=base/'reports/02_有限视角重建优化路线.pptx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
now=datetime.now().astimezone().isoformat(timespec='seconds')
read=lambda p:json.loads(p.read_text('utf-8-sig'))
save=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf-8')
assert read(r24/'phase-conclusion.json')['body_geometry_accepted'] is False
assert read(r25/'phase-conclusion.json')['body_geometry_accepted'] is False
receipt=read(build/'gvhmr025-deck-receipt.json')
assert receipt['canonical_replaced'] and sha(canonical)==receipt['final_sha256']
for root,arms in [(r24,['control','body-texture']),(r25,['free','shared'])]:
    for arm in arms:
        file=root/arm/'run-state.json';state=read(file)
        assert state['running'] is False and state['step']==1200
        state.update(state='reviewed_not_accepted',reviewed_at=now,accepted=False,
                     review_scope='Selected actual source overlays and fixed-world renders. Not all 59 images or full continuous video accepted.')
        save(file,state)

protected=[canonical,base.parent/'viewer/body-smplx.html',
    base/'current/smplx-gvhmr-shoe-motion-fresh-023/fitted.npz',
    r25/'shared/fitted.npz',r25/'shared/initialization.npz',r25/'shared/starting-state.npz',
    Path('D:/GS-SLAM-Data/models/vposer/V02_05/snapshots/V02_05_epoch=08_val_loss=0.03.ckpt'),
    build/'gvhmr025-final-renders/55.png',build/'gvhmr025-page-proof.png']
for p in protected:assert p.is_file(),p
targets=[]
for root,arms in [(r24,['control','body-texture']),(r25,['free'])]:
    for arm in arms:
        for name in ['fitted.npz','initialization.npz','starting-state.npz']:
            p=root/arm/name
            if p.is_file():targets.append((p,'Unadopted body fit or duplicate raw initialization; compact metrics, sources and hashes retained.'))
    keep={'source-000023.jpg','source-000045.jpg','texture-45.jpg','fixed-front.jpg','fixed-left90.jpg','fixed-left90-time-sheet.jpg'}
    if root==r25:keep.update({'source-000000.jpg','source-000011.jpg','source-000058.jpg'})
    for p in (root/'review').iterdir():
        if p.suffix.lower() in {'.jpg','.mp4'} and p.name not in keep:
            targets.append((p,'Redundant reviewed-run render; selected real comparisons and publication assets retained.'))
for name in ['gvhmr025-candidate.pptx','gvhmr025-candidate-r2.pptx',
             'gvhmr025-candidate.pptx.inspect.ndjson','gvhmr025-candidate-r2.pptx.inspect.ndjson',
             'gvhmr025-slide-before.json']:
    p=build/name
    if p.is_file():targets.append((p,'Private draft or intermediate; canonical report validated and retained.'))
for name in ['02_有限视角重建优化路线_20261023_运动与几何核验.pptx','02_有限视角重建优化路线_20261023_运动与几何核验_定稿.pptx']:
    p=out/name
    if p.is_file():targets.append((p,'Duplicate validated export; one canonical report retained.'))
for i in range(1,55):
    p=build/'gvhmr025-final-renders'/f'{i:02d}.png'
    if p.is_file():targets.append((p,'Unchanged slide pixels verified against protected earlier renders.'))
rows=[]
for p,reason in targets:
    assert p not in protected
    row=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),reason=reason)
    if p.suffix=='.npz':
        with np.load(p,allow_pickle=False) as arrays:
            row['arrays']={k:dict(shape=list(arrays[k].shape),dtype=str(arrays[k].dtype),
                                sha256=hashlib.sha256(arrays[k].tobytes()).hexdigest()) for k in arrays.files}
    rows.append(row)
manifest=dict(created=now,state='prepared',allowed_roots=[str(p) for p in [r24,r25,build,out]],
    scope='Only fresh phases024/025 and this publication drafts. No earlier blocked deletion targets.',
    retained_control='Phase025 shared fit is a read-only observation diagnostic control, not an accepted body or a future training initialization.',
    protected=[dict(path=str(p),sha256=sha(p)) for p in protected],
    files=rows,count=len(rows),bytes=sum(r['bytes'] for r in rows))
save(r25/'cleanup-receipt.json',manifest)
print(json.dumps({k:manifest[k] for k in ['state','count','bytes','retained_control']},ensure_ascii=False))
