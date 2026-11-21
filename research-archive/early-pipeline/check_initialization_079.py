from pathlib import Path
import json,hashlib,datetime
import numpy as np
B=Path('C:/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
R=B/'person-static-surface-consistency-fresh-079';A=B/'person-static-observed-correspondence-fresh-077'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(R/'clothing-init/audit.json');assert audit['points']==200000 and audit['from_scratch'] and not audit['old_gaussian_loaded']
points=np.load(R/'clothing-init/points.npz')['xyz'];old=np.load(A/'clothing-init/points.npz')['xyz']
assert np.array_equal(points,old)
assert sha(R/'data/training-inputs.json')==sha(A/'data/training-inputs.json')
fitlog=(R/'fit.log').read_text();assert '"step": 1200' in fitlog
result=dict(created=datetime.datetime.now().astimezone().isoformat(),actually_inspected=True,
    file=str(R/'clothing-init/initialization-review.jpg'),sha256=sha(R/'clothing-init/initialization-review.jpg'),
    views=[1,34,70,100,136,175],visible='Coarse clothed silhouette with head, torso, arms and two legs; no catastrophic initialization loss.',
    observations='Initial gray hull has grid texture and thin feet/hand support; initialization is not a finished reconstruction.',
    initial_arrays_equal_077=True,body_fit_recomputed_this_round=True,initialization_recomputed_this_round=True,
    camera_manifest_equal_077=True,training_allowed=True,quality_accepted=False)
(R/'initialization-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
