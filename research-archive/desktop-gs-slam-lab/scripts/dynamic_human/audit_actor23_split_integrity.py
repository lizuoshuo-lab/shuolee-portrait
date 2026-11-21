"""Verify actual source isolation before any output-based judgement."""
from pathlib import Path
import json,hashlib
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106';read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
train=read(O/'training-input-audit.json')['records'];cal=read(O/'evaluation-time/calibration-input-audit.json')['records'];ev=read(O/'evaluation-protocol.json')['records'];extra=read(O/'continuity-protocol.json')['records'];keys=lambda rs:{(r['camera'],r['frame']) for r in rs}
assert len(train)==195 and len(keys(train))==195 and len(cal)==126 and len(keys(cal))==126 and len(ev)==264 and len(keys(ev))==264 and len(extra)==10
assert keys(cal).isdisjoint(keys(train)|keys(ev)|keys(extra))
assert all(r['camera'] not in ['00','06','12'] and r['frame']%2==0 for r in train)
assert keys(train)&keys(ev)==keys([r for r in ev if r['split']=='train_check']) and len(keys(train)&keys(ev))==9
assert keys(extra).issubset(keys(train)) and keys(extra).isdisjoint(keys(ev))
assert all(r['frame']%2==1 for r in ev if r['split'] in ['validation_time','validation_camera_time'])
checked=set()
for group in [train,cal,ev,extra]:
    for r in group:
        p=O/r['input'].replace('\\','/');assert sha(p)==r['input_sha256']
        if str(p) in checked:continue
        with np.load(p) as z:
            assert z['rgb'].shape==(1080,1920,3) and z['rgb'].dtype==np.uint8
            assert z['matte'].shape==(1080,1920) and z['valid'].dtype==bool
            assert z['K'].shape==(3,3) and z['w2c'].shape==(4,4)
        checked.add(str(p))
assert len(checked)==576
frozen=read(O/'comparison-protocol.json')
for n,digest in frozen['source_sha256'].items():assert sha(L/'scripts/dynamic_human'/n)==digest
report=dict(status='195training_126calibration_264evaluation_10supplementary_records_verified',unique_native_files=len(checked),training_eval_overlap='Exactly9explicit trainingcheck images; all255validation images are disjoint.',calibration_vs_appearance_overlap=0,supplementary='10existing training images used only to complete consecutive playback;not added to heldout metrics.',camera_scope='Other cameras00/06/12are calibration-visible at582..594only, appearance-heldout at600..624.',source_sha_verified=True,source_native_shape_verified=True,training_code_unchanged=True,goal_complete=False)
(O/'split-integrity.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report))
