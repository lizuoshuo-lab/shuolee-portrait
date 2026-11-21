"""Reserve complete consecutive sequences using existing sources only."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
import numpy as np
L=Path(__file__).resolve().parents[2];O=L/'outputs/dynamic-human-actor23-independent-106'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'continuity-protocol.json').exists()
assert not any((O/b/'native-evaluation/metrics.json').exists() for b in ['static','dynamic_nominal','dynamic_clock'])
main=read(O/'evaluation-protocol.json');known={(r['camera'],r['frame']):r for r in main['records']};train={(r['camera'],r['frame']):r for r in read(O/'training-input-audit.json')['records']};extra=[];sequences={}
for c in ['00','13']:
    rows=[];boxes=[]
    for f in range(600,625):
        if (c,f) in known:r=dict(known[c,f],prediction_directory='native-evaluation')
        else:
            old=train[c,f];path=old['input'].replace('\\','/');assert sha(O/path)==old['input_sha256'];z=np.load(O/path)
            r=dict(camera=c,frame=f,split='train_continuity',input=path,input_sha256=old['input_sha256'],bbox=z['bbox'].tolist(),prediction_directory='continuity-extra');extra.append(r)
        rows.append(r);boxes.append(r['bbox'])
    b=np.array(boxes);union=[int(b[:,0].min()),int(b[:,1].min()),int(b[:,2].max()),int(b[:,3].max())]
    sequences[c]=dict(records=rows,full_person_union_bbox=union)
assert len(extra)==10
save(O/'continuity-protocol.json',dict(status='frozen_before_candidate_rendering',recorded_utc=datetime.now(timezone.utc).isoformat(),records=extra,sequences=sequences,evaluation_clock_sha256=sha(O/'evaluation-clock-decision.json'),review_fps=5,scope='00:25appearance-heldoutframes.13:13trainingframesplus12heldoutoddframes. All consecutive indices600..624;5fpsslowreviewdoesnotassertcaptureFPS. Sources already prepared, no new image access or training.',goal_complete=False));print('Frozen 2x25 consecutive native views;10additional known-camera training views only.')
