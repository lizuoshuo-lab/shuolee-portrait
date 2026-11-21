"""Reject locally ambiguous image matches without consulting model residuals."""
from pathlib import Path
import json, hashlib, shutil
import numpy as np, cv2

L=Path(__file__).resolve().parents[2];R=L/'outputs/dynamic-human-enerf-native-motion-tracks-075';I=L/'outputs/dynamic-human-enerf-articulated-control-064'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=R/'unique-patches';out.mkdir(exist_ok=True)
assert not (out/'summary.json').exists()
save(out/'protocol.json',dict(status='development_revision_fixed_after_seeing_edge_sliding_counterexample_before_uniqueness_scores',
    trigger='Largest075 disagreement lies on a forearm boundary; two 11px patches can correlate strongly while sliding along an edge. FB consistency and high NCC alone do not certify a material correspondence.',
    method='OpenCV TM_CCOEFF_NORMED. Reference11x11 subpixel grayscale patch; target91x91 subpixel crop centered at LK endpoint, yielding81x81 score surface (+/-40px).',
    thresholds=dict(min_peak_score=.97,max_peak_distance_from_LK_px=1.5,min_peak_gap=.02,second_peak_exclusion_radius_px=3),
    criterion='All previous075 gates plus unique dominant local NCC peak. Neither model projected point nor endpoint residual enters filtering.',
    limits='Translation template only; rotating/deforming genuine patches may be rejected. Local uniqueness does not certify cross-camera identity, occlusion, or finger labels.',
    training_labels_accepted=False,goal_complete=False))
cv2.setNumThreads(4);records=[]
for c in read(R/'protocol.json')['cameras']:
    tr=dict(np.load(R/'tracks'/f'camera{c}.npz'))
    gray=cv2.cvtColor(np.load(I/'native-input'/f'camera{c}-frame000750.npz')['rgb'],cv2.COLOR_RGB2GRAY)
    shape=tr['accepted'].shape;passed=np.zeros(shape,bool);peak=np.full(shape,np.nan,np.float32);gap=peak.copy();shift=peak.copy()
    for j,t in enumerate(tr['target_frame']):
        target=cv2.cvtColor(np.load(I/'native-input'/f'camera{c}-frame{t:06d}.npz')['rgb'],cv2.COLOR_RGB2GRAY)
        for k in np.flatnonzero(tr['accepted'][j]):
            p=cv2.getRectSubPix(gray,(11,11),tuple(map(float,tr['source_xy'][k])))
            q=cv2.getRectSubPix(target,(91,91),tuple(map(float,tr['target_xy'][j,k])))
            scores=cv2.matchTemplate(q,p,cv2.TM_CCOEFF_NORMED)
            _,best,_,loc=cv2.minMaxLoc(scores);x,y=loc
            yy,xx=np.mgrid[:81,:81];d=(xx-x)**2+(yy-y)**2
            second=float(scores[d>9].max());delta=float(np.hypot(x-40,y-40))
            peak[j,k]=best;gap[j,k]=best-second;shift[j,k]=delta
            passed[j,k]=best>=.97 and delta<=1.5 and best-second>=.02
    np.savez_compressed(out/f'camera{c}.npz',accepted=passed,peak=peak,peak_gap=gap,peak_distance=shift)
    record=dict(camera=c,input_accepted=int(tr['accepted'].sum()),unique=int(passed.sum()),
        right_hand_box_unique=int((passed&(tr['region'][None]=='right_hand_box')).sum()),
        observations_per_frame={str(t):int(passed[j].sum()) for j,t in enumerate(tr['target_frame'])},sha256=sha(out/f'camera{c}.npz'))
    records.append(record);print(json.dumps(record),flush=True)
w=read(R/'motion-summary.json')['worst_model_disagreement'];tr=np.load(R/'tracks'/f"camera{w['camera']}.npz");q=np.load(out/f"camera{w['camera']}.npz");j=list(tr['target_frame']).index(w['frame']);k=w['feature']
summary=dict(status='local_patch_uniqueness_candidates_ready_visual_review_pending',records=records,
    initial_observations=sum(x['input_accepted'] for x in records),unique_observations=sum(x['unique'] for x in records),
    right_hand_box_unique_observations=sum(x['right_hand_box_unique'] for x in records),
    edge_counterexample=dict(camera=w['camera'],frame=w['frame'],feature=k,accepted=bool(q['accepted'][j,k]),peak=float(q['peak'][j,k]),gap=float(q['peak_gap'][j,k]),peak_shift=float(q['peak_distance'][j,k])),
    model_updates=0,training_labels_accepted=False,goal_complete=False)
save(out/'summary.json',summary);shutil.copy2(__file__,out/Path(__file__).name)
print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
