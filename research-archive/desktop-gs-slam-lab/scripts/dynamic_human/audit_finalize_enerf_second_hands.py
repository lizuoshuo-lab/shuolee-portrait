"""Independently recompute source hashes and every saved projection residual."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

L=Path(__file__).resolve().parents[2];S=L/'outputs/dynamic-human-enerf-second-interval-079';O=L/'outputs/dynamic-human-enerf-hand-motion-082'
read=lambda p:json.loads(p.read_text(encoding='utf8'));save=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (O/'final-summary.json').exists()
obs=read(O/'observations.json')['records'];by={(r['camera'],r['frame']):r for r in obs};assert len(by)==195 and not {c for c,f in by}&{'00','06','12'}
offsets=read(O/'geometry-protocol.json')['offsets'];maxerr=0.
for r in obs:
    p=S/'native-input'/f"camera{r['camera']}-frame{r['frame']:06d}.npz";assert sha(p)==r['input_sha256']
    with np.load(p) as z:
        assert hashlib.sha256(z['rgb'].tobytes()).hexdigest()==r['rectified_sha256']
        assert np.array_equal(z['K'],r['K']) and np.array_equal(z['w2c'],r['w2c'])
    xy=np.array(r['all_crop_keypoints']);dv=np.linalg.norm(xy-xy[1],axis=-1).max(0)
    assert np.array_equal(xy[1],r['keypoints']);assert np.max(abs(dv-r['crop_deviation_px']))<1e-10
def interp(c,t,key):
    nominal=t-offsets[c];lo=2*int(np.floor(nominal/2));hi=lo+2;assert 720<=lo<hi<=732;f=(nominal-lo)/2
    return (1-f)*np.array(by[c,lo][key])+f*np.array(by[c,hi][key]),lo,hi
def projection(c,X):
    r=by[c,726];q=np.array(r['K'])@(np.array(r['w2c'])[:3,:3]@X+np.array(r['w2c'])[:3,3]);assert q[2]>0;return q[:2]/q[2]
geo=read(O/'cross-camera-records.json')['records'];assert len(geo)==4725
for r in geo:
    if not r['valid']:continue
    c=r['target_camera'];assert c not in r['support_cameras'] and len(r['support_cameras'])>=4
    key='wholebody_keypoints' if r['method']=='wholebody' else 'keypoints';X=np.array(r['point']);uv=projection(c,X);target,_,_=interp(c,r['time'],key);e=float(np.linalg.norm(uv-target[r['joint']]))
    maxerr=max(maxerr,abs(e-r['error_px']))
    for support in r['support_cameras']:
        xy,lo,hi=interp(support,r['time'],key);assert np.linalg.norm(projection(support,X)-xy[r['joint']])<3+1e-7
        if r['method']=='hand_stable_crop':assert max(by[support,lo]['crop_deviation_px'][r['joint']],by[support,hi]['crop_deviation_px'][r['joint']])<=3
coef=read(O/'trajectory-fit.json')['coefficients'];tr=read(O/'trajectory-records.json')['records']
for r in tr:
    o=by[r['camera'],r['frame']];tau=(r['frame']+offsets[r['camera']]-726)/6;X=(tau**np.arange(4))@np.array(coef[f"{r['method']}/{r['joint']}"]);uv=projection(r['camera'],X)
    target=np.array(o['wholebody_keypoints' if r['method']=='wholebody' else 'keypoints'])[r['joint']];e=float(np.linalg.norm(uv-target));maxerr=max(maxerr,abs(e-r['error_px']),float(np.max(abs(uv-r['xy']))))
palm=read(O/'palm-summary.json');pr=read(O/'palm-records.json')['records'];pars=palm['parameters'];ref=np.array(pars['reference']);local=ref-ref.mean(0);joints=[0,5,9,13,17]
for r in pr:
    o=by[r['camera'],r['frame']];tau=(r['frame']+offsets[r['camera']]-726)/6;basis=tau**np.arange(4);rot=Rotation.from_rotvec(basis[1:]@np.array(pars['rotation_coefficients'])).as_matrix();T=basis@np.array(pars['translation_coefficients']);X=rot@local[joints.index(r['joint'])]+T;uv=projection(r['camera'],X);e=float(np.linalg.norm(uv-np.array(o['keypoints'])[r['joint']]));maxerr=max(maxerr,abs(e-r['error_px']),float(np.max(abs(uv-r['xy']))))
    if r['used_for_fit']:assert r['camera']!='04' and r['frame']%2==0 and o['crop_deviation_px'][r['joint']]<=3 and r['old_error_px']<6
assert maxerr<1e-8
files=[f'source-panels/camera{c}-source.png' for c in sorted({r['camera'] for r in obs})]+[f'candidate-panels/camera{c}-candidates.png' for c in sorted({r['camera'] for r in obs})]+[f'motion-panels/camera{c}-motion.png' for c in ['01','04','08','09','13']]+[f'motion-panels/camera{c}-palm.png' for c in ['04','09','13']]
pngs=[]
for f in files:
    with Image.open(O/f) as im:im.load();pngs.append(dict(path=f,sha256=sha(O/f),size=list(im.size)))
assert len(files)==38
save(O/'numerical-audit.json',dict(status='195native_inputs_and_all_saved_projection_errors_independently_recomputed',native_inputs=195,model_crops=585,geometry_rows=len(geo),trajectory_rows=len(tr),palm_rows=len(pr),max_error_difference=maxerr,decoded_panels=38,public_model_exports=False))
save(O/'visual-review.json',dict(status='38actual_native_source_and_prediction_panels_reviewed',files=pngs,
    source='15sheets90rawhandcropsat720726732; all lightpersonright hands observable in part, lefttoygrip deliberately excluded.',
    candidates='15sheets45anchors; dedicated predictions change when crop changes during grasp. Opening palm at732 more stable, hidden fingertips not verified.',
    motion='5sheets25oddtime crops, eachwithnative/detection/three3Dpredictionvariants. Unconstrained joints occasionally project offhand, wrist/palm partial alignment doesnotvalidateallfingers.',
    palm='3sheets15oddtime crops. Fixedshape avoidsbone stretching by construction but palm projection remains displaced, especially04at721723727. No RGB reconstruction produced.',
    accept_as_finger_labels=False,accept_palm_as_Gaussian_constraint=False,goal_complete=False))
g=read(O/'geometry-summary.json');t=read(O/'trajectory-summary.json')
out=dict(status='native_hand_observations_and_motion_qualification_complete_not_accepted',scene='ENeRF-Outdoor actor5_6',frames=[720,732],person='Light-clothed person right hand only',images=195,model_crops=585,actual_panels_reviewed=38,
    geometry=g['methods'],trajectory_joint_counts=t['accepted_joint_counts'],rigid_palm=palm['metrics'],
    decision_zh='局部检测的手腕跨机位一致性改善，但指尖一致性退化。裁剪稳定性筛选留下的轨迹只有9个关节，不能组成完整手部。五点刚性手掌在排除机位04后的投影误差也未改善。未将这些预测加入高斯训练，人物清晰度仍不通过。',
    no_new_Gaussian_training=True,adopt_as_development_baseline=False,goal_complete=False,
    next_action='Stop promoting RTMHand predictions into finger constraints. Prioritize calibrated per-view local photometric correspondence/visibility in the opening-hand interval and evaluate an actual reconstruction intervention. Preserve singleframe failures and no-hidden-hand-label policy.',
    note='Detector consistency is not 3D/2D GT accuracy, and landmark overlay is not a rendered human.',updated_at=datetime.now(timezone.utc).isoformat())
save(O/'final-summary.json',out)
p=L/'manifests/dynamic-human-work.json';m=read(p);m.update(status='hand_observation_and_motion_candidates_rejected_clarity_unresolved',outdoor_hand_motion='outputs/'+O.name+'/final-summary.json',current_runs=[],next_action=out['next_action'],goal_turn_classification='Progress:195native handobservations585crops;4725independentcamera checks;three3Dtrajectoryfits andfixedshape5pointpalm control;38realpanelsreviewed; allresiduals independentlyrecomputed. No accepted Gaussian or clarity improvement.',updated_at=out['updated_at']);m['completed'].append('082: native hand195images585crops,4725crosscamera hypotheses,3trajectory variants andrigidpalm control;38nativepanelsreviewed. Wristconsistency improves buttipsregress; labels andpalmmotion notadopted; nonewGaussiantraining.');save(p,m)
shutil.copy2(__file__,O/Path(__file__).name);print('HAND_082_AUDIT_AND_DECISION_SAVED',len(tr),len(pr),maxerr)
