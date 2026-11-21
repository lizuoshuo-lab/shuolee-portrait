from difflib import unified_diff
from pathlib import Path
import json
import numpy as np
from shuolee_portrait import official_sequence as seq, official_smplifyx as official, stable_selection as sel, runtime as rt

base = Path('/opt/gs-slam-lab/shuolee-smpl-placement-006')
source = base/'run'
output = Path('/opt/gs-slam-lab/shuolee-smpl-placement-006-neighbor-control')
cfg = rt.read(source/'config.json')
deployment = rt.read(base/'deployment.json')
upstream = Path(deployment['official_smplifyx']['source_root'])
assets = Path(deployment['assets'])
anchors = rt.read(source/'fit/shared-body.json')
beta = np.asarray(anchors['betas'], dtype=np.float32)
selection = rt.read(source/'fit/selection.json')
pairs = [('frame_000009', 'frame_000012'), ('frame_000045','frame_000044')]
output.mkdir(exist_ok=False)
pins = {str(source/name):rt.sha(source/name) for name in ('data/inputs.json','data/observations.npz','fit/shared-body.json','fit/selection.json','config.json')}
for pair in pairs:
    for frame in pair:
        for name in ('result.json','winner-camera-space.npz'):
            file=source/'fit/frames'/frame/name
            pins[str(file)]=rt.sha(file)
rt.save(output/'preregistration.json', dict(updated=rt.now(),candidate_pairs=pairs,input_pins=pins,
    source_visual_support=str(base/'source-side-neighbors.jpg'),
    sole_change='Seed the VPoser body stage latent from the closest adjacent accepted source-bin frame. No donor root, K, translation, shape, face or hand parameters.',
    bound='Two independent CPU target solves, same original selection screen. Both target frames remain excluded from the delivered model.',
    driver_sha256=rt.sha(Path(__file__))))
inputs = seq.load_inputs(source/'data/inputs.json',source/'data/observations.npz',
    camera_length_contract=cfg['camera_length_contract'],reference_frame=selection['reference_frame'],capture_layout=cfg['capture_layout'])
seq._init_cpu_fit(upstream,assets)
_,body,vposer,device=seq._CPU_FIT
shared_pose=official.decode_shared_body_pose(vposer,anchors['pose_latent'],device=device)
original=official.adapted_entry_source
winners=[]
for target, donor in pairs:
    assert target not in selection['retained_holdouts'] and donor in selection['selected_frames']
    assert selection['reasons'][target]['status']=='pose_screen_rejection'
    assert abs(selection['reasons'][target]['source_bin']-selection['reasons'][donor]['source_bin'])==1
    def frozen(f):
        with np.load(source/'fit/frames'/f/'winner-camera-space.npz',allow_pickle=False) as data:
            return dict(data)
    old=frozen(target)
    pose_seed=frozen(donor)['pose_embedding'].astype(np.float32)
    assert pose_seed.shape==(1,32) and np.isfinite(pose_seed).all()
    assert np.array_equal(beta,old['betas'])
    i=inputs['ids'].index(target)
    row=inputs['rows'][i]
    def adapter(upstream, *, refine_body_placement=False):
        text,diff=original(upstream,refine_body_placement=refine_body_placement)
        assert refine_body_placement and text.count('pose_embedding.fill_(0)')==1
        update=text.replace('pose_embedding.fill_(0)','pose_embedding.copy_(pose_embedding.new_tensor('+repr(pose_seed.tolist())+'))')
        compile(update,'adjacent-body-latent-only','exec')
        extra=''.join(unified_diff(text.splitlines(True),update.splitlines(True),fromfile='unchanged-placement',tofile='neighbor-latent'))
        return update,diff+'\n'+extra
    official.adapted_entry_source=adapter
    try:
        new, receipt=seq._frame_fit((row,output/target,inputs['targets'][i:i+1],inputs['K'][i],beta,True,shared_pose),upstream,body,vposer,device)
    finally:
        official.adapted_entry_source=original
    rep=rt.read(receipt['path'])
    prior=rt.read(source/'fit/frames'/target/'result.json')
    assert rep['body_placement_initial_m']==prior['body_placement_initial_m']
    assert rep['shared_body_target_sha256']==prior['shared_body_target_sha256']
    obs=inputs['targets'][i]
    def scores(value,report):
        score,flags,_,_=sel._candidate(row,value,obs,inputs['K'][i],beta,None,report,True)
        errors=np.linalg.norm(value['joints_projected_native'][0]-obs[:,:2],axis=1)
        box=row['initial_box_xyxy']; h=box[3]-box[1]
        return dict(flags=flags,score=score,
            max_body_pct=float(100*errors[sel.BODY[obs[sel.BODY,2]**2>0]].max()/h),
            max_feet_pct=float(100*errors[sel.FEET[obs[sel.FEET,2]>0]].max()/h))
    report=dict(target=target, donor=donor,before=scores(old,prior),after=scores(new,rep),fit_receipt=receipt)
    winners.append(report)
    print(json.dumps(report),flush=True)
assert all(rt.sha(file)==digest for file,digest in pins.items())
rt.save(output/'result.json',dict(state='completed',rows=winners,production_changed=False,quality_accepted=False))

