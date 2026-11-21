"""Read saved boundary projections and the official kinematic tree; no fit."""
import os
import time
import numpy as np
import control_sofa_local_joint124 as control

c=control.c
OUT=control.OUT/'regional-motion-audit.json'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
start=time.monotonic();row,target=c.base.target()
result=c.read(control.OUT/'result.json')
arrays=c.a.load_npz(control.OUT/'winner-camera-space.npz')
trajectory=c.a.load_npz(control.OUT/'trajectory.npz')
stage=next(s for s in result['stages'] if s['kind']=='body' and s['stage']==0)
prefix=trajectory[f"state_{stage['after_state']:03d}_joints_projected_native"]
records=[]
for side,body_index,hand_index in [('left',7,25),('right',4,46)]:
    records.append(dict(side=side,body_index=body_index,hand_index=hand_index,
        body_hand_observation_distance_px=float(np.linalg.norm(target[0,body_index,:2]-target[0,hand_index,:2])),
        confidences=[float(target[0,index,2]) for index in (body_index,hand_index)],
        model_same_wrist_projection_difference_px=float(np.linalg.norm(arrays['joints_projected_native'][0,body_index]-arrays['joints_projected_native'][0,hand_index])),
        wrist_projection_change_during_local_stage_px=float(np.linalg.norm(arrays['joints_projected_native'][0,body_index]-prefix[0,body_index])),
        body_target_distance_px=float(np.linalg.norm(arrays['joints_projected_native'][0,body_index]-target[0,body_index,:2])),
        hand_root_target_distance_px=float(np.linalg.norm(arrays['joints_projected_native'][0,hand_index]-target[0,hand_index,:2]))))
with np.load(c.a.ASSETS/'smplx/SMPLX_NEUTRAL.npz',allow_pickle=True) as z:
    parents=z['kintree_table'][0].astype(np.int64)
from smplx.joint_names import JOINT_NAMES
def ancestors(index):
    chain=[index]
    while chain[-1]!=0:
        chain.append(int(parents[chain[-1]]));assert len(chain)<56
    return list(reversed(chain))
upper=[i for i in range(1,22) if 3 in ancestors(i)]
lower=[1,2,4,5,7,8,10,11]
assert set(upper).isdisjoint(lower)
value=dict(updated=c.now(),elapsed_seconds=time.monotonic()-start,optimizer_steps=0,
    source_result_sha256=c.sha(control.OUT/'result.json'),source_npz_sha256=c.sha(control.OUT/'winner-camera-space.npz'),
    wrist_records=records,
    upper_body_subtree=dict(root=3,root_name=JOINT_NAMES[3],body_joint_ids=upper,names=[JOINT_NAMES[i] for i in upper]),
    protected_lower_joint_ids=lower,paths={JOINT_NAMES[i]:ancestors(i) for i in (12,15,20,21)},
    explanation='A wrist rotation changes hand orientation but cannot translate its own skeletal joint center. The saved local stage leaves both wrist centers unchanged. Fitting hand-root position requires upstream arm/torso motion or consistent observations; this audit does not choose a new training configuration.',
    limits=['No assertion that2D detector pairs identify the exact same physical skin point.',
            'No changed observations, weights, optimizer, fitted pose or dataset selection.',
            'An upper-body subtree would add39 local angles outside the VPoser manifold; this design is untested and not admitted.'],
    quality_accepted=False,gaussian_admitted=False)
c.save(OUT,value);print(value,flush=True)
