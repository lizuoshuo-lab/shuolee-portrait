"""Verify original data identity and parameter scope after the completed controls."""
from pathlib import Path
import hashlib,json
import numpy as np
base=Path('/mnt/d/Projects/Shuolee-analysis-20261028')
run=Path('/opt/gs-slam-lab/smpl-x-alignment-20261028/grayT-7b1c0fd30822c3de-r2')
trial=Path('/opt/gs-slam-lab/smpl-x-alignment-20261029/gray-dynamic-face-fan')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
protocol=read(trial/'protocol.json');camera=read(run/'gaussian-run/training-cameras.json');init=read(run/'clothing-init/audit.json')
checks=dict(cameras_unchanged=sha(run/'data/training-inputs.json')==protocol['cameras_sha256']==camera['source_sha256'],
    observations_unchanged=sha(run/'data/observations.npz')==protocol['observation_sha256'],
    untrained_initial_points_unchanged=sha(run/'clothing-init/points.npz')==protocol['initialization_sha256'],
    original_fit_unchanged=sha(run/'fit/fitted.npz')==init['fit_sha256'])
before=np.load(run/'fit/fitted.npz');after=np.load(base/'dynamic-face-fan-control/fitted.npz')
mask=np.ones(21,bool);mask[[11,14]]=False
delta=float(abs(before['body_pose'].reshape(-1,21,3)[:,mask]-after['body_pose'].reshape(-1,21,3)[:,mask]).max())
checks['body_joint_parameters_except_neck_head_unchanged']=delta<1e-5
checks['full_original_frame_replay_verified']=read(base/'dynamic-face-fan-all-views/report.json')['frames']==95
assert all(checks.values()),checks
report=dict(checks=checks,maximum_nonhead_body_pose_delta=delta,passed=True,quality_accepted=False)
(base/'source-invariants.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
