"""CPU-only replay of original prior terms on the saved failed pose."""
import importlib.util,inspect,os,sys
from pathlib import Path
import numpy as np
import torch
import official_smplifyx_entry123 as entry
from run_official_gaussians123 import read,save,sha,now
torch.set_num_threads(2)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
base=Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123')
official=base/'smplifyx-reference-control/official';root=base/'cross-subject-sh0-124/grass'
receipt=read(base/'smplifyx-reference-control/official-first-cycle/full94-shared-shape/frames/frame_000032/result.json')
hashes={k:sha(official/k)==v for k,v in receipt['source_hashes'].items()};assert all(hashes.values())
entry.adapted_entry_source(official)
prior=entry.module(official/'smplifyx/prior.py','_prior124_readonly')
cfg=entry.load_official_config(official)
with np.load(root/'fit/fitted.npz',allow_pickle=False) as z:
    i=list(z['frame_ids']).index('frame_000032');latent=torch.tensor(z['pose_embedding'][i:i+1],requires_grad=True)
    aa=torch.tensor(z['body_pose_axis_angle'][i:i+1],requires_grad=True)
    elbow=z['observed_native_keypoints'][i,[2,3,4],2].tolist()
p=prior.SMPLifyAnglePrior();terms=p(aa);raw_latent=latent.square().sum();raw_angle=terms.sum()
w=cfg['body_pose_prior_weights'][-1];body=raw_latent*w*w;bending=raw_angle*(3.17*w)
body.backward();bending.backward()
assert torch.isfinite(latent.grad).all() and torch.isfinite(aa.grad).all()
assert latent.grad.abs().max()>0 and aa.grad.abs().max()>0
try:collision_module=importlib.util.find_spec('mesh_intersection') is not None
except (ModuleNotFoundError,ValueError):collision_module=False
result=dict(updated=now(),mode='CPU read-only saved-pose term replay; no fitting/parameter update',frame='frame_000032',
    pinned_source_hashes_match=hashes,adapter_sha256=sha(entry.__file__),original_interpenetration=cfg['interpenetration'],actual_interpenetration=receipt['interpenetration'],
    effective_body_prior_weights=cfg['body_pose_prior_weights'],effective_bending_weights=[3.17*v for v in cfg['body_pose_prior_weights']],
    maxiters=cfg['maxiters'],lr=cfg['lr'],ftol=cfg['ftol'],gtol=cfg['gtol'],optim_type=cfg['optim_type'],
    original_angle_selected_indices_without_root=(p.angle_prior_idxs-3).tolist(),angle_terms_raw=terms.detach().tolist(),
    final_stage_body_prior=float(body.detach()),final_stage_bending_prior=float(bending.detach()),
    latent_gradient_max=float(latent.grad.abs().max()),angle_gradient_max=float(aa.grad.abs().max()),
    shoulder_elbow_wrist_confidence=elbow,has_nonzero_limb_prior=True,has_hard_joint_angle_limits=False,
    temporal_term_in_original_single_frame=False,independent_3d_joint_term_used=False,
    collision_python_module_present=collision_module,
    differences=['Full measured/conditional K replaces fixed5000 and centered principal point','RTMW observations and relative/flip confidence replace OpenPose outputs','Existing VPoser V02_05 with original-model API adapter, not a matched upstream example weight reproduction','Shared reference betas frozen for per-frame pose fits','interpenetration=False disables repo collision penalty'],
    caution='Collision absence changes the objective; it is not a harmless runtime setting. Collision alone does not require an arm to be in front of the torso.',
    training_started=False,quality_accepted=False)
save(root/'pose-cause-audit/smplify-upstream-constraints.json',result)
print(result)
