"""Exercise the exact round-eleven face guard on a small protected prior-fit sample."""

import fcntl
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, '/mnt/d/Projects/Shuolee-smpl-face-guard-20261111/src')
from shuolee_portrait import official_sequence as seq, official_smplifyx as official, runtime as rt
from shuolee_portrait.joint_body_fit import refine_joint_body

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round11 as launch

lab = launch.cold.LAB
asset = launch.cold.ASSETS
official_root = launch.cold.OFFICIAL
prior = lab/'shuolee-smpl-delivery-round10-016/run'
layout = launch.read(prior.parent/'camera.json')['capture_layout']
rt.configure(asset, prior)
native = seq.load_inputs(prior/'data/inputs.json', prior/'data/observations.npz',
    reference_frame='frame_000027', capture_layout=layout,
    camera_length_contract=dict(translation_unit='meters', meters_per_camera_unit=1.,
        basis='model_conditioned', evidence='Original explicitly declared identity native cameras.'))
frames = ['frame_000001', 'frame_000005', 'frame_000010', 'frame_000025', 'frame_000027',
          'frame_000041', 'frame_000048', 'frame_000051', 'frame_000055']
scope = native['scope']
launch.need(set(frames)<=set(scope['fit_frame_ids']) and set(frames).isdisjoint(scope['retained_holdouts']),
            'Engineering check cannot use holdouts.')
native['scope'] = dict(scope, fit_frame_ids=frames)
winners = {}
for frame in frames:
    with np.load(prior/'fit/frames'/frame/'winner-camera-space.npz', allow_pickle=False) as arrays:
        winners[frame] = dict(arrays)
shared = launch.read(prior/'fit/shared-body.json')
start = time.monotonic()
with launch.cold.LEASE.open('a+') as lease:
    fcntl.flock(lease, fcntl.LOCK_EX|fcntl.LOCK_NB)
    body, vposer, _ = official.create_models(official_root=official_root, assets=asset, device='cpu', static_topology=False)
    improved, proof = refine_joint_body(inputs=native, winners=winners, body=body, vposer=vposer,
        shared_betas=np.asarray(shared['betas'], np.float32), shared_latent=np.asarray(shared['pose_latent'], np.float32))
    from shuolee_portrait.training import _frozen_smpl_surface
    selected = seq._selection_inputs(native, frames, 'frame_000027')
    arrays, _ = seq.assemble(selected, improved, np.asarray(shared['betas'], np.float32), seq.pelvis_basis(body))
    _, _, _, _, replay_error = _frozen_smpl_surface(arrays, asset/'smplx/SMPLX_NEUTRAL.npz')
launch.need(set(improved)==set(frames) and proof['state']=='complete'
            and all(np.array_equal(improved[id]['betas'],winners[id]['betas']) for id in frames)
            and all(np.array_equal(improved[id]['K_integer_centers'],winners[id]['K_integer_centers']) for id in frames),
            'Joint fitter changed engineering check identities.')
print(dict(state=proof['state'], subject='016', source='fixed R10 initializers for engineering check',
    fit_frames=len(improved), holdouts_used=0, elapsed_seconds=time.monotonic()-start,
    initial=proof.get('objective_initial'), final=proof.get('objective_final'),
    config=proof.get('config'), preflight=proof.get('preflight'), world_mesh_replay_max_abs_m=replay_error,
    face_protection=proof.get('face_protection'), checkpoint=proof.get('objective_final_step')),flush=True)
