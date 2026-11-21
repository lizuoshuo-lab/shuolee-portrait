"""Portable full-roster SMPL-X fitting with one identity and frozen cameras."""
from pathlib import Path
import json
import time
import numpy as np

from . import runtime as rt


def export_snapshot(model, inputs, output):
    import torch
    vc, vw, joints, pixels = [], [], [], []
    with torch.no_grad():
        for i in range(len(model.frame_ids)):
            state = model.forward_state(frame_indices=[i])
            uv = model.project(state, frame_indices=[i])[0].cpu().numpy()
            camera = state.vertices[0].cpu().numpy()
            world = model.world_vertices(state, frame_indices=[i])[0].cpu().numpy()
            if not all(np.isfinite(a).all() for a in (uv, camera, world)):
                raise RuntimeError('Nonfinite sequence fit output: ' + model.frame_ids[i])
            if float(state.joints[..., 2].min()) <= .05:
                raise RuntimeError('Sequence fit has nonpositive/near-zero depth: ' + model.frame_ids[i])
            vc.append(camera.astype(np.float32)); vw.append(world.astype(np.float32))
            joints.append(state.joints[0].cpu().numpy()); pixels.append(uv)
    manifest = rt.read(next(p for p, _ in inputs.bound_files if p.name == 'inputs.json'))
    rows = manifest['records']
    np.savez_compressed(output, **{k: v.detach().cpu().numpy() for k, v in model.state.items()},
        vertices_camera=np.stack(vc), vertices_world=np.stack(vw), joints_camera=np.stack(joints),
        projected=np.stack(pixels), targets=inputs.targets, confidence=inputs.confidence,
        faces=model.body.faces, frame_ids=np.asarray(model.frame_ids), source_frames=inputs.source_frames,
        image_sha256=np.asarray([r['source_sha256'] for r in rows]),
        mask_sha256=np.asarray([r['mask_sha256'] for r in rows]),
        K_integer_centers=model.K.cpu().numpy(), w2c=model.w2c.cpu().numpy(), source_w2c=inputs.source_w2c,
        reference_frame_id=np.asarray(inputs.reference_frame_id), shared_betas=model.state['betas'].detach().cpu().numpy(),
        camera_length_contract_json=np.asarray(json.dumps(model.camera_length_contract, sort_keys=True)))


def fit_sequence(inputs_path, output, *, assets, seconds=3600, max_evaluations=6000):
    import torch
    import smplx
    from torch import nn
    from .sequence_input_adapter import load_sequence_inputs
    from .sequence_fitting import SharedIdentitySequence
    from .sequence_optimizer import optimize_sequence
    from hmr4d.utils.body_model.utils import smpl_to_openpose
    from shuolee_vposer import load_vposer

    inputs_path, output = Path(inputs_path).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = load_sequence_inputs(inputs_path, decode_images=True)
    n = len(inputs.frame_ids)
    plan = dict(mode='one_full_sequence_research_adaptation', inputs=str(inputs_path),
        source_sha256={str(inputs_path): rt.sha(inputs_path)}, frames=n,
        reference_frame_id=inputs.reference_frame_id, camera_length_contract=dict(inputs.camera_length_contract),
        evaluation_mode='streamed', seconds=seconds, max_evaluations=max_evaluations)
    rt.save(output / 'plan.json', plan)
    rt.save(output / 'progress.json', dict(state='running', stage='initialize'))
    mapping = smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                              use_face_contour=False, openpose_format='coco25')
    if len(mapping) != 118 or not np.array_equal(mapping[67:], np.arange(76, 127)):
        raise RuntimeError('Unexpected SMPL-X to observation joint mapping.')

    class JointMapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('indices', torch.tensor(mapping, dtype=torch.long))

        def forward(self, joints, **kwargs):
            return torch.index_select(joints, 1, self.indices)

    dtype = torch.float64
    body = smplx.SMPLX(str(Path(assets) / 'smplx/SMPLX_NEUTRAL.npz'), num_betas=10,
        num_expression_coeffs=10, use_pca=True, num_pca_comps=12, flat_hand_mean=False,
        use_face_contour=False, batch_size=1, joint_mapper=JointMapper()).to(device='cuda', dtype=dtype)
    vposer, _, _ = load_vposer('cuda')
    vposer = vposer.double()
    with torch.no_grad():
        latent = vposer.encode(torch.tensor(inputs.body_pose.copy(), device='cuda', dtype=dtype)).mean.detach().clone()
    model = SharedIdentitySequence(body, vposer, inputs.frame_ids, inputs.K_integer_centers.copy(),
        inputs.source_w2c.copy(), dict(global_orient=inputs.global_orient.copy(), transl=inputs.transl.copy(),
            latent=latent, jaw_pose=np.zeros((n, 3)), expression=np.zeros((n, 10))),
        camera_length_contract=dict(inputs.camera_length_contract), dtype=dtype)
    observations = inputs.make_observations(device='cuda', dtype=dtype)
    export_snapshot(model, inputs, output / 'initial.npz')
    initial_K, initial_E = model.K.clone(), model.w2c.clone()
    started = time.monotonic()

    def progress(info, theta):
        rt.save(output / 'progress.json', dict(state='running', updated=rt.now(), **info))

    result, _ = optimize_sequence(model, observations, inputs.wh[:, 1].tolist(), seconds=seconds,
        max_evaluations=max_evaluations, progress=progress, root_policy='provided_initialization',
        initialization_source=inputs.report()['initialization_evidence'], evaluation_mode='streamed')
    if result['state'] not in ('schedule_complete', 'budget_ended'):
        rt.save(output / 'result.json', dict(state=result['state'], optimizer=result,
            all_named_frames_exported=False, quality_accepted=False))
        rt.save(output / 'progress.json', dict(state='failed', pid=None, finished=rt.now()))
        raise RuntimeError('Sequence fitting failed: ' + str(result.get('stop', result['state'])))
    if not torch.equal(initial_K, model.K) or not torch.equal(initial_E, model.w2c):
        raise RuntimeError('Frozen cameras changed during sequence fitting.')
    export_snapshot(model, inputs, output / 'fit.npz')
    inputs.assert_unchanged()
    rt.save(output / 'result.json', dict(state=result['state'], optimizer=result, frames=n,
        shared_identity=True, all_named_frames_exported=True, reference_frame_id=inputs.reference_frame_id,
        fresh_optimizer=True, old_optimized_fit_loaded=False, all_K_E_unchanged=True,
        camera_length_contract=model.camera_length_contract,
        output_fit_sha256=rt.sha(output / 'fit.npz'), plan_sha256=rt.sha(output / 'plan.json'),
        seconds=time.monotonic() - started, quality_accepted=False, gaussian_steps=0))
    rt.save(output / 'progress.json', dict(state=result['state'], pid=None, finished=rt.now(), gaussian_steps=0))
    return output / 'fit.npz'
