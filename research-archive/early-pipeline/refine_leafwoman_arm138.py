"""Three 022 controls using the saved fit and original SMPLify-X stages 3/4."""
import difflib
import inspect
import os
from pathlib import Path
import time

import numpy as np
import torch
import smplx

import replay_leafwoman_arm138 as run
import official_smplifyx_entry123 as entry
import run_official_sequence123 as sequence
import smplify_seeded_refine124 as seeded

FREE = (13, 14, 16, 17, 18, 19)
LOCKED = tuple(i - 1 for i in range(1, 22) if i not in FREE)
RUNTIME = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-new-capture-118')


class ArmDecoder:
    def __init__(self, original, anchor):
        self.original = original
        self.anchor = torch.as_tensor(anchor).reshape(1, 21, 3).detach()
        self.locked = torch.tensor(LOCKED)

    def decode(self, latent):
        pose = self.original.decode(latent)['pose_body'].reshape(1, 21, 3)
        return {'pose_body': pose.index_copy(1, self.locked, self.anchor[:, self.locked])}


def arm_source(root):
    original, diff = seeded.source(root)
    changed = seeded.once(original,
        '                body_params = list(body_model.parameters())\n                body_params.append(camera.translation)',
        '                body_params = []\n                camera.translation.requires_grad_(False)')
    marker = '                body_optimizer, body_create_graph = optim_factory.create_optimizer(\n'
    changed = seeded.once(changed, marker,
        '                assert len(final_params) == 1 and final_params[0] is pose_embedding\n'
        '                assert tuple(pose_embedding.shape) == (1,32)\n'
        "                kwargs['_actual_stages'].append(opt_idx)\n" + marker)
    delta = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
                                      fromfile='seeded-stage34', tofile='arm-latent-only'))
    return changed, diff + '\n' + delta


def main():
    assert run.WSL and os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    assert not (run.OUT/'arm-launch.json').exists()
    replay = run.read(run.OUT/'zero-replay-result.json')
    plan = run.read(run.OUT/'plan.json')
    assert replay['plan_sha256'] == entry.sha(run.OUT/'plan.json')
    assert replay['state'] == 'pinned_SMPLX_zero_replay_passed'
    for path, digest in plan['source_hashes'].items():
        assert entry.sha(run.mounted(path)) == digest
    fit_plan = run.read(run.FIT/'plan.json')
    for path, digest in fit_plan['source_hashes'].items():
        if path.endswith(('smplx/body_models.py', 'smplx/lbs.py', 'official_smplifyx_entry123.py')):
            assert entry.sha(path) == digest
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    for file in ('body_models.py', 'lbs.py'):
        assert Path(smplx.__file__).parent/file == run.SMPLX_ROOT/'smplx'/file
    source, _ = arm_source(run.OFFICIAL)
    compile(source, '<022 original stage 3/4 arm control>', 'exec')
    loader = entry.module(RUNTIME/'src/avatar_core/shuolee_vposer.py', '_arm138_vposer')
    loader.MODEL = run.ASSET.parent.parent/'vposer/V02_05'
    loader.CODE = RUNTIME/'third_party/human_body_prior'
    vposer, _, vposer_meta = loader.load_vposer('cpu')
    utils = entry.module(run.OFFICIAL/'smplifyx/utils.py', '_leaf138_fit_utils')
    mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                   use_face_contour=False, openpose_format='coco25')
    body = smplx.SMPLX(str(run.ASSET), batch_size=1, num_betas=10, num_expression_coeffs=10,
        use_pca=True, num_pca_comps=12, flat_hand_mean=False, use_face_contour=False,
        create_body_pose=False, create_transl=False, joint_mapper=utils.JointMapper(mapping),
        dtype=torch.float32).eval().requires_grad_(False)

    scope = dict(seeded.__dict__, source=arm_source)
    rows = {r['frame']: r for r in fit_plan['rows']}
    started = time.monotonic()
    run.atomic(run.OUT/'arm-launch.json', dict(pid=os.getpid(), frames=plan['frames'], device='cpu',
        runner_sha256=entry.sha(__file__), vposer=vposer_meta, pose_joint_ids=list(FREE),
        holdout_used=False, no_new_camera=True, gaussian_started=False))
    records = []
    with np.load(run.ARCHIVE, allow_pickle=False) as archive:
        assert np.array_equal(body.faces, archive['faces'])
        shared = {name: archive[name] for name in ('betas', 'body_pose_axis_angle', 'pose_embedding')}
        per_view = {name: archive[name] for name in (*run.PARAMS[1:], 'camera_translation')}
        K = archive['K_integer_centers']
        observed, projected = archive['observed_native_keypoints'], archive['joints_projected_native']
        old_joints = archive['joints_camera']
        for i, frame in zip(plan['slots'], plan['frames']):
            row = rows[frame]
            assert np.array_equal(K, row['K_integer_centers'])
            assert row['source_sha256'] == str(archive['image_sha256'][i])
            image = sequence.checked_image(row)
            seed = dict(shared, **{k: v[i:i+1] for k, v in per_view.items()})
            decoder = ArmDecoder(vposer, seed['body_pose_axis_angle'])
            latent = torch.from_numpy(seed['pose_embedding']).requires_grad_(True)
            angles = decoder.decode(latent)['pose_body']
            assert np.max(np.abs(angles.reshape(1, 63).detach().numpy() - seed['body_pose_axis_angle'])) < 2e-6
            grad = torch.autograd.grad(angles[:, [i-1 for i in FREE]].square().sum(), latent)[0]
            assert torch.isfinite(grad).all() and grad.abs().max() > 0
            stages = []

            def config(root):
                return dict(seeded.BASE_CONFIG(root), hand_joints_weights=[0.]*5, _actual_stages=stages)

            scope['BASE_CONFIG'] = config
            exec(compile(inspect.getsource(seeded.fit), '<022 seeded original fit>', 'exec'), scope)
            output, report = scope['fit'](saved_seed=seed, vposer2=decoder, official_root=run.OFFICIAL,
                image=image, keypoints=observed[i:i+1], K_integer_centers=K, body_model=body,
                result_path=run.OUT/(frame+'.pkl'), shared_betas=seed['betas'], freeze_betas=True,
                device='cpu', diff_path=run.OUT/'executed-arm-interface.diff')
            assert stages == [3, 4] and report['anchor_calls'] > 0
            for name in (*run.PARAMS, 'camera_translation'):
                assert np.array_equal(seed[name], output[name]), name
            assert np.array_equal(output['body_pose_axis_angle'].reshape(1, 21, 3)[:, LOCKED],
                                  seed['body_pose_axis_angle'].reshape(1, 21, 3)[:, LOCKED])
            assert np.array_equal(output['K_integer_centers'], K)
            arm = slice(2, 8)
            target = observed[i, arm, :2]
            old, new = projected[i, arm], output['joints_projected_native'][0, arm]
            before, after = (np.linalg.norm(a-target, axis=1) for a in (old, new))
            uv = output['joints_projected_native'][0]
            leg = [*range(9, 17), *range(19, 25)]
            depth = lambda j: (j[[4, 7], 2] - j[[2, 5], 2]).tolist()
            np.savez_compressed(run.OUT/(frame+'.npz'), **output)
            records.append(dict(frame=frame, optimizer_stages=stages, closure_calls=report['anchor_calls'],
                before_arm_detector_px=before.tolist(), after_arm_detector_px=after.tolist(),
                before_wrist_minus_shoulder_z=depth(old_joints[i]), after_wrist_minus_shoulder_z=depth(output['joints_camera'][0]),
                face_projection_max_change_px=float(np.linalg.norm(uv[67:]-projected[i,67:],axis=1).max()),
                legs_projection_max_change_px=float(np.linalg.norm(uv[leg]-projected[i,leg],axis=1).max())))
            print('ARM138', frame, np.round(before, 2), np.round(after, 2), flush=True)
    assert not torch.cuda.is_initialized()
    run.atomic(run.OUT/'arm-result.json', dict(state='three_controls_completed_review_pending', rows=records,
        seconds=time.monotonic()-started, optimizable_tensors_per_stage=['pose_embedding[1,32]'],
        fixed_K_root_beta_translation_face_hands=True, physical_depth_accepted=False,
        gaussian_started=False, quality_accepted=False))


if __name__ == '__main__':
    main()
