"""Input-only handoff from the completed seven-view sofa fit to frozen SH0 GS.

No pose fitting is performed here. This is the user-authorized exploratory
pipeline run; prior rejected pose-quality evidence remains unchanged.
"""
import argparse
import copy
import difflib
import json
from pathlib import Path
import numpy as np
import cross_subject_sh0124 as gs

SUBJECT = 'sofa-selected-sh0-pipeline124'
OUT = gs.ROOT / SUBJECT
SOURCE = gs.ROOT / 'sofa-selected-view-pose124'
REFERENCE = 'frame_000026'


def prepare():
    assert not OUT.exists(), 'Never overwrite or repeat an existing experiment.'
    source_plan = gs.read(SOURCE / 'plan.json')
    with np.load(SOURCE / 'selected-fit.npz', allow_pickle=False) as z:
        fit = dict(z)
    rows = copy.deepcopy(source_plan['rows'])
    ids = fit['frame_ids'].astype(str).tolist()
    n = len(ids)
    assert n == 7 and ids == [r['frame'] for r in rows]
    order = [ids.index(REFERENCE)] + [i for i, f in enumerate(ids) if f != REFERENCE]
    fit = {k: (v[order] if v.ndim and v.shape[0] == n else v) for k, v in fit.items()}
    rows = [rows[i] for i in order]
    fit['body_pose_axis_angle'] = np.repeat(fit['body_pose_axis_angle'], n, axis=0)
    fit['pose_embedding'] = np.repeat(fit['pose_embedding'], n, axis=0)
    fit['shared_betas'] = fit['betas'].copy()
    fit['K_integer_centers'] = np.asarray([r['K_integer_centers'] for r in rows], dtype=np.float64)
    fit['w2c'] = np.asarray([r['w2c'] for r in rows], dtype=np.float64)
    assert np.array_equal(fit['w2c'], np.broadcast_to(np.eye(4), (n, 4, 4)))
    fit['vertices_world'] = fit['vertices_camera'].astype(np.float32)
    fit['reference_frame_id'] = np.asarray(REFERENCE)
    fit['static_offset'] = np.zeros((1, 10475, 3), dtype=np.float32)
    for i, r in enumerate(rows):
        assert gs.sha(r['image']) == r['source_sha256'] == str(fit['image_sha256'][i])
        assert gs.sha(r['mask']) == r['mask_sha256'] == str(fit['mask_sha256'][i])
        assert fit['source_frames'][i] == r['source_frame_index']
        kc = fit['K_integer_centers'][i].copy()
        kc[0, 2] += .5
        kc[1, 2] += .5
        assert np.array_equal(kc, r['K'])
    import torch
    from frozen_skeleton_surface123 import FrozenSkeletonSurface
    from run_official_sequence123 import root_to_world
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    bridge = FrozenSkeletonSurface(fit, fit['static_offset'][0], gs.ASSETS / 'smplx/SMPLX_NEUTRAL.npz')
    body = bridge.model
    reg = body.J_regressor[0].double().numpy()
    rt = reg @ body.v_template.double().numpy()
    rs = np.einsum('v,vck->ck', reg, body.shapedirs.double().numpy())
    re = np.einsum('v,vck->ck', reg, body.expr_dirs.double().numpy())
    root = [root_to_world(dict(betas=fit['betas'], expression=fit['expression'][i],
                global_orient=fit['global_orient'][i], camera_translation=fit['camera_translation'][i]),
                fit['w2c'][i], rt, rs, re) for i in range(n)]
    fit['root_rotation'] = np.asarray([r[0] for r in root])
    fit['world_translation'] = np.asarray([r[1] for r in root])
    fit['root_pivot_model'] = np.asarray([r[2] for r in root])
    replay = float((bridge.base_vertices - torch.as_tensor(fit['vertices_world'])).abs().max())
    assert replay < 2e-6, replay
    with np.load(gs.BASE / 'fit/surface-cache.npz', allow_pickle=False) as z:
        policy = dict(z)
    assert np.array_equal(policy['faces'], fit['faces'])
    oracle = []
    with torch.no_grad():
        delta = torch.sin(torch.arange(10475 * 3).reshape(10475, 3).float()) * .0001
        delta *= torch.tensor(policy['clothing_vertices'])[:, None]
        for i in range(n):
            v, j = bridge.independent_original_forward(delta, i)
            ve = float((bridge.vertices(delta, i) - v).abs().max())
            je = float((bridge.joints(i) - j).abs().max())
            assert ve < 2e-6 and je < 2e-6
            oracle.append(dict(frame=rows[i]['frame'], vertices_error_m=ve, joints_error_m=je))
    (OUT / 'fit').mkdir(parents=True)
    (OUT / 'data').mkdir()
    np.savez_compressed(OUT / 'fit/fitted.npz', **fit)
    gs.save(OUT / 'data/training-inputs.json', dict(records=rows, vhap_used=False,
        quality_accepted=False, source_fit=str(SOURCE / 'selected-fit.npz'),
        source_fit_sha256=gs.sha(SOURCE / 'selected-fit.npz'), camera_kind='camera_relative_fixedK_body_motion'))
    np.savez_compressed(OUT / 'fit/surface-cache.npz', frame_ids=fit['frame_ids'], faces=fit['faces'],
        base_vertices=bridge.base_vertices.numpy(), linear=bridge.linear.numpy(),
        skeletal_joints=bridge.base_skeleton.numpy(), clothing_vertices=policy['clothing_vertices'],
        protected_face_vertices=policy['protected_face_vertices'])
    gs.save(OUT / 'smplify-gaussian-interface-receipt.json', dict(updated=gs.now(), frames=n,
        reference_frame=REFERENCE, fit_sha256=gs.sha(OUT / 'fit/fitted.npz'),
        training_inputs_sha256=gs.sha(OUT / 'data/training-inputs.json'),
        surface_cache_sha256=gs.sha(OUT / 'fit/surface-cache.npz'), replay_max_abs_m=replay,
        oracle=oracle, no_demo_person_geometry=True, topology_policy_shared=True, vhap_used=False,
        quality_accepted=False))
    selected = [r['frame'] for r in rows]
    gs.save(OUT / 'selection.json', dict(selected_frames=selected, count=n,
        policy='All seven preregistered selected-view fit inputs; no additional manual best-frame selection.',
        source_plan_sha256=gs.sha(SOURCE / 'plan.json'), quality_accepted=False))
    gs.save(OUT / 'selection-rule.json', dict(rule='Reuse the frozen selected-view roster, no post-training selection',
        frames=selected, held_out_preserved=True))
    old = gs.generate_source(OUT, n, REFERENCE)
    assert old.count('assert len(training_slots)==8') == 1 and old.count('training_frames=8') == 1
    new = old.replace('assert len(training_slots)==8', 'assert len(training_slots)==7').replace('training_frames=8', 'training_frames=7')
    (OUT / 'adapter.py').write_text(new)
    (OUT / 'seven-frame-interface.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='eight-slot-adapter', tofile='seven-slot-adapter')))
    p = copy.deepcopy(gs.read(gs.BASE / 'plan.json'))
    for k in ['supersedes', 'baseline_run', 'prior_baseline', 'admission']:
        p.pop(k, None)
    p.update(updated=gs.now(), state='prepared', subject=SUBJECT, label='沙发人物7帧 SH0 全流程试跑',
        user_authorization='不用一直修复姿态，而是考虑整个流程的可复制性，先训练高斯看看',
        route='Frozen selected SMPLify-X inputs -> original GaussianAvatars training with existing SH0 clothing-offset adapter',
        source_fit=str(SOURCE / 'selected-fit.npz'), reference_frame=REFERENCE, frames=n,
        evaluation_frames=n, training_frames=n, selected_frames=selected, record_id='006',
        from_scratch=True, gaussian_parameters_reused=False, optimizer_state_reused=False,
        cached_subject_fitting=True, cold_start_benchmark=False, demo_person_shape_reused=False,
        only_parameter_change='Seven preregistered sofa inputs; existing SH0, offsets, losses, optimizer and density schedule unchanged.',
        comparison_budget=18000, snapshots=[1000, 10000, 18000],
        review_gate='Actual1000 review; continue same live optimizer to18000 unless new implementation or severe degeneration appears.',
        source_fit_quality_accepted=False, quality_accepted=False, model_promoted=False,
        end_to_end_15min_accepted=False, upstream_training_function_unchanged=True,
        prior_pose_failure=str(SOURCE / 'conclusion.json'))
    sources = [Path(__file__), Path(gs.__file__), gs.BASE / 'adapter.py', SOURCE / 'selected-fit.npz', SOURCE / 'plan.json',
        gs.UP / 'train.py', gs.UP / 'scene/gaussian_model.py', gs.UP / 'arguments/__init__.py',
        gs.PROJECT / 'src/avatar_core/scene/smplx_gaussian_model.py', gs.PROJECT / 'src/shuolee_portrait/training.py',
        gs.PROJECT / 'src/avatar_core/gaussian_renderer/__init__.py', Path(p['laplacian_cache'])]
    sources += [gs.HERE / name for name in ['smplify_direct_surface123.py', 'frozen_skeleton_surface123.py',
        'facial_geometry_boundary123.py', 'body_root_sh_colors123.py', 'run_official_sequence123.py']]
    sources += [OUT / name for name in ['adapter.py', 'selection.json', 'selection-rule.json', 'fit/fitted.npz',
        'fit/surface-cache.npz', 'data/training-inputs.json', 'smplify-gaussian-interface-receipt.json']]
    p['source_hashes'] = {str(f): gs.sha(f) for f in sources}
    gs.save(OUT / 'plan.json', p)
    gs.save(OUT / 'progress.json', dict(updated=gs.now(), state='prepared', step=0, pid=None))
    print(json.dumps(dict(state='prepared', subject=SUBJECT, frames=n, replay_max_abs_m=replay)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'train'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        gs.train(SUBJECT)
