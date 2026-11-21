"""One frozen-body SH0 appearance polish on the exact delivered 006 terminal."""

import argparse
from collections import Counter
import fcntl
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
from plyfile import PlyData, PlyElement

import launch_face_photometric_006 as face_trial
from review_bound_alpha_ab import delivered, pin, read, require, sha
from review_bound_scale_coverage import exported_view
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, save


STOCK = LAB/'shuolee-smpl-unified-006/run'
PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-unified-20261110')
OUT = LAB/'shuolee-smpl-006-head-appearance-pilot'
SNAPSHOT = 'gaussian-run/step-000600'
ACCEPTANCE = face_trial.ACCEPTANCE.with_name('smpl-head-appearance-pilot-preregistration.json')
ACCEPTANCE_SHA = 'f1026da9d703692c2584ac2a8c10933bc0bd38ece06c1088cc511fc145bfefe0'
STOCK_RESULT = '18333c3592caa278dfdd9cb8c11ee3ec2b561e94688178587824b31208c6faf7'
LOSS_SOURCE = '912e2dc3943618b81c6738cf8f85bfea9cfa3e18942a284bf50fd9f7942fb0d6'
GRADIENT_AUDIT = LAB/'shuolee-smpl-face-photometric-006-gradient-support/result.json'
GRADIENT_SHA = '11c3f345858ca41483557cb6ec191d46231b56737c7378ad713620ffe0b1361d'
STEPS = 600
LEARNING_RATES = {'_features_dc': .0025, '_opacity': .05}
FIELDS = ('f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity')
FROZEN_FILES = ('fit/fitted.npz', 'fit/selection.json', 'fit/shared-body.json', 'fit/official-fit-result.json',
                'data/training-inputs.json', 'gaussian-run/training-cameras.json')
PORTABLE_FILES = ('cameras.json', 'reference-mesh.obj', 'colmap/cameras.txt', 'colmap/images.txt', 'colmap/points3D.txt')


def bitwise(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def table(path):
    data = PlyData.read(path, mmap=False)
    require(not data.text and [element.name for element in data.elements] == ['vertex'],
            f'Expected a binary vertex-only Gaussian PLY: {path}.')
    return data['vertex'].data


def head_mask(faces, binding):
    with np.load(ASSET, allow_pickle=False) as asset:
        require(np.array_equal(asset['f'], faces), 'Model asset topology changed.')
        dominant = asset['weights'].argmax(axis=1)
        landmarks = np.concatenate((asset['lmk_faces_idx'].ravel(), asset['dynamic_lmk_faces_idx'].ravel()))
    protected = np.isin(dominant, [12, 15, 22, 23, 24])
    protected[faces[landmarks].ravel()] = True
    return protected[faces].all(axis=1)[binding]


def load_loss():
    path = face_trial.PROJECT/'src/shuolee_portrait/training.py'
    pin(path, LOSS_SOURCE)
    spec = importlib.util.spec_from_file_location('shuolee_portrait._registered_face_loss', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._face_interior_pixels, module._sh0_foreground_alpha_loss


def preflight():
    face_trial.base.common.git_state(face_trial.BASE_COMMIT)
    face_trial.verify_source()
    face_trial.verify_acceptance()
    pin(ACCEPTANCE, ACCEPTANCE_SHA)
    acceptance = read(ACCEPTANCE)
    require(acceptance['state'] == 'registered_before_pilot_optimization' and acceptance['subject'] == '006'
            and acceptance['baseline_commit'] == face_trial.BASE_COMMIT
            and acceptance['baseline_terminal'] == {'path': str(STOCK/'result.json'), 'sha256': STOCK_RESULT}
            and acceptance['locked_metrics_plan_sha256'] == face_trial.ACCEPTANCE_SHA
            and acceptance['gates'] == read(face_trial.ACCEPTANCE)['gates']
            and acceptance['geometry']['verify_bitwise'] is True
            and acceptance['objective'] == dict(original_foreground_and_alpha=True, face_two_background_weight=.05,
                                               extra_raster_passes=0)
            and acceptance['optimizer'] == dict(iterations=STEPS, fresh=True, position_updates=0, densification=False,
                opacity_reset=False, feature_dc_lr=LEARNING_RATES['_features_dc'], opacity_logit_lr=LEARNING_RATES['_opacity'],
                seed=0, frame_order='same balanced source-bin routine'), 'The registered pilot contract changed.')
    pin(STOCK/'result.json', STOCK_RESULT)
    pin(GRADIENT_AUDIT, GRADIENT_SHA)
    audit = read(GRADIENT_AUDIT)
    require(audit['state'] == 'completed' and audit['groups'][0] == 'all_three_protected'
            and all(groups is None or all(g['nonzero'] == 0 for g in groups[1:])
                    for record in audit['records'] for groups in record['gradients'].values()),
            'The registered direct-gradient ownership result changed.')
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import inspect_bound_ply
    from shuolee_portrait.stable_selection import source_bin_training_order
    from shuolee_portrait.training import _native_foreground_target

    require(rt.PROJECT == PROJECT, 'Use the frozen stock runtime.')
    rows = read(STOCK/'data/training-inputs.json')['records']
    selection = read(STOCK/'fit/selection.json')
    profile = read(STOCK/'gaussian-run/config.json')
    saved = delivered(STOCK, ASSET, rows, selection['retained_holdouts'])
    require(len(rows) == len(audit['records']) == 24 and saved.facts['source_files_sha256'] == saved.bound['source_files_sha256']
            and profile['seed'] == 0 and profile['alpha_loss_weight'] == .1
            and [profile['optimization'][name] for name in ('feature_lr', 'opacity_lr')] == list(LEARNING_RATES.values())
            and profile['sampling_policy'] == 'seeded_equal_occupied_source_bin_cycles', 'Wrong stock objective or training roster.')
    inputs = {str(STOCK/name): sha(STOCK/name) for name in FROZEN_FILES}
    inputs.update({str(STOCK/'result.json'): STOCK_RESULT, str(GRADIENT_AUDIT): GRADIENT_SHA,
                   str(ASSET): saved.facts['asset_sha256'], str(ACCEPTANCE): ACCEPTANCE_SHA,
                   str(STOCK/'gaussian-run/config.json'): sha(STOCK/'gaussian-run/config.json')})
    for key in ('fit', 'rows', 'selection', 'head_ceiling', 'direct_gradient', 'heldout_owner_attribution', 'native_and_dolly_cameras'):
        proof = acceptance[key]
        inputs[proof['path']] = pin(Path(proof['path']), proof['sha256'])
    for name, digest in face_trial.CAMERA_PINS.items():
        inputs[str(STOCK/name)] = pin(STOCK/name, digest)
    export = STOCK/'export'
    mesh = read(export/'mesh-provenance.json')
    camera = read(export/'camera_provenance.json')
    for name, digest in camera['files_sha256'].items():
        inputs[str(export/name)] = pin(export/name, digest)
    inputs[str(export/'reference-mesh.obj')] = pin(export/'reference-mesh.obj', mesh['mesh_sha256'])
    for name in ('mesh-provenance.json', 'camera_provenance.json', 'validation.json'):
        inputs[str(export/name)] = sha(export/name)
    for name in ('reference-mesh.npz', 'bound-replay.json', 'bound-local.ply', 'world.ply', 'native-ply-parity.json'):
        inputs[str(saved.snapshot/name)] = sha(saved.snapshot/name)
    ids, _ = inspect_bound_ply(saved.snapshot/'bound-local.ply', len(saved.fit['faces']))
    editable = head_mask(saved.fit['faces'], ids)
    require(editable.any() and (~editable).any(), 'Expected a strict owner subset.')
    local = table(saved.snapshot/'bound-local.ply')
    for path in (saved.snapshot/'world.ply', export/'point_cloud.ply'):
        values = table(path)
        require(all(bitwise(values[field], local[field]) for field in FIELDS), 'Stock PLY attribute order differs.')
        inputs[str(path)] = sha(path)
    roi, loss = load_loss()
    targets, roi_counts = [], {}
    for row, item in zip(rows, saved.facts['frames'], strict=True):
        inputs[row['image']] = pin(Path(row['image']), item['source_sha256'])
        inputs[row['mask']] = pin(Path(row['mask']), item['mask_sha256'])
        rgb, mask, box = _native_foreground_target(row)
        pixels = roi(saved.fit['observed_native_keypoints'][item['pose_index'], 67:118], mask, box)
        require(pixels is not None, f'No registered face ROI for {row["frame"]}.')
        targets.append((rgb, mask, box, pixels))
        roi_counts[row['frame']] = pixels.shape[1]
    require(roi_counts == {item['frame']: item['roi_pixels'] for item in audit['frames']}, 'Frozen native face ROIs differ.')
    row_bins, generator = source_bin_training_order(rows, selection, seed=profile['seed'])
    order = list(islice(generator, STEPS))
    draws = Counter(rows[i]['frame'] for i in order)
    require(len(draws) == 24 and set(draws).isdisjoint(selection['retained_holdouts']), 'Protected optimizer inputs differ.')
    plan = dict(schema='shuolee-head-appearance-pilot-v1', source_terminal_sha256=STOCK_RESULT,
                acceptance_plan_sha256=ACCEPTANCE_SHA, locked_metrics_plan_sha256=face_trial.ACCEPTANCE_SHA,
                starting_body_steps=10000, appearance_steps=STEPS, seed=profile['seed'], learning_rates=LEARNING_RATES,
                optimizer='Fresh Adam, eps=1e-15, two strict-owned parameter arrays', face_weight=.05, alpha_weight=.1,
                source_fit_reused=True, fresh_body=False, cold_video_clock=False, fit_updates=0,
                editable_points=int(editable.sum()), frozen_points=int((~editable).sum()),
                editable_rule='All three bound triangle vertices protected by the existing SMPL-X head mask.',
                frozen_attributes=['xyz', 'scaling', 'rotation', 'binding', 'all posed meshes', 'shared static offset',
                                   'non-owned DC and opacity'], no_densification_pruning_or_resets=True,
                training_frames=[row['frame'] for row in rows], protected_holdouts=selection['retained_holdouts'],
                ordered_slots=order, frame_draws=dict(draws), bin_draws=dict(Counter(str(row_bins[i]) for i in order)),
                roi_pixels=roi_counts, input_sha256=inputs, code_sha256=sha(Path(__file__)), loss_source_sha256=LOSS_SOURCE)
    return plan, SimpleNamespace(rows=rows, targets=targets, saved=saved, editable=editable, loss=loss, profile=profile)


def update_attributes(source, target, editable, colors, opacity):
    values = table(source).copy()
    for name, replacement in zip(FIELDS, (*colors.T, opacity), strict=True):
        values[name][editable] = replacement
    PlyData([PlyElement.describe(values, 'vertex')], text=False, byte_order='<').write(target)


def verify_attributes(before, after, editable):
    a, b = table(before), table(after)
    require(a.dtype == b.dtype and len(a) == len(b) == len(editable)
            and bitwise(a[~editable], b[~editable]), 'Frozen PLY rows changed.')
    require(all(bitwise(a[name], b[name]) for name in a.dtype.names if name not in FIELDS),
            'Geometry or binding bytes changed.')
    return int(np.any(np.stack([a[name] != b[name] for name in FIELDS]), axis=0).sum())


def polish(state, plan):
    import torch
    from gaussian_renderer import render
    from shuolee_portrait.bound_replay import RAW_PARAMETERS, _load_model
    from shuolee_portrait.training import camera, render_pose

    saved = state.saved
    model, original_parity = _load_model(saved.snapshot/'bound-local.ply', saved.snapshot, saved.fit,
                                         saved.mesh, saved.facts, ASSET)
    owned = torch.from_numpy(np.flatnonzero(state.editable)).to(device='cuda')
    fixed = {key: getattr(model, key).detach().clone() for key in RAW_PARAMETERS}
    leaves = {key: torch.nn.Parameter(fixed[key].index_select(0, owned)) for key in LEARNING_RATES}
    optimizer = torch.optim.Adam([{'params': [leaf], 'lr': LEARNING_RATES[key]} for key, leaf in leaves.items()],
                                 lr=0., eps=1e-15)
    require(not optimizer.state, 'Expected fresh head-only Adam state.')
    geometry = [model.binding.clone(), model.faces.clone(), model.vertices_sequence.clone()]
    views = [camera(row, i, pose_index=item['pose_index'])[0]
             for i, (row, item) in enumerate(zip(state.rows, saved.facts['frames'], strict=True))]
    targets = [(rgb.cuda(), mask.cuda(), box, torch.as_tensor(pixels, device='cuda'))
               for rgb, mask, box, pixels in state.targets]
    state.targets.clear()
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')
    override = torch.ones_like(model._xyz)
    start = time.monotonic()
    with (OUT/'loss.jsonl').open('x', encoding='utf-8') as log:
        for step, i in enumerate(plan['ordered_slots'], 1):
            for key, leaf in leaves.items():
                setattr(model, key, fixed[key].index_copy(0, owned, leaf))
            slot = saved.facts['frames'][i]['pose_index']
            rgb = render_pose(render, views[i], model, pipe, black, slot)['render']
            alpha = render(views[i], model, pipe, black, override_color=override)['render'][:1]
            target_rgb, target_mask, box, pixels = targets[i]
            photo, l1, ssim, alpha_loss = state.loss(rgb, alpha, target_rgb, target_mask, box,
                alpha_floor=state.profile['straight_rgb_alpha_floor'], face_pixels=pixels, face_weight=plan['face_weight'])
            loss = photo + plan['alpha_weight']*alpha_loss
            loss.backward()
            report_step = step == 1 or step % 50 == 0
            if report_step:
                require(torch.isfinite(loss.detach()) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                                             for p in leaves.values()), 'Invalid head-only optimizer gradient.')
            if step == 1:
                require(all(p.grad.count_nonzero() for p in leaves.values()), 'Head-only appearance gradients are absent.')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if report_step:
                status = dict(step=step, target_steps=STEPS, frame=state.rows[i]['frame'], seconds=time.monotonic()-start,
                              loss=float(loss.detach()), alpha_loss=float(alpha_loss.detach()),
                              face_term=float((photo-.8*l1-.2*(1-ssim)).detach()))
                log.write(json.dumps(status)+'\n')
                log.flush()
                save(OUT/'progress.json', status)
    for key, leaf in leaves.items():
        setattr(model, key, fixed[key].index_copy(0, owned, leaf.detach()))
    require(all(torch.equal(a, b) for a, b in zip(geometry, (model.binding, model.faces, model.vertices_sequence), strict=True))
            and all(torch.equal(getattr(model, key), fixed[key]) for key in RAW_PARAMETERS if key not in LEARNING_RATES)
            and model.optimizer is None and len(optimizer.state) == 2
            and all(int(s['step']) == STEPS for s in optimizer.state.values()),
            'An immutable model attribute or optimizer scope changed.')
    return model, original_parity, time.monotonic()-start


def delivery(state, model, plan):
    import torch
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from shuolee_portrait.bound_replay import save_bound_snapshot
    from shuolee_portrait.export import _package_smpl_sh0
    from shuolee_portrait.training import camera, image, reference_ply_comparison, render_pose

    saved = state.saved
    snapshot, portable = OUT/SNAPSHOT, OUT/'export'
    snapshot.mkdir(parents=True)
    portable.mkdir()
    copies = [(STOCK/name, OUT/name) for name in FROZEN_FILES]
    copies += [(STOCK/'export'/name, portable/name) for name in PORTABLE_FILES]
    copies.append((saved.snapshot/'reference-mesh.npz', snapshot/'reference-mesh.npz'))
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        pin(target, plan['input_sha256'][str(source)])

    ref = saved.facts['reference_pose_index']
    model.select_mesh_by_timestep(ref)
    colors = model._features_dc.detach().cpu().numpy()[state.editable, 0, :]
    opacity = model._opacity.detach().cpu().numpy()[state.editable, 0]
    stock_portable = STOCK/'export'
    update_attributes(saved.snapshot/'world.ply', snapshot/'world.ply', state.editable, colors, opacity)
    update_attributes(stock_portable/'point_cloud.ply', portable/'point_cloud.ply', state.editable, colors, opacity)
    world, viewer = GaussianModel(0), GaussianModel(0)
    world.load_ply(str(snapshot/'world.ply'))
    viewer.load_ply(str(portable/'point_cloud.ply'))
    row, = [item for item in state.rows if item['frame'] == saved.facts['reference_frame_id']]
    validation = read(stock_portable/'validation.json')
    cam, _ = camera(row, 0, pose_index=ref)
    viewer_cam, _ = camera(exported_view(row, validation), 0, pose_index=ref)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')

    def check(delta):
        evidence = dict(frame=row['frame'], mean_abs=float(delta.mean()), max_abs=float(delta.max()))
        require(evidence['mean_abs'] < 2e-6 and evidence['max_abs'] < .005, 'Native PLY/camera readback mismatch.')
        return evidence

    with torch.no_grad():
        rgb, delta = reference_ply_comparison(render, cam, model, world, pipe, black, ref)
        native_check = check(delta)
        native = render_pose(render, cam, world, pipe, black, ref)['render']
        presented = render_pose(render, viewer_cam, viewer, pipe, black, ref)['render']
        viewer_check = check(abs(native-presented))
        image(rgb).save(snapshot/'reference-render.png')
    save(snapshot/'native-ply-parity.json', dict(passed=True, checks=[native_check], sha256=sha(snapshot/'world.ply'),
        mesh_sha256=sha(snapshot/'reference-mesh.npz'), reference_frame_id=row['frame'], reference_pose_index=ref))
    _, manifest = save_bound_snapshot(OUT, snapshot, model, asset_path=ASSET)
    pairs = {'bound-local': (saved.snapshot/'bound-local.ply', snapshot/'bound-local.ply'),
             'native-world': (saved.snapshot/'world.ply', snapshot/'world.ply'),
             'viewer-world': (stock_portable/'point_cloud.ply', portable/'point_cloud.ply')}
    changed = {name: verify_attributes(*pair, state.editable) for name, pair in pairs.items()}
    require(len(set(changed.values())) == 1 and min(changed.values()) > 0, 'No consistent owned appearance update in the PLYs.')
    validation.update(sha256=sha(portable/'point_cloud.ply'), native_render_checks=[viewer_check], format_passed=True)
    save(portable/'validation.json', validation)
    provenance = read(stock_portable/'camera_provenance.json')
    save(portable/'camera_provenance.json', dict(provenance, world_to_viewer=validation))
    mesh = dict(read(stock_portable/'mesh-provenance.json'), ply_sha256=validation['sha256'])
    save(portable/'mesh-provenance.json', mesh)
    scene_sha = _package_smpl_sh0(portable, {'steps': plan['starting_body_steps']}, validation, mesh)
    return dict(snapshot=SNAPSHOT, bound_ply=str((snapshot/'bound-local.ply').relative_to(OUT)),
                bound_ply_sha256=manifest['bound_ply_sha256'], bound_manifest_sha256=sha(snapshot/'bound-replay.json'),
                native_world_ply_sha256=sha(snapshot/'world.ply'),
                viewer_ply='export/point_cloud.ply', viewer_ply_sha256=validation['sha256'],
                scene='export/scene.zip', scene_sha256=scene_sha, camera='export/cameras.json', camera_sha256=mesh['camera_sha256'],
                frozen_reference_mesh_sha256=mesh['mesh_sha256'], native_and_exported_camera_parity=[native_check, viewer_check],
                exact_frozen_and_geometry_ply_bytes=True, changed_head_rows_by_ply=changed)


def run(plan, state):
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.cli import gpu_lock

    require(read(OUT/'preregistration.json') == plan and not (OUT/'loss.jsonl').exists(), 'Unexpected pilot state or input.')
    start = time.monotonic()
    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            require(torch.cuda.device_count() == 1, 'Expose one CUDA device.')
            rt.configure(ASSET.parents[1])
            model, original_parity, seconds = polish(state, plan)
            result = delivery(state, model, plan)
    for path, digest in plan['input_sha256'].items():
        pin(Path(path), digest)
    target = OUT/'result.json'
    save(target, dict(state='diagnostic_completed', preregistration_sha256=sha(OUT/'preregistration.json'),
                      acceptance_plan_sha256=ACCEPTANCE_SHA, locked_metrics_plan_sha256=face_trial.ACCEPTANCE_SHA,
                      start_terminal_sha256=STOCK_RESULT, original_readback=original_parity, head_optimizer_updates=STEPS,
                      reused_terminal=True, cold_video_clock=False, from_scratch=False,
                      head_training_seconds=seconds, head_stage_to_package_seconds=time.monotonic()-start,
                      fit_updates=0, quality_accepted=False, **result))
    print(json.dumps(dict(result=str(target), sha256=sha(target))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'register', 'run'))
    args = parser.parse_args()
    if args.action != 'run':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    plan, state = preflight()
    if args.action == 'run':
        run(plan, state)
        return
    if args.action == 'register':
        OUT.mkdir(exist_ok=False)
        save(OUT/'preregistration.json', plan)
    print(json.dumps(dict(plan, gpu_calls=0)), flush=True)


if __name__ == '__main__':
    main()
