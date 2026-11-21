"""Fit and render pose-conditioned views that do not participate in training."""

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
from PIL import Image

from . import runtime as rt
from .camera_world_units import normalize_camera_w2c
from .official_sequence import _fit_cpu_job, _init_cpu_fit, assemble, checked_image, pelvis_basis
from .official_smplifyx import adapt_rtmw_observations, create_models


def _source_rows(root, held, camera_contract):
    rows = rt.read(root/'data/inputs.json')['records']
    lookup = {row['frame']: row for row in rows}
    if len(lookup) != len(rows) or len(set(held)) != len(held) or any(frame not in lookup for frame in held):
        raise ValueError('Missing or duplicate reserved source frame.')
    source = [lookup[frame] for frame in held]
    units = normalize_camera_w2c([row['w2c'] for row in source], camera_contract)
    result = []
    for i, row in enumerate(source):
        checked_image(row)
        if rt.sha(row['mask']) != row['mask_sha256']:
            raise ValueError('Held-out mask bytes changed: '+row['frame'])
        result.append(dict(row, w2c=units.w2c_meters[i].tolist(), source_w2c=units.source_w2c[i].tolist()))
    return result, units


def fit_holdouts(root, assets, directory, selection, shared_betas):
    """Use the pinned native per-frame SMPLify-X recipe with fixed beta and K."""
    config = rt.read(root/'config.json')
    training = rt.read(root/'data/training-inputs.json')
    held = selection['retained_holdouts']
    if not held or set(held) & set(selection['selected_frames']):
        raise ValueError('Held-out pose fitting needs reserved frames outside the training roster.')
    rows, units = _source_rows(root, held, training['source_camera_length_contract'])
    observations = root/'data/observations.npz'
    with np.load(observations, allow_pickle=False) as archive:
        names = archive['frame_ids'].tolist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate source observations.')
        slots = [names.index(frame) for frame in held]
        targets, _ = adapt_rtmw_observations(
            *(archive[name][slots] for name in ('keypoints', 'scores', 'boxes', 'flip_disagreement')))
    K = np.asarray([row['K_integer_centers'] for row in rows], dtype=np.float64)
    for i, row in enumerate(rows):
        corner = K[i].copy()
        corner[:2, 2] += .5
        if not np.array_equal(row['K'], corner):
            raise ValueError('Held-out native pixel convention changed.')
        outside = ((targets[i, :, :2] < -.5) | (targets[i, :, :2] >= np.asarray(row['wh'])-.5)).any(axis=1)
        targets[i, outside, 2] = 0
    source_root = config['official_smplifyx']['source_root']
    work = directory/'holdout-official'
    work.mkdir()
    jobs = [(row, work/row['frame'], targets[i:i+1], K[i], shared_betas, True, None)
            for i, row in enumerate(rows)]
    policy = dict(schema='shuolee-heldout-local-pose-policy/1', frames=held, shared_shape_frozen=True,
        camera_frozen=True, selection_frozen=True, gaussian_updates=0,
        shared_identity_updates=0, official_smplifyx_source_root=source_root,
        method='Independent stock five-stage native-keypoint fit with frozen shared betas and native K. '
               'Per-frame subject placement and articulation are estimated from that frame’s 2D observations.',
        use_rgb_for_gaussian_fitting=False, use_final_render_for_pose_fit=False,
        prior_pose_source='stock SMPLify-X initialization', camera_length_contract=dict(units.contract))
    policy_path = directory/'holdout-policy.json'
    rt.save(policy_path, policy)
    with ProcessPoolExecutor(max_workers=min(4, len(jobs)), mp_context=get_context('spawn'), initializer=_init_cpu_fit,
                             initargs=(source_root, assets)) as pool:
        outputs = list(pool.map(_fit_cpu_job, jobs))
    body, _, _ = create_models(official_root=source_root, assets=assets, device='cpu', static_topology=False)
    winners = {row['frame']: pair[0] for row, pair in zip(rows, outputs, strict=True)}
    array, _ = assemble(dict(rows=rows, ids=held, K=K, units=units, targets=targets,
                             reference_frame=held[0]), winners, shared_betas, pelvis_basis(body))
    path = directory/'holdout-poses.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **array)
    records = []
    for i, (row, pair) in enumerate(zip(rows, outputs, strict=True)):
        records.append(dict(frame=row['frame'], source_frame_index=row['source_frame_index'],
            source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'], K_integer_centers=row['K_integer_centers'],
            K=row['K'], wh=row['wh'], w2c=row['w2c'], source_w2c=row['source_w2c'],
            official_fit=dict(pair[1], path=str(Path(pair[1]['path']).relative_to(root))),
            pose_index=i, role='pose_conditioned_holdout'))
    plan = dict(state='frozen_before_gaussian', policy_sha256=rt.sha(policy_path), poses_sha256=rt.sha(path),
        native_inputs_sha256=rt.sha(root/'data/inputs.json'), observations_sha256=rt.sha(observations),
        training_inputs_sha256=rt.sha(root/'data/training-inputs.json'), records=records, **policy)
    rt.save(directory/'holdout-plan.json', plan)
    return plan


def verify_holdouts(root):
    directory = root/'fit/head-alignment'
    plan = rt.read(directory/'holdout-plan.json')
    selection = rt.read(root/'fit/selection.json')
    ids = selection['retained_holdouts']
    training_path = root/'data/training-inputs.json'
    contract = rt.read(training_path)['source_camera_length_contract']
    rows, _ = _source_rows(root, ids, contract)
    expected = dict(native_inputs_sha256=root/'data/inputs.json', observations_sha256=root/'data/observations.npz',
        training_inputs_sha256=training_path, policy_sha256=directory/'holdout-policy.json',
        poses_sha256=directory/'holdout-poses.npz')
    if (plan['state'] != 'frozen_before_gaussian' or plan['frames'] != ids
            or plan['gaussian_updates'] != 0 or plan['shared_identity_updates'] != 0
            or any(plan[key] != rt.sha(path) for key, path in expected.items())
            or [row['frame'] for row in plan['records']] != ids):
        raise ValueError('Pose-conditioned holdout manifest changed.')
    for row, record in zip(rows, plan['records'], strict=True):
        keys = ('frame', 'source_frame_index', 'source_sha256', 'mask_sha256', 'K_integer_centers',
                'K', 'wh', 'w2c', 'source_w2c')
        item = record['official_fit']
        source = (root/item['path']).resolve(strict=True)
        if (any(row[key] != record[key] for key in keys) or not source.is_relative_to(directory.resolve())
                or item['sha256'] != rt.sha(source)):
            raise ValueError('Held-out fitted pose lost its native observation identity.')
    with np.load(expected['poses_sha256'], allow_pickle=False) as archive:
        fit = dict(archive)
    with np.load(root/'fit/fitted.npz', allow_pickle=False) as active:
        if (fit['frame_ids'].tolist() != ids or not np.array_equal(fit['shared_betas'], active['shared_betas'])
                or not np.array_equal(fit['faces'], active['faces'])
                or not np.array_equal(fit['K_integer_centers'], [row['K_integer_centers'] for row in rows])
                or not np.array_equal(fit['w2c'], [row['w2c'] for row in rows])):
            raise ValueError('Held-out identity, topology, camera or pose roster changed.')
    return plan, fit, rows


def render_holdouts(root, snapshot, replay):
    """Render a separate evaluation roster without mutating the training pose sequence."""
    import torch
    from gaussian_renderer import render
    from .bound_review_panels import _crop, _features, _mesh_panel
    from .training import _frozen_smpl_surface, camera, image

    model, manifest = replay
    plan, fit, rows = verify_holdouts(root)
    if model.optimizer is not None or model.active_sh_degree != 0:
        raise ValueError('Held-out rendering requires the frozen terminal model.')
    destination = snapshot/'heldout-replay'
    if destination.exists():
        raise FileExistsError(destination)
    base, linear, clothing, _, _ = _frozen_smpl_surface(fit, rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    mesh_path = snapshot/'reference-mesh.npz'
    if rt.sha(mesh_path) != manifest['inputs_sha256']['reference_mesh']:
        raise ValueError('Held-out renderer received a different frozen clothing offset.')
    with np.load(mesh_path, allow_pickle=False) as mesh:
        offset = mesh['static_offset']
    if np.any(offset[~clothing]):
        raise ValueError('Invalid nonclothing offset in held-out mesh replay.')
    device = model.binding.device
    sequence = model.vertices_sequence
    original_slot = model.timestep
    bg = torch.zeros(3, device=device)
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    records = []
    with TemporaryDirectory(prefix='.holdout-', dir=snapshot) as temporary, torch.no_grad():
        stage = Path(temporary)/'replay'
        stage.mkdir()
        vertices = (base + (linear @ torch.from_numpy(offset)[None, ..., None]).squeeze(-1)).to(device)
        model.vertices_sequence = torch.cat((sequence, vertices))
        try:
            for i, (row, record) in enumerate(zip(rows, plan['records'], strict=True)):
                slot = len(sequence)+i
                model.select_mesh_by_timestep(slot)
                cam, calibration = camera(row, i, pose_index=slot)
                rgb = render(cam, model, pipe, bg)['render']
                alpha = render(cam, model, pipe, bg, override_color=torch.ones_like(model.get_xyz))['render'][:1]
                if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
                    raise RuntimeError('Nonfinite held-out Gaussian render: '+row['frame'])
                white = rgb+(1-alpha)
                with Image.open(row['image']) as raw:
                    box = _crop(row)
                    original = raw.convert('RGB').crop(box)
                E = fit['w2c'][i]
                camera_vertices = vertices[i].cpu().numpy() @ E[:3, :3].T+E[:3, 3]
                panels = dict(black=image(rgb), white=image(white), alpha=image(alpha.expand_as(rgb)).convert('L'),
                    mesh=_mesh_panel(original, box, camera_vertices, fit['faces'], fit['K_integer_centers'][i]),
                    feature=_features(original, box, fit['observed_native_keypoints'][i], fit['joints_projected_native'][i]))
                files = {}
                for role, panel in panels.items():
                    name = f'{row["frame"]}-{role}.png'
                    panel.save(stage/name)
                    files[role] = dict(file=name, sha256=rt.sha(stage/name))
                records.append(dict(record, pose_index=slot, fitted_pose_index=i, camera=calibration,
                                    crop_xyxy=list(box), files=files))
        finally:
            model.vertices_sequence = sequence
            model.select_mesh_by_timestep(original_slot)
        mesh_file = stage/'meshes.npz'
        np.savez_compressed(mesh_file, vertices_world=vertices.cpu().numpy(), faces=fit['faces'],
                           frame_ids=fit['frame_ids'], pose_indices=np.asarray([row['pose_index'] for row in records]))
        result = dict(schema='shuolee-pose-conditioned-holdout-replay/1', state='completed', records=records,
            holdout_plan_sha256=rt.sha(root/'fit/head-alignment/holdout-plan.json'),
            holdout_pose_sha256=plan['poses_sha256'], bound_ply_sha256=manifest['bound_ply_sha256'],
            mesh_file=mesh_file.name, mesh_sha256=rt.sha(mesh_file), optimizer_updates=0,
            fully_unobserved_prediction=False, uses_frozen_clothing_offset=True)
        rt.save(stage/'result.json', result)
        stage.rename(destination)
    return result
