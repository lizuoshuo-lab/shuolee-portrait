"""Intake exact archived SH0 scenes for the same-person R28 transfer."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import runtime as rt
from .bound_replay import RAW_PARAMETERS, _compare_geometry, _geometry, inspect_bound_ply
from .training import camera, named_pose_layout


LEGACY_SCHEMA = 'shuolee-r28-legacy-bound-source/1'
ROW_IDENTITY = ('frame', 'source_frame_index', 'source_sha256', 'mask_sha256', 'K', 'w2c', 'wh')


def _require(check, message):
    if not check:
        raise ValueError(message)


def _pinned(record):
    path = Path(record['file'])
    _require(path.is_absolute() and path.is_file() and rt.sha(path) == record['sha256'],
             f'An archived R28 source file changed: {path}')
    return path


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def _row_ids(rows):
    return [row['frame'] for row in rows]


def _row_match(actual, expected, keys=ROW_IDENTITY):
    return all(actual[key] == expected[key] for key in keys)


def _load_intake(baseline, descriptor_path):
    """Validate archived source identities on CPU before constructing a GPU scene."""
    baseline, descriptor_path = Path(baseline).resolve(strict=True), Path(descriptor_path).resolve(strict=True)
    descriptor = rt.read(descriptor_path)
    _require(descriptor.get('schema') == LEGACY_SCHEMA and Path(descriptor['source_root']).resolve() == baseline,
             'The legacy scene descriptor does not belong to this source run.')
    main, held = descriptor['selected_records'], descriptor['heldout_records']
    _require(bool(main) and not descriptor['auxiliary_records'] and
             len(set(_row_ids(main+held))) == len(main)+len(held),
             'The archived main and reserved native rosters are not disjoint.')
    local = _pinned(descriptor['local_ply'])
    world = _pinned(descriptor['world_ply'])
    snapshot = local.parent
    _require(local == snapshot/'bound-local.ply' and world == snapshot/'world.ply' and
             snapshot.parent == baseline/'gaussian-run', 'Legacy PLY files do not share an archived snapshot.')
    manifest_path = _pinned(descriptor['bound_manifest'])
    manifest = rt.read(manifest_path)
    terminal = rt.read(_pinned(descriptor['terminal']))
    _require(manifest_path == snapshot/'bound-replay.json' and manifest['sh_degree'] == 0
             and manifest['bound_ply_sha256'] == descriptor['local_ply']['sha256']
             and manifest['inputs_sha256']['world_ply'] == descriptor['world_ply']['sha256']
             and terminal['format_passed'] is True and terminal['state'] == 'completed'
             and terminal['points'] == manifest['points'], 'Wrong legacy SH0 export identity.')
    fitted_path, training_path = (_pinned(descriptor[key]) for key in ('fit','training_inputs'))
    _require(fitted_path == baseline/'fit/fitted.npz' and training_path == baseline/'data/training-inputs.json' and
             descriptor['fit']['sha256'] == manifest['inputs_sha256']['fitted'] and
             descriptor['training_inputs']['sha256'] == manifest['inputs_sha256']['training_rows'],
             'The archived fitted mesh or native training roster changed.')
    fit, original = _arrays(fitted_path), rt.read(training_path)['records']
    layout = named_pose_layout(fit, original)
    reference = layout['reference_pose_index']
    _require(_row_ids(main) == _row_ids(original) and all(_row_match(a,b) for a,b in zip(main,original,strict=True))
             and manifest['reference_pose_index'] == reference
             and manifest['reference_frame_id'] == layout['reference_frame_id'],
             'Legacy selected pose, source or calibrated native camera differs.')
    metadata = rt.read(baseline/'run.json')
    asset = _pinned(descriptor['asset'])
    source_video = _pinned(descriptor['source_video'])
    _require(source_video == Path(metadata['video']) and descriptor['source_video']['sha256'] == metadata['video_sha256'] and
             asset == Path(metadata['assets'])/'smplx/SMPLX_NEUTRAL.npz' and
             descriptor['asset']['sha256'] == manifest['asset_sha256'], 'Legacy SMPL-X asset changed.')
    pose_path = _pinned(descriptor['pose_mesh'])
    replay_path = _pinned(descriptor['selected_pose_replay'])
    replay = rt.read(replay_path)
    _require(replay_path.parent == snapshot/'posed-replay' and
             replay['bound_replay_sha256'] == descriptor['bound_manifest']['sha256'] and
             pose_path == replay_path.parent/replay['mesh_file'] and
             replay['mesh_sha256'] == descriptor['pose_mesh']['sha256'] and
             len(replay['records']) == len(main), 'Legacy posed replay is not from this exact bound model.')
    pose = _arrays(pose_path)
    ref_path = _pinned(descriptor['reference_mesh'])
    ref = _arrays(ref_path)
    vertices = pose['vertices_world']
    faces = fit['faces']
    _require(ref_path == snapshot/'reference-mesh.npz' and manifest['inputs_sha256']['reference_mesh'] ==
             descriptor['reference_mesh']['sha256'] and np.array_equal(pose['faces'], faces)
             and np.array_equal(ref['faces'], faces) and pose['frame_ids'].tolist() == _row_ids(main)
             and pose['pose_indices'].tolist() == list(range(len(main)))
             and ref['reference_frame_id'].item() == layout['reference_frame_id']
             and ref['reference_pose_index'].item() == reference
             and vertices.shape == (len(main), len(ref['vertices_world']), 3)
             and vertices.dtype == np.float32 and np.isfinite(vertices).all()
             and np.array_equal(vertices[reference], ref['vertices_world']),
             'Legacy posed vertices do not reproduce their exact archived reference and topology.')
    owners, counts = inspect_bound_ply(local, len(faces))
    _require(len(owners) == manifest['points'], 'Legacy bound Gaussian row count changed.')
    native_inputs, native_observations = (_pinned(descriptor[key]) for key in ('native_inputs','native_observations'))
    native = {row['frame']: row for row in rt.read(native_inputs)['records']}
    for slot, row in enumerate(main+held):
        _require(row['frame'] in native and _row_match(row,native[row['frame']]) and row['pose_index'] == slot
                 and rt.sha(row['image']) == row['source_sha256'] and rt.sha(row['mask']) == row['mask_sha256'],
                 f'A legacy native source frame, mask or pose changed: {row["frame"]}')
        if slot < len(main):
            saved = replay['records'][slot]
            _require(_pinned(row['baseline_rgb']) == replay_path.parent/saved['image'] and
                     row['baseline_rgb']['sha256'] == saved['image_sha256'] and
                     _row_match(row,saved,ROW_IDENTITY[:4]) and
                     _row_match(row,saved['camera'],('frame','K','w2c','wh')) and saved['pose_index'] == slot,
                     f'Native archived RGB replay changed: {row["frame"]}')
    held_vertices = None
    if held:
        _pinned(descriptor['formal_review'])
        held_fit = _arrays(_pinned(descriptor['heldout_formal_fit']))
        held_mesh = _arrays(_pinned(descriptor['heldout_pose_mesh']))
        held_vertices = held_mesh['vertices_world']
        pose_mask_sha = []
        for row in held:
            _pinned(row['baseline_panel'])
            archive = row.get('archived_fit_mask')
            if archive is not None:
                _pinned(archive)
            pose_mask_sha.append(archive['sha256'] if archive is not None else row['mask_sha256'])
        _require(held_fit['frame_ids'].tolist() == _row_ids(held) and
                 np.array_equal(held_fit['faces'], faces) and np.array_equal(held_fit['shared_betas'], fit['shared_betas']) and
                 np.array_equal(held_fit['K_integer_centers'], [row['K_integer_centers'] for row in held]) and
                 np.array_equal(held_fit['w2c'], [row['w2c'] for row in held]) and
                 np.array_equal(held_fit['image_sha256'], [row['source_sha256'] for row in held]) and
                 np.array_equal(held_fit['mask_sha256'], pose_mask_sha) and
                 held_mesh['frame_ids'].tolist() == _row_ids(held) and np.array_equal(held_mesh['faces'], faces) and
                 held_mesh['pose_indices'].tolist() == list(range(len(main),len(main)+len(held))) and
                 held_vertices.shape == (len(held),vertices.shape[1],3) and held_vertices.dtype == np.float32 and
                 np.isfinite(held_vertices).all(),
                 'Legacy held-out fit changed identity, native camera or shared shape.')
    return SimpleNamespace(baseline=baseline, descriptor_path=descriptor_path, descriptor=descriptor,
        source_snapshot=snapshot, source_manifest=manifest, local=local, world=world, vertices=vertices,
        faces=faces, owners=owners, counts=counts, asset=asset, fit=fit, held_vertices=held_vertices,
        reference_mesh=ref, reference=reference, main=main, held=held, native_inputs=native_inputs,
        native_observations=native_observations)


@torch.no_grad()
def prepare_legacy_source(baseline, descriptor_path):
    """Load a source-bound model, archived poses and disjoint evaluation cameras."""
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    source = _load_intake(baseline, descriptor_path)
    model = SMPLXGaussianModel(0)
    vertices = torch.from_numpy(source.vertices).cuda()
    if source.held_vertices is not None:
        vertices = torch.cat((vertices,torch.from_numpy(source.held_vertices).cuda()))
    model.vertices_sequence = vertices
    model.faces = torch.from_numpy(source.faces).long().cuda()
    model.num_timesteps = len(vertices)
    model.load_ply(str(source.local))
    model.binding = model.binding.long()
    model.binding_counter = torch.from_numpy(source.counts).int().cuda()
    _require(torch.equal(model.binding.cpu(),torch.from_numpy(source.owners)) and model.optimizer is None,
             'Stock PLY loading changed archive row order or attached an optimizer.')
    for name in RAW_PARAMETERS:
        getattr(model,name).requires_grad_(False)
    model.select_mesh_by_timestep(source.reference)
    world = GaussianModel(0)
    world.load_ply(str(source.world))
    _require(model.active_sh_degree == world.active_sh_degree == 0 and world.binding is None
             and torch.equal(model.get_features,world.get_features)
             and torch.equal(model.get_opacity,world.get_opacity),
             'The native reference world PLY changed archived color or opacity.')
    parity = _compare_geometry(_geometry(model),_geometry(world))
    model.densification_ids = torch.arange(len(model.binding),dtype=torch.long,device=model.binding.device)
    model.next_densification_id = len(model.binding)
    model.spatial_lr_scale = 1.  # The archived stock SH0 route calls create_from_pcd(None, 1.).
    rows = source.main+source.held
    cameras, camera_records = map(list,zip(*(camera(row,i,pose_index=i) for i,row in enumerate(rows)),strict=True))
    return SimpleNamespace(
        model=model,rows=rows,cameras=cameras,camera_records=camera_records,reference=source.reference,
        source_snapshot=source.source_snapshot,source_manifest=source.source_manifest,
        source_raw=None,source_state_sha256=None,auxiliary=None,auxiliary_records=[],fit=source.fit,
        baseline=source.baseline,main_count=len(source.main),holdout_count=len(source.held),aux_count=0,legacy=True,
        legacy_source_sha256=rt.sha(source.descriptor_path),legacy_source=source.descriptor,
        legacy_reference_parity=parity,native_inputs=source.native_inputs,native_observations=source.native_observations,
        reserved_holdouts=source.descriptor.get('reserved_holdouts',_row_ids(source.held)),
        unavailable_holdouts=source.descriptor.get('unavailable_holdouts',[]))
