"""Save and verify the final SH0 triangle-local model for named-pose replay."""

from pathlib import Path

import numpy as np
from plyfile import PlyData

from . import runtime as rt
from .training import _frozen_smpl_surface, named_pose_layout


SCHEMA = 'shuolee-smplx-bound-v1'
LOCAL_PLY = 'bound-local.ply'
MANIFEST = 'bound-replay.json'
LOCAL_FIELDS = ('x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
                'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'binding_0')
RAW_PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')
SOURCE_FILES = (
    'src/shuolee_portrait/bound_replay.py', 'src/shuolee_portrait/training.py',
    'src/avatar_core/scene/gaussian_model.py', 'src/avatar_core/scene/smplx_gaussian_model.py',
    'src/avatar_core/scene/cameras.py', 'src/avatar_core/utils/graphics_utils.py',
    'src/avatar_core/gaussian_renderer/__init__.py',
    'src/avatar_core/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py',
    'src/avatar_core/submodules/diff-gaussian-rasterization/rasterize_points.cu',
    'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu',
    'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu',
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_bound_ply(path, triangle_count):
    """Validate stock SH0 PLY and exact integer triangle IDs before CUDA load."""
    ply = PlyData.read(path, mmap=False)
    _require(not ply.text and [element.name for element in ply.elements] == ['vertex'],
             'Expected a binary stock vertex-only bound Gaussian PLY.')
    vertices = ply['vertex'].data
    _require(len(vertices) > 0 and vertices.dtype.names == LOCAL_FIELDS and
             all(vertices[name].dtype.kind == 'f' and vertices[name].dtype.itemsize == 4 and
                 np.isfinite(vertices[name]).all() for name in LOCAL_FIELDS),
             'Incomplete, nonfinite or non-SH0 bound Gaussian fields.')
    ids = vertices['binding_0']
    _require(np.all((ids >= 0) & (ids < triangle_count) & (ids == np.floor(ids))),
             'Bound triangle IDs are not exact in-range indices.')
    ids = ids.astype(np.int64)
    counts = np.bincount(ids, minlength=triangle_count)
    _require(np.all(counts > 0), 'The bound Gaussian snapshot lost an original triangle.')
    return ids, counts


def _source_hashes():
    return {name: rt.sha(rt.PROJECT/name) for name in SOURCE_FILES}


def _inputs(root, snapshot, asset_path, manifest=None):
    paths = dict(fitted=root/'fit/fitted.npz', training_rows=root/'data/training-inputs.json',
                 reference_mesh=snapshot/'reference-mesh.npz', world_ply=snapshot/'world.ply',
                 native_ply_parity=snapshot/'native-ply-parity.json')
    pins = {name: rt.sha(path) for name, path in paths.items()}
    asset_hash, source_hashes = rt.sha(asset_path), _source_hashes()
    if manifest is not None:
        _require(manifest.get('inputs_sha256') == pins and manifest.get('asset_sha256') == asset_hash
                 and manifest.get('source_files_sha256') == source_hashes,
                 'A bound replay input, SMPL-X asset or renderer source changed.')
    with np.load(paths['fitted'], allow_pickle=False) as archive:
        fit = dict(archive)
    with np.load(paths['reference_mesh'], allow_pickle=False) as archive:
        mesh = dict(archive)
    rows = rt.read(paths['training_rows'])['records']
    layout = named_pose_layout(fit, rows)
    parity = rt.read(paths['native_ply_parity'])
    reference, slot = layout['reference_frame_id'], layout['reference_pose_index']
    _require(layout['named'] and parity.get('passed') is True and parity.get('sha256') == pins['world_ply']
             and parity.get('mesh_sha256') == pins['reference_mesh'] and
             parity.get('reference_frame_id') == mesh['reference_frame_id'].item() == reference and
             parity.get('reference_pose_index') == mesh['reference_pose_index'].item() == slot and
             np.array_equal(mesh['faces'], fit['faces']),
             'The terminal world PLY, saved offset and named reference mesh are not bound together.')
    frames = [dict(frame=row['frame'], pose_index=pose, source_frame_index=row['source_frame_index'],
                   source_sha256=row['source_sha256'], mask_sha256=row['mask_sha256'])
              for row, pose in zip(rows, layout['pose_indices'], strict=True)]
    facts = dict(inputs_sha256=pins, asset_sha256=asset_hash, source_files_sha256=source_hashes,
                 reference_frame_id=reference, reference_pose_index=slot, frames=frames)
    if manifest is not None:
        _require(all(manifest.get(key) == value for key, value in facts.items()),
                 'Saved frame-to-pose identities changed.')
    return fit, mesh, facts


def _geometry(model):
    return model.get_xyz, model.get_rotation, model.get_scaling


def _compare_geometry(actual, expected):
    import torch

    centers, rotations, scales = actual
    other_centers, other_rotations, other_scales = expected
    _require(all(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
                 for a, b in zip(actual, expected, strict=True)) and
             torch.all(scales > 0) and torch.all(other_scales > 0) and
             torch.all(rotations.norm(dim=1) > .99) and torch.all(other_rotations.norm(dim=1) > .99),
             'Invalid reconstructed Gaussian centers, quaternion rotations or scales.')
    # Quaternion sign is immaterial. Compare whole quaternions under both signs.
    rotation_error = torch.minimum((rotations-other_rotations).abs().amax(dim=1),
                                   (rotations+other_rotations).abs().amax(dim=1))
    errors = dict(center_max_abs_m=float((centers-other_centers).abs().max()),
                  rotation_max_abs=float(rotation_error.max()),
                  scale_max_relative=float(((scales-other_scales).abs()/other_scales).max()))
    _require(errors['center_max_abs_m'] < 2e-6 and errors['rotation_max_abs'] < 2e-5 and
             errors['scale_max_relative'] < 2e-5, f'Bound Gaussian replay differs: {errors}')
    return errors


def _motion_from_reference(posed, reference):
    import torch

    xyz, rotations, scales = posed
    ref_xyz, ref_rotations, ref_scales = reference
    distance = (xyz-ref_xyz).norm(dim=1)
    q = torch.nn.functional.normalize(rotations.double(), dim=1)
    r = torch.nn.functional.normalize(ref_rotations.double(), dim=1)
    q = torch.where((q*r).sum(dim=1, keepdim=True) < 0, -q, q)
    # Unit-quaternion chord ratio gives half of the quaternion-space angle.
    angle = torch.rad2deg(4*torch.atan2((q-r).norm(dim=1), (q+r).norm(dim=1)))
    relative_scale = ((scales-ref_scales).abs()/ref_scales).amax(dim=1)
    return dict(center_motion_from_reference_median_m=float(distance.median()),
                center_motion_from_reference_max_m=float(distance.max()),
                rotation_motion_from_reference_median_deg=float(angle.median()),
                rotation_motion_from_reference_max_deg=float(angle.max()),
                scale_max_axis_relative_change_from_reference_median=float(relative_scale.median()),
                scale_max_axis_relative_change_from_reference_max=float(relative_scale.max()))


def _load_model(path, snapshot, fit, mesh, facts, asset_path):
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.smplx_gaussian_model import SMPLXGaussianModel

    ids, counts = inspect_bound_ply(path, len(fit['faces']))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    base, linear, clothing, _, replay_error = _frozen_smpl_surface(fit, asset_path)
    offset = mesh['static_offset']
    _require(offset.shape == tuple(base.shape[1:]) and offset.dtype == np.float32 and np.isfinite(offset).all()
             and not np.any(offset[~clothing]), 'Terminal shared offset violates the fitted clothing boundary.')
    with torch.no_grad():
        base, linear = base.cuda(), linear.cuda()
        delta = torch.from_numpy(offset).cuda()[..., None]
        # Preserve the stock training evaluation order for each named pose.
        sequence = torch.stack([v + (A @ delta).squeeze(-1) for v, A in zip(base, linear, strict=True)])
        slot = facts['reference_pose_index']
        reference_mesh = torch.as_tensor(mesh['vertices_world'], device=sequence.device)
        _require(sequence[slot].shape == reference_mesh.shape and torch.isfinite(sequence).all(),
                 'Invalid trained mesh replay.')
        mesh_error = float((sequence[slot]-reference_mesh).abs().max())
        _require(mesh_error < 2e-6, f'Terminal reference mesh replay differs by {mesh_error} m.')
        model = SMPLXGaussianModel(0)
        model.vertices_sequence = sequence
        model.faces = torch.as_tensor(fit['faces'], dtype=torch.long, device=sequence.device)
        model.num_timesteps = len(sequence)
        model.load_ply(str(path))
        model.binding_counter = torch.as_tensor(counts, dtype=torch.int32, device=sequence.device)
        _require(model.optimizer is None and model.active_sh_degree == 0 and
                 torch.equal(model.binding.long(), torch.as_tensor(ids, device=sequence.device)),
                 'Stock PLY loader changed the frozen binding or SH degree.')
        for name in RAW_PARAMETERS:
            getattr(model, name).requires_grad_(False)
        model.select_mesh_by_timestep(slot)
        world = GaussianModel(0)
        world.load_ply(str(snapshot/'world.ply'))
        _require(world.binding is None and torch.equal(model.get_features, world.get_features) and
                 torch.equal(model.get_opacity, world.get_opacity),
                 'Reference world PLY changed colors, opacity or binding.')
        reference_parity = _compare_geometry(_geometry(model), _geometry(world))
    return model, dict(fit_replay_max_abs_m=replay_error, reference_mesh_max_abs_m=mesh_error,
                       reference_world_ply=reference_parity)


def save_bound_snapshot(root, snapshot, live_model, *, asset_path=None):
    """Save local PLY and prove each named transform against the live terminal model."""
    import torch

    root, snapshot = Path(root), Path(snapshot)
    asset = Path(asset_path) if asset_path is not None else rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    path, receipt = snapshot/LOCAL_PLY, snapshot/MANIFEST
    temporary = path.with_suffix('.ply.tmp')
    if any(item.exists() for item in (path, temporary, receipt)):
        raise FileExistsError('A bound Gaussian snapshot already exists.')
    _require(live_model.max_sh_degree == live_model.active_sh_degree == 0 and live_model.binding is not None,
             'Only the live SH0 triangle-bound model can be saved.')
    fit, mesh, facts = _inputs(root, snapshot, asset)
    live_model.save_ply(str(temporary))
    reloaded, geometry = _load_model(temporary, snapshot, fit, mesh, facts, asset)
    with torch.no_grad():
        _require(torch.equal(live_model.binding.long(), reloaded.binding.long()) and
                 torch.equal(live_model.faces, reloaded.faces) and
                 all(torch.equal(getattr(live_model, name), getattr(reloaded, name)) for name in RAW_PARAMETERS),
                 'The local PLY roundtrip changed a bound parameter, triangle ID or topology.')
        original_slot = live_model.timestep
        reference = _geometry(reloaded)
        measurements = []
        try:
            for item in facts['frames']:
                slot = item['pose_index']
                live_model.select_mesh_by_timestep(slot)
                reloaded.select_mesh_by_timestep(slot)
                posed = _geometry(reloaded)
                errors = _compare_geometry(posed, _geometry(live_model))
                measurements.append(dict(frame=item['frame'], pose_index=slot, **errors,
                                         **_motion_from_reference(posed, reference)))
        finally:
            live_model.select_mesh_by_timestep(original_slot)
            reloaded.select_mesh_by_timestep(facts['reference_pose_index'])
    geometry['live_to_reloaded'] = measurements
    manifest = dict(schema=SCHEMA, bound_ply=LOCAL_PLY, bound_ply_sha256=rt.sha(temporary),
                    points=len(reloaded.binding), sh_degree=0, **facts, geometry_parity=geometry,
                    optimizer_updates=0, quality_accepted=False,
                    coordinates='Triangle-local PLY, replayed in the fitted model-conditioned world; '
                                'camera-relative inputs do not establish physical extrinsics.')
    temporary.rename(path)
    rt.save(receipt, manifest)
    return reloaded, manifest


def load_bound_snapshot(root, snapshot, *, asset_path=None):
    """Reload a previously verified bound snapshot using pinned source and inputs."""
    root, snapshot = Path(root), Path(snapshot)
    asset = Path(asset_path) if asset_path is not None else rt.ASSETS/'smplx/SMPLX_NEUTRAL.npz'
    manifest = rt.read(snapshot/MANIFEST)
    path = snapshot/LOCAL_PLY
    _require(manifest.get('schema') == SCHEMA and manifest.get('sh_degree') == 0 and
             manifest.get('bound_ply') == LOCAL_PLY and rt.sha(path) == manifest.get('bound_ply_sha256'),
             'Missing or altered SH0 bound snapshot.')
    fit, mesh, facts = _inputs(root, snapshot, asset, manifest)
    model, _ = _load_model(path, snapshot, fit, mesh, facts, asset)
    _require(len(model.binding) == manifest.get('points'), 'Bound snapshot point count differs.')
    return model, manifest
