"""Apply an auditable native head refit to a named official SMPL-X sequence."""

from pathlib import Path
from shutil import copyfile

import numpy as np

from . import runtime as rt
from .heldout_pose import fit_holdouts, verify_holdouts


SOURCE_NAMES = ('fitted.npz', 'joint-candidates.npz', 'official-fit-result.json')
POSE_FIELDS = {'vertices_camera', 'vertices_world', 'joints_camera', 'joints_projected_native', 'body_pose_axis_angle'}


def _write_arrays(path, values):
    temporary = path.with_suffix('.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _check_fit_delta(source, active, allowed_frames):
    with np.load(source, allow_pickle=False) as archive:
        previous = dict(archive)
    if set(previous) != set(active):
        raise ValueError('A native head refit must preserve the official fitted array contract.')
    ids = previous['frame_ids'].tolist()
    unchanged_slots = [i for i, frame in enumerate(ids) if frame not in allowed_frames]
    for key, original in previous.items():
        current = active[key]
        if current.shape != original.shape or current.dtype != original.dtype:
            raise ValueError('Native refit changed array type or shape: '+key)
        if key not in POSE_FIELDS and not np.array_equal(current, original):
            raise ValueError('Native refit changed frozen identity, camera or observed landmarks: '+key)
        if key in POSE_FIELDS and (not np.isfinite(current).all()
                or not np.array_equal(current[unchanged_slots], original[unchanged_slots])):
            raise ValueError('Refit affected a frame outside the frozen training scope: '+key)
    local = np.ones(63, dtype=bool)
    local[33:36] = False  # Neck, SMPL-X joint 12.
    local[42:45] = False  # Head, SMPL-X joint 15.
    if not np.array_equal(previous['body_pose_axis_angle'][:, local], active['body_pose_axis_angle'][:, local]):
        raise ValueError('Native head refinement changed non-head body articulation.')
    return [frame for i, frame in enumerate(ids)
            if not np.array_equal(previous['body_pose_axis_angle'][i], active['body_pose_axis_angle'][i])]


def prepare_head_fit(root, *, assets, auxiliary_frames):
    """Freeze source and held-out provenance before Gaussian optimization starts."""
    from .head_alignment import fit_head_alignment
    from .training import verify_official_fit

    root, assets = Path(root).resolve(strict=True), Path(assets).resolve(strict=True)
    if any((root/name).exists() for name in ('gaussian-run', 'export', 'result.json', 'status.json')):
        raise ValueError('Prepare native head fits before starting Gaussian training.')
    rt.configure(assets, root)
    verify_official_fit(root)
    fit_dir = root/'fit'
    official_path = fit_dir/'official-fit-result.json'
    official = rt.read(official_path)
    selection = official['selection']
    held = set(selection['retained_holdouts'])
    main = set(selection['selected_frames'])
    aux = list(auxiliary_frames)
    if len(aux) != len(set(aux)) or set(aux) & (held | main):
        raise ValueError('The head fit roster mixes held-out, body and face-only observations.')
    with np.load(fit_dir/'joint-candidates.npz', allow_pickle=False) as candidates:
        if not set(aux) <= set(candidates['frame_ids'].tolist()):
            raise ValueError('Unknown auxiliary pose identity.')
    stage = fit_dir/'head-alignment'
    original = stage/'source'
    original.mkdir(parents=True)
    for name in SOURCE_NAMES:
        copyfile(fit_dir/name, original/name)
    source = {name: rt.sha(original/name) for name in SOURCE_NAMES}
    cfg = rt.read(root/'config.json')
    base, candidates, report = fit_head_alignment(root, assets=assets,
        official_root=cfg['official_smplifyx']['source_root'], auxiliary_frames=aux)
    main_changed = _check_fit_delta(original/'fitted.npz', base, main)
    aux_changed = _check_fit_delta(original/'joint-candidates.npz', candidates, main | set(aux))
    lookup = {frame: i for i, frame in enumerate(candidates['frame_ids'].tolist())}
    slots = [lookup[frame] for frame in base['frame_ids'].tolist()]
    for key in ('vertices_camera', 'joints_camera', 'joints_projected_native', 'body_pose_axis_angle'):
        if not np.array_equal(base[key], candidates[key][slots]):
            raise ValueError('The main and candidate fits have different refined head poses: '+key)
    plan = fit_holdouts(root, assets, stage, selection, base['shared_betas'])
    _write_arrays(fit_dir/'fitted.npz', base)
    _write_arrays(fit_dir/'joint-candidates.npz', candidates)
    path = stage/'result.json'
    active = dict(fit_sha256=rt.sha(fit_dir/'fitted.npz'), candidate_sha256=rt.sha(fit_dir/'joint-candidates.npz'))
    rt.save(path, dict(schema='shuolee-native-head-fit-stage/1', state='complete', fitting=report,
        source_sha256=source, active=active, frozen_auxiliary_frames=aux, heldout_frames=plan['frames'],
        selected_frames=selection['selected_frames'], changed_main_frames=main_changed,
        changed_candidate_frames=aux_changed, holdout_plan_sha256=rt.sha(stage/'holdout-plan.json'),
        note='Archived pose embeddings document the original official fit. Revised local head rotation '
             'is replayed directly from body_pose_axis_angle. Shared identity and camera transforms are frozen.'))
    official.update(fit_sha256=active['fit_sha256'], native_head_alignment=dict(file='fit/head-alignment/result.json',
        sha256=rt.sha(path), **active))
    rt.save(official_path, official)
    verify_official_fit(root)
    verify_holdouts(root)
    return rt.read(path)


def verify_head_fit(root, official):
    """Bind the refit and both immutable source fit archives to the active result."""
    descriptor = official['native_head_alignment']
    stage = root/'fit/head-alignment'
    path = stage/'result.json'
    if descriptor['file'] != str(path.relative_to(root)) or descriptor['sha256'] != rt.sha(path):
        raise ValueError('Missing pinned native head refinement.')
    result = rt.read(path)
    if (result['state'] != 'complete'
            or result['active']['fit_sha256'] != descriptor['fit_sha256']
            or result['active']['candidate_sha256'] != descriptor['candidate_sha256']
            or descriptor['fit_sha256'] != official['fit_sha256']
            or descriptor['candidate_sha256'] != rt.sha(root/'fit/joint-candidates.npz')
            or set(result['source_sha256']) != set(SOURCE_NAMES)
            or any(rt.sha(stage/'source'/name) != digest for name, digest in result['source_sha256'].items())
            or rt.sha(stage/'holdout-plan.json') != result['holdout_plan_sha256']):
        raise ValueError('Native head refinement provenance differs from the staged fit.')
    return result
