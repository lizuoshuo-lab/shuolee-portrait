"""Compare one continuous R21 trajectory on exact common Gaussian row IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
import torch


STEPS = (10000, 18000)
OPACITY_MIN = .02
FLAT_RATIO = 1.5


def _require(test, message):
    if not test:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _sha(path):
    with path.open('rb') as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pin(path, expected):
    actual = _sha(path)
    _require(actual == expected, f'Unexpected SHA-256 for {path}')
    return dict(file=str(path), sha256=actual)


def _array(item):
    tensor = item['tensor']
    _require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu',
             'Expected a CPU tensor in the frozen duration checkpoint.')
    return tensor.numpy()


def _rotation(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    length = np.linalg.norm(quaternion, axis=-1)
    _require(np.isfinite(length).all() and (length > 1e-12).all(), 'Invalid bound Gaussian quaternion.')
    w, x, y, z = (quaternion/length[:, None]).T
    return np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=1).reshape(-1, 3, 3)


def _normalize(vector):
    return vector/np.sqrt(np.maximum((vector*vector).sum(-1, keepdims=True), 1e-20))


def _reference_frame(mesh):
    vertices, faces = mesh['vertices_world'].astype(np.float64), mesh['faces']
    xyz = vertices[faces]
    edge, other = xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]
    x = _normalize(edge)
    normal = _normalize(np.cross(x, other))
    z = -_normalize(np.cross(normal, x))
    scale = (np.sqrt(np.maximum((edge*edge).sum(-1), 1e-20))+abs((z*other).sum(-1)))*.5
    return xyz.mean(1), np.stack((x, normal, z), axis=-1), np.maximum(scale, 1e-8)


def _world_ply_parity(path, item, mesh):
    vertices = PlyData.read(path, mmap=False)['vertex'].data
    owner = item['binding']
    _require(len(vertices) == len(owner), 'The raw native PLY has a different Gaussian row count.')
    centers, basis, _ = _reference_frame(mesh)
    scale = item['face_scale'][owner, None]
    xyz = (basis[owner] @ item['xyz'][..., None]).squeeze(-1)*scale+centers[owner]
    got_xyz = np.stack([vertices[axis] for axis in 'xyz'], axis=-1)
    xyz_error = float(np.max(abs(xyz-got_xyz)))
    sigma = np.exp(item['log_sigma'].astype(np.float64))*scale
    got_sigma = np.exp(np.stack([vertices[f'scale_{axis}'] for axis in range(3)], axis=-1).astype(np.float64))
    log_error = float(np.max(abs(np.log(sigma)-np.log(got_sigma))))
    wxyz = np.stack([vertices[f'rot_{axis}'] for axis in range(4)], axis=-1)
    world_axes = (basis[owner] @ _rotation(item['quaternion']))*sigma[:, None, :]
    ply_axes = _rotation(wxyz)*got_sigma[:, None, :]
    world_cov = world_axes @ world_axes.transpose(0, 2, 1)
    ply_cov = ply_axes @ ply_axes.transpose(0, 2, 1)
    covariance_error = float(np.max(np.linalg.norm(world_cov-ply_cov, axis=(1, 2)) /
                                    np.linalg.norm(world_cov, axis=(1, 2)).clip(1e-20)))
    color = np.stack([vertices[f'f_dc_{axis}'] for axis in range(3)], axis=-1)
    _require(xyz_error <= 2e-6 and log_error <= 2e-6 and covariance_error <= 2e-5
             and np.array_equal(vertices['opacity'], item['opacity_logit'])
             and np.array_equal(color, item['dc']),
             'Serialized row order or world PLY geometry differs from the bound optimizer state.')
    return dict(center_max_abs_m=xyz_error, log_sigma_max_abs=log_error,
                covariance_max_relative_frobenius=covariance_error, opacity_and_dc_bit_equal=True)


def _endpoint(out, step):
    directory = out/f'raw-step-{step:06d}'
    manifest = directory/'state.json'
    meta = _read(manifest)
    _require(meta['schema'] == 'shuolee-duration-endpoint-v1'
             and meta['step'] == meta['main_optimizer_updates'] == meta['scheduler_step'] == step,
             'The raw checkpoint update identity changed.')
    state_path = directory/'state.pt'
    pins = dict(manifest=dict(file=str(manifest), sha256=_sha(manifest)),
                state=_pin(state_path, meta['state_sha256']))
    for name in ('world.ply', 'reference-mesh.npz', 'attachment.json', 'native-ply-parity.json'):
        pins[name] = _pin(directory/name, meta['files_sha256'][name])
    bundle = torch.load(state_path, weights_only=True, map_location='cpu')
    state = bundle['state']
    _require(state['schema'] == 'shuolee-duration-checkpoint/1'
             and all(meta[key] == value for key, value in bundle['training'].items()),
             'The exact bound state is detached from its duration receipt.')
    draws = bundle['sampled_rows']
    _require(isinstance(draws, torch.Tensor) and draws.device.type == 'cpu' and draws.dtype == torch.uint8
             and tuple(draws.shape) == (step,) and int(draws.max()) < 24
             and hashlib.sha256(draws.numpy().tobytes()).hexdigest() == meta['source_draw_sha256']
             and hashlib.sha256(draws[:STEPS[0]].numpy().tobytes()).hexdigest()
             == meta['source_draw_prefix_10000_sha256'], 'The original native source draw trace differs.')
    row = state['parameters']
    value = {name: _array(row[name]['value']) for name in
             ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', 'static_offset')}
    ids = _array(state['model']['densification_ids'])
    binding = _array(state['model']['binding'])
    face_scale = _array(state['world']['reference_scale'])
    count = meta['points']
    _require(count > 0 and ids.dtype == binding.dtype == np.int64 and ids.shape == binding.shape == (count,)
             and (np.diff(ids) > 0).all() and ids[0] >= 0
             and value['_xyz'].shape == value['_scaling'].shape == (count, 3)
             and value['_rotation'].shape == (count, 4) and value['_opacity'].shape == (count, 1)
             and value['_features_dc'].shape == (count, 1, 3)
             and all(np.isfinite(item).all() for item in (*value.values(), face_scale)) and (face_scale > 0).all(),
             'Stable IDs, parameters or triangle scale are invalid.')
    with np.load(directory/'reference-mesh.npz', allow_pickle=False) as archive:
        mesh = dict(archive)
    _, _, scale_check = _reference_frame(mesh)
    scale_error = float(np.max(np.abs(scale_check-face_scale)))
    _require(mesh['reference_frame_id'].item() == meta['reference_frame_id']
             and len(face_scale) == len(mesh['faces']) and scale_error < 2e-6
             and (binding >= 0).all() and (binding < len(face_scale)).all(),
             'The face scale, named mesh or triangle binding changed.')
    parity = _read(directory/'native-ply-parity.json')
    _require(parity['passed'] and parity['sha256'] == pins['world.ply']['sha256']
             and parity['mesh_sha256'] == pins['reference-mesh.npz']['sha256'],
             'The raw native PLY does not bind the same reference surface.')
    item = dict(step=step, ids=ids, binding=binding, face_scale=face_scale, xyz=value['_xyz'],
                log_sigma=value['_scaling'], quaternion=value['_rotation'],
                opacity_logit=value['_opacity'][:, 0], dc=value['_features_dc'][:, 0],
                static_offset=value['static_offset'], meta=meta, files=pins,
                attachment=_read(directory/'attachment.json'))
    item['world_ply_row_parity'] = _world_ply_parity(directory/'world.ply', item, mesh)
    item['reference_scale_max_abs_m'] = scale_error
    return item


def _geometry(item, rows):
    scale = item['face_scale'][item['binding'][rows]].astype(np.float64)
    rotation = _rotation(item['quaternion'][rows])
    sigma = np.exp(item['log_sigma'][rows].astype(np.float64))*scale[:, None]
    order = np.argsort(sigma, axis=1, kind='stable')
    ordered = np.take_along_axis(sigma, order, axis=1)
    axes = rotation*sigma[:, None, :]
    tangent = axes[:, [0, 2]]
    tangent_cov = tangent @ tangent.transpose(0, 2, 1)
    thin = np.take_along_axis(rotation[:, 1], order[:, :1], axis=1)[:, 0]
    return dict(abs_normal_center_mm=1000*abs(item['xyz'][rows, 1]*scale),
                normal_sigma_mm=1000*np.linalg.norm(axes[:, 1], axis=1),
                tangent_major_sigma_mm=1000*np.sqrt(np.linalg.eigvalsh(tangent_cov)[:, 1].clip(0)),
                flattened=ordered[:, 1]/ordered[:, 0].clip(1e-9) >= FLAT_RATIO,
                flattened_thin_axis_angle_deg=np.degrees(np.arccos(np.clip(abs(thin), 0, 1))))


def _stats(values):
    if not len(values):
        return None
    _require(np.isfinite(values).all(), 'Nonfinite matched geometry metric.')
    return dict(zip(('p05', 'p50', 'p95'), np.quantile(values, [.05, .5, .95]).tolist()))


def _comparison(first, last):
    delta = last-first
    return dict(count=len(delta), at_10000=_stats(first), at_18000=_stats(last),
                paired_delta_18000_minus_10000=_stats(delta), reduced=int((delta < 0).sum()),
                increased=int((delta > 0).sum()), exact_ties=int((delta == 0).sum()))


def _match_clothing(first, last, supported):
    common, i, j = np.intersect1d(first['ids'], last['ids'], assume_unique=True, return_indices=True)
    _require(np.array_equal(first['binding'][i], last['binding'][j]),
             'A surviving stable ID changed triangle owner.')
    active = [torch.from_numpy(item['opacity_logit']).sigmoid().ge(OPACITY_MIN).numpy() for item in (first, last)]
    support = [supported[item['binding']] & mask for item, mask in zip((first, last), active, strict=True)]
    _require(all(int(mask.sum()) == item['attachment']['clothing']['active_rows']
                 for item, mask in zip((first, last), support, strict=True)),
             'CPU active clothing classification differs from the actual raw training receipt.')
    paired = support[0][i] & support[1][j]
    old, new = _geometry(first, i[paired]), _geometry(last, j[paired])
    both_flat = old['flattened'] & new['flattened']
    metrics = {name: _comparison(old[name], new[name]) for name in
               ('abs_normal_center_mm', 'normal_sigma_mm', 'tangent_major_sigma_mm')}
    metrics['flattened_thin_axis_angle_deg'] = _comparison(
        old['flattened_thin_axis_angle_deg'][both_flat], new['flattened_thin_axis_angle_deg'][both_flat])
    ids = common[paired]
    return dict(state='measured' if len(ids) else 'no_matched_active_clothing_rows',
                active_opacity_min=OPACITY_MIN, flattened_middle_over_min_sigma_min=FLAT_RATIO,
                supported_faces=int(supported.sum()), all_stable_rows_10000=len(first['ids']),
                all_stable_rows_18000=len(last['ids']), surviving_all_rows=len(common),
                removed_rows=len(first['ids'])-len(common), added_rows=len(last['ids'])-len(common),
                active_supported_rows_10000=int(support[0].sum()), active_supported_rows_18000=int(support[1].sum()),
                matched_active_supported_rows=len(ids),
                matched_stable_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                matched_flattened_both=int(both_flat.sum()), matched_flattened_only_10000=int((old['flattened'] & ~new['flattened']).sum()),
                matched_flattened_only_18000=int((new['flattened'] & ~old['flattened']).sum()), metrics=metrics)


def _skin_populations(first, last):
    first_rows, last_rows = (item['attachment']['skin']['views'] for item in (first, last))
    a, b = (item['attachment']['support']['views'] for item in (first, last))
    _require([row['frame'] for row in first_rows] == [row['frame'] for row in last_rows]
             == [row['frame'] for row in a] == [row['frame'] for row in b], 'The native skin roster changed.')
    _require(all(x.get('native_material_mask_sha256') == y.get('native_material_mask_sha256')
                 and x['repeated_skin_triangles'] == y['repeated_skin_triangles']
                 for x, y in zip(a, b, strict=True)), 'Fixed native skin evidence differs across endpoints.')
    comparisons = {}
    for name in ('abs_normal_center_mm', 'normal_sigma_mm', 'tangent_major_sigma_mm', 'flattened_thin_axis_angle_deg'):
        pairs = [(x[name]['p50'], y[name]['p50']) for x, y in zip(first_rows, last_rows, strict=True)
                 if x.get(name) is not None and y.get(name) is not None]
        if pairs:
            old, new = np.asarray(pairs).T
            comparisons[name] = dict(common_views=len(pairs), median_view_p50_10000=float(np.median(old)),
                                     median_view_p50_18000=float(np.median(new)))
    return dict(state='per_view_unmatched_populations', fixed_skin_roster=True, stable_id_matching=False,
                scope='Native mask identities and measurement views match. The receipt stores no skin row membership.',
                gaussian_view_rows_10000=sum(row['supported_rows'] for row in first_rows),
                gaussian_view_rows_18000=sum(row['supported_rows'] for row in last_rows),
                view_median_summary=comparisons,
                views=[dict(frame=x['frame'], at_10000=x, at_18000=y) for x, y in zip(first_rows, last_rows, strict=True)])


def audit(run):
    out = run/'gaussian-run'
    first, last = (_endpoint(out, step) for step in STEPS)
    _require(first['files']['reference-mesh.npz']['sha256'] == last['files']['reference-mesh.npz']['sha256']
             and np.array_equal(first['face_scale'], last['face_scale'])
             and np.array_equal(first['static_offset'], last['static_offset'])
             and first['meta']['reference_frame_id'] == last['meta']['reference_frame_id']
             and first['meta']['source_draw_sha256'] == last['meta']['source_draw_prefix_10000_sha256'],
             'Matched audit requires the same exact frozen surface, scale, pose and training trajectory.')
    guide_path = out/'clothing-surface.json'
    guide = _read(guide_path)
    guide_support = (out/guide['support_file']).resolve(strict=True)
    _require(guide_support.is_relative_to(out), 'The clothing support file escaped its run.')
    guide_pin = _pin(guide_support, guide['support_file_sha256'])
    _require(guide['active_mesh_sha256'] == first['files']['reference-mesh.npz']['sha256'],
             'The raw endpoints do not use this active garment guide.')
    with np.load(guide_pin['file'], allow_pickle=False) as support:
        supported = support['supported_faces']
        _require(supported.dtype == bool and supported.shape == first['face_scale'].shape
                 and np.array_equal(support['final_offset'], first['static_offset'])
                 and hashlib.sha256(np.flatnonzero(supported).astype('<i8').tobytes()).hexdigest()
                 == guide['supported_face_ids_sha256'],
                 'The admitted clothing support or shared offset differs from the raw model.')
    return dict(schema='shuolee-r21-matched-stable-geometry/1', run=str(run), raw_steps=list(STEPS),
                audit_source_sha256=_sha(Path(__file__).resolve()),
                reference_frame=first['meta']['reference_frame_id'],
                reference_normal='Frozen triangle local axis +Y, positive follows triangle winding; absolute offsets in mm.',
                covariance='Full one-sigma covariance from normalized bound wxyz and actual metric face scale.',
                guide=dict(file=str(guide_path), sha256=_sha(guide_path), state=guide['state'], support=guide_pin),
                raw=[dict(step=item['step'], files=item['files'], native_world_ply_row_parity=item['world_ply_row_parity'],
                          reference_scale_max_abs_m=item['reference_scale_max_abs_m']) for item in (first, last)],
                clothing=_match_clothing(first, last, supported), native_skin=_skin_populations(first, last))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    torch.set_num_threads(2)
    result = json.dumps(audit(run), indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output is None:
        print(result, end='')
    else:
        output = args.output.resolve()
        _require(not output.is_relative_to(run), 'Save this read-only audit outside the production run.')
        with output.open('x', encoding='utf-8') as stream:
            stream.write(result)


if __name__ == '__main__':
    main()
