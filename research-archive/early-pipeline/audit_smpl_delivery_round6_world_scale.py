"""Independently audit exported SH0 world widths and named-pose projections on CPU."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_smpl_delivery_round5_ear016 import pin, read, require, sha


LAB = Path('/opt/gs-slam-lab')
ASSET = Path('/mnt/d/GS-SLAM-Data/shuolee-portrait-runtime-assets/smplx/SMPLX_NEUTRAL.npz')
DIRECTOR = Path('/mnt/c/Users/Administrator/Documents/Codex/2026-11-09/3dgs-00/outputs/round5_director_review_20261111.json')
DIRECTOR_SHA = 'f50e7766b668ca343db640771fc727c8410f64261af9ffac07fac58ed8eb9d66'
COMMITS = {5: '9013c26bc1b0758a5798ae2c3b0cf28e4d1a72c0', 6: '8c722b9aabd7577803118c910bb9971a82f009d6'}
ACTIVE, NEAR, SMALL, SLACK = .1, .9, 5., .025
SCALE_KEYS = [f'scale_{i}' for i in range(3)]
QUAT_KEYS = [f'rot_{i}' for i in (1, 2, 3, 0)]
LOG_CAPS = dict(triangle_scale_multiple=3., minimum_extent_ratio=.016, global_extent_ratio=.025)


def array(ply, keys):
    return np.column_stack([ply[key] for key in keys]).astype(np.float64)


def vertices(path):
    ply = PlyData.read(path, mmap=False)
    require(not ply.text and [e.name for e in ply.elements] == ['vertex'], 'Expected a binary vertex-only Gaussian PLY.')
    return ply['vertex'].data


def face_frame(vertices, faces):
    a, b, c = vertices[faces].astype(np.float64).transpose(1, 0, 2)
    edge, other = b-a, c-a
    length = np.linalg.norm(edge, axis=1)
    first = edge/length[:, None]
    normal = np.cross(first, other)
    altitude = np.linalg.norm(normal, axis=1)
    normal /= altitude[:, None]
    third = np.cross(first, normal)
    return (a+b+c)/3., np.stack((first, normal, third), axis=-1), np.maximum((length+altitude)/2., 1e-8)


def mesh_front_depth(uv, vertex_depth, faces, xy):
    tri = uv[faces]
    front = np.full(len(xy), np.inf)
    good = (vertex_depth[faces] > .2).all(1)
    in_box = ((tri.min(1)[None] <= xy[:, None]) & (tri.max(1)[None] >= xy[:, None])).all(2) & good
    pixel, owner = np.nonzero(in_box)
    a, b, c = tri[owner].transpose(1, 0, 2)
    p, u, v = xy[pixel]-a, b-a, c-a
    denom = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        l1 = (p[:, 0]*v[:, 1]-p[:, 1]*v[:, 0])/denom
        l2 = (u[:, 0]*p[:, 1]-u[:, 1]*p[:, 0])/denom
        bary = np.column_stack((1-l1-l2, l1, l2))/vertex_depth[faces[owner]]
        z = np.where((bary >= -1e-8).all(1), 1/bary.sum(1), np.inf)
    np.minimum.at(front, pixel, z)
    return front


def summary(values, limit):
    quantiles = np.quantile(values, [.50, .95, .99, 1.]).tolist() if len(values) else [None]*4
    return dict(points=len(values), **dict(zip(('p50_m','p95_m','p99_m','max_m'), quantiles, strict=True)),
        over_005m=int((values > .05).sum()), over_010m=int((values > .10).sum()), over_limit=int((values > limit+1e-6).sum()))


def compare_summary(saved, measured):
    for group in ('all', 'active'):
        for key, want in measured[group].items():
            got = saved[group][key]
            equal = got == want if key in ('points','over_005m','over_010m','over_limit') or want is None else abs(got-want) <= 2e-7
            require(equal, f'World-width terminal mismatch {group}.{key}: log={got}, PLY={want}')


def projections(mesh, faces, matrices, rows, per_face, owner, center, rotations, sigma, active, strict_head, reference, limit):
    world_ref = sigma.max(1)*per_face[reference, owner]
    world_max = sigma.max(1)*per_face.max(0)[owner]
    large_ref = active & (world_ref > .05)
    large_any = active & (world_max > .05)
    near = active & (world_max >= NEAR*limit) & (sigma.max(1) >= 2*sigma.min(1))
    selected = np.flatnonzero(large_any | near)
    per_view = []
    for slot, (record, row) in enumerate(zip(matrices, rows, strict=True)):
        require(record['frame'] == row['frame'] == mesh['frame_ids'][slot] and record['pose_index'] == slot
                and record['pixel_convention'] == 'pixel_corners'
                and all(record[key] == row[key] for key in ('K','w2c','wh')), 'Named native camera does not match fitted row.')
        v = mesh['vertices_world'][slot]
        origin, axes, s = face_frame(v, faces)
        ids = owner[selected]
        world = origin[ids] + np.einsum('nij,nj->ni', axes[ids], center[selected])*s[ids, None]
        K, E = (np.asarray(record[key]) for key in ('K','w2c'))
        cam = world @ E[:3, :3].T+E[:3, 3]
        q = cam @ K.T
        xy = q[:, :2]/q[:, 2:3].clip(.01)
        valid = (q[:, 2] > .2) & (xy >= 0).all(1) & (xy < record['wh']).all(1)
        kept = np.flatnonzero(valid)
        pixel = np.floor(xy[kept]).astype(int)
        vcam = v @ E[:3, :3].T+E[:3, 3]
        vq = vcam @ K.T
        uv = vq[:, :2]/vq[:, 2:3].clip(.01)
        depth = mesh_front_depth(uv, vq[:, 2], faces, pixel+.5)
        shown = q[kept, 2] <= depth+SLACK
        matte_path = pin(row['mask'], row['mask_sha256'])
        with Image.open(matte_path) as image:
            require(image.size == tuple(record['wh']), 'Native matte dimensions differ.')
            matte = np.asarray(image.convert('L'))[pixel[:, 1], pixel[:, 0]]
        jacobian = (K[:2]-xy[kept, :, None]*K[2])/q[kept, 2, None, None]
        point = selected[kept]
        posed_axes = axes[ids[kept]] @ rotations[point]
        projected = jacobian @ E[:3, :3] @ (posed_axes*(sigma[point]*s[ids[kept], None])[:, None, :])
        axis_id = sigma[point].argmax(1)
        major_projected = np.linalg.norm(projected[np.arange(len(point)), :, axis_id], axis=1)
        covariance = projected @ projected.transpose(0, 2, 1)
        ellipse = np.sqrt(np.maximum(0., np.linalg.eigvalsh(covariance)[:, -1]))
        require(np.all(major_projected <= ellipse+1e-10), 'An individual projected axis exceeds covariance major width.')

        def counts(group):
            here = group[kept]
            visible = here & shown
            small_axis = major_projected < SMALL
            small_ellipse = ellipse < SMALL
            return dict(pose_candidates=int(group.sum()), in_frame=int(here.sum()),
                in_frame_depth_tested=int(visible.sum()), supported_foreground=int((visible & (matte >= 253)).sum()),
                strict_head_depth_tested=int((visible & strict_head[point]).sum()),
                world_major_projected_lt_5px=int((visible & small_axis).sum()),
                full_ellipse_major_lt_5px=int((visible & small_ellipse).sum()),
                world_major_lt_5px_but_ellipse_ge_5px=int((visible & small_axis & ~small_ellipse).sum()))

        per_view.append(dict(frame=record['frame'], pose_index=slot,
            reference_large=counts(large_ref[selected]),
            pose_large=counts((sigma[selected].max(1)*s[ids] > .05) & active[selected]),
            near_limit_repair_eligible=counts(near[selected])))
    groups = ('reference_large','pose_large','near_limit_repair_eligible')
    return dict(small_sigma_px=SMALL, mesh_depth_slack_m=SLACK,
        notes='Selected source-training poses only. Center in image with z>0.2 m, depth-tested against nearest posed mesh pixel-center ray. Analytic full-K Jacobian before raster low-pass; projected world major is one axis, ellipse includes all three. Foreground support is native matte only, not point contribution.',
        unique_reference_large=int(large_ref.sum()), unique_any_pose_large=int(large_any.sum()),
        unique_near_limit_repair_eligible=int(near.sum()), per_pose=per_view,
        total_pose_pairs={group: {key: sum(row[group][key] for row in per_view) for key in per_view[0][group]} for group in groups})


def measure(subject, round_number, director, project_views):
    root = LAB/f'shuolee-smpl-delivery-round{round_number}-{subject}'
    run, out = root/'run', root/'run/gaussian-run'
    wrapper = read(root/'wrapper-terminal.json')
    result_path = pin(run/'result.json', wrapper['result_sha256'])
    result = read(result_path)
    train = read(out/'result.json')
    integration_path = pin(out/'integration.json', wrapper['integration_sha256'])
    integration = read(integration_path)
    require(wrapper['state'] == result['state'] == train['state'] == integration['state'] == 'completed'
            and wrapper['candidate_commit'] == COMMITS[round_number]
            and result['gaussian_from_scratch'] and result['fit_calls'] == 0 and result['steps'] == 10000
            and result['final_snapshot'] == train['final_snapshot'] == 'step-010000'
            and result['unified_main_training'] == train['unified_main_training'] == integration, 'Incomplete fresh Gaussian delivery.')
    snapshot = out/result['final_snapshot']
    parity = read(snapshot/'native-ply-parity.json')
    require(parity['passed'] is True, 'Native PLY replay failed.')
    replay_path = pin(run/result['bound_replay'], result['bound_replay_sha256'])
    replay = read(replay_path)
    posed_path = pin(replay_path.parent/replay['mesh_file'], replay['mesh_sha256'])
    bound = read(pin(snapshot/'bound-replay.json', replay['bound_replay_sha256']))
    fit_path = pin(run/'fit/fitted.npz', bound['inputs_sha256']['fitted'])
    ref_mesh = pin(snapshot/'reference-mesh.npz', parity['mesh_sha256'])
    world_path = pin(snapshot/'world.ply', parity['sha256'])
    local_path = pin(snapshot/'bound-local.ply', bound['bound_ply_sha256'])
    portable_path = pin(run/result['ply'], result['sha256'])
    world, local, portable = map(vertices, (world_path, local_path, portable_path))
    require(len(world) == len(local) == len(portable) == train['points'] == result['points'] == bound['points']
            and all(world[k].tobytes() == portable[k].tobytes() for k in (*SCALE_KEYS, 'opacity'))
            and world['opacity'].tobytes() == local['opacity'].tobytes(), 'Metric axes, opacity or point count changed at export.')
    with np.load(posed_path, allow_pickle=False) as npz, np.load(pin(ASSET, bound['asset_sha256']), allow_pickle=False) as asset:
        mesh = dict(npz)
        faces = mesh['faces']
        require(np.array_equal(faces, asset['f']), 'Posed mesh topology differs from SMPL-X.')
        protected = np.isin(asset['weights'].argmax(1), [12,15,22,23,24])
        protected[faces[np.r_[asset['lmk_faces_idx'].ravel(), asset['dynamic_lmk_faces_idx'].ravel()]].ravel()] = True
    reference = replay['reference_pose_index']
    with np.load(ref_mesh, allow_pickle=False) as mesh_ref, np.load(fit_path, allow_pickle=False) as fit:
        require(np.array_equal(mesh_ref['vertices_world'], mesh['vertices_world'][reference]), 'The exported mesh does not match the named reference pose.')
        extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][reference], axis=0)))
    owner = local['binding_0'].astype(int)
    require(np.array_equal(owner, local['binding_0']) and owner.min() >= 0 and owner.max() < len(faces)
            and (np.bincount(owner, minlength=len(faces)) > 0).all(), 'Invalid bound triangle IDs.')
    face_scales = np.stack([face_frame(pose, faces)[2] for pose in mesh['vertices_world']])
    max_scale = face_scales.max(0)
    face_cap = np.clip(3.*max_scale, .016*extent, .025*extent)
    cap = face_cap[owner]
    logs, local_logs = array(world, SCALE_KEYS), array(local, SCALE_KEYS)
    require(np.isfinite(logs).all() and np.isfinite(local_logs).all(), 'Nonfinite Gaussian scale.')
    max_log_error = float(np.abs(logs-local_logs-np.log(face_scales[reference, owner, None])).max())
    require(max_log_error < 4e-6, 'World PLY does not reproduce posed triangle-local scale.')
    sigma = np.exp(local_logs)
    ref_world, max_world = np.exp(logs).max(1), sigma.max(1)*max_scale[owner]
    opacity = expit(world['opacity'].astype(np.float64))
    require(np.isfinite(opacity).all(), 'Nonfinite Gaussian opacity.')
    active, strict_head = opacity >= ACTIVE, protected[faces].all(1)[owner]
    all_points = np.ones(len(world), dtype=bool)
    groups = dict(all=all_points, active=active, strict_head=strict_head, strict_head_active=strict_head & active,
                  body_or_mixed=~strict_head, body_or_mixed_active=~strict_head & active)
    stats = {label: {group: summary(values[m], cap[m]) for group, m in groups.items()}
             for label, values in (('reference',ref_world), ('all_poses_max',max_world))}
    opacity_quantiles = {group: np.quantile(opacity[m], [.05,.50,.95]).tolist() for group, m in groups.items() if m.any()}
    if round_number == 5:
        expected = director['direct_ply_audit'][f'{subject}_round5']
        actual = stats['reference']['active']
        require(stats['reference']['all']['points'] == expected['points'] and actual['points'] == expected['active_points']
                and actual['over_005m'] == expected['major_over_005m'] and actual['over_010m'] == expected['major_over_010m']
                and abs(actual['p99_m']-expected['major_sigma_p99_m']) < 2e-7
                and abs(actual['max_m']-expected['major_sigma_max_m']) < 2e-7, 'R5 portable metrics differ from director baseline.')
        scale_report = None
    else:
        registered = integration['world_scale']
        log = read(pin(out/registered['file'], registered['sha256']))
        options = log['config']
        require(registered['file'] == 'world-scale.json' and log['state'] == 'completed'
                and log['terminal'] == registered['terminal'] and log['reference_pose_index'] == reference
                and log['pose_count'] == len(mesh['vertices_world']) and abs(log['extent_m']-extent) < 2e-7
                and all(options[key] == value for key, value in LOG_CAPS.items())
                and options['active_opacity'] == ACTIVE and options['near_limit_ratio'] == NEAR
                and options['small_projected_sigma_px'] == SMALL and options['axis_ratio_min'] == 2.
                and options['audit_thresholds_m'] == [.05,.10], 'Logged R6 world scale setup differs.')
        require(max(abs(log['limits_m']['min']-face_cap.min()), abs(log['limits_m']['max']-face_cap.max())) < 2e-7,
                'World cap limits differ from independent posed mesh calculation.')
        for key in ('reference','all_poses_max'):
            compare_summary(log['terminal'][key], stats[key])
        require(stats['all_poses_max']['all']['over_limit'] == 0 and [event['step'] for event in log['splits']] == options['split_steps'],
                'R6 hard cap or split schedule failed.')
        for event in log['splits']:
            require(event['split_parents']*2 == event['children'] and event['assessed'] <= event['near_limit_candidates']
                    and event['assessed'] == sum(event[k] for k in ('split_parents','exterior_trimmed','retained_without_child_support'))
                    and event['small_projected_ellipse_pairs'] <= event['small_projected_world_axis_pairs'] <= event['visible_parent_view_pairs'],
                    'R6 event statistics violate support accounting.')
        checkpoints = log['checkpoints']
        stages = [row['stage'] for row in checkpoints]
        require(stages.count('stock_densification') == integration['stock_densifications']
                and [row['step'] for row in checkpoints if row['stage'] == 'coverage_densification'] == integration['schedule']['coverage_steps']
                and [row['step'] for row in checkpoints if row['stage'] == 'snapshot'] == [1000]
                and all(row['all_poses_max']['all']['over_limit'] == 0 for row in checkpoints),
                'R6 world-scale checkpoints do not follow the original densification and snapshot schedule.')
        scale_report = dict(sha256=registered['sha256'], terminal=log['terminal'], splits=log['splits'],
            checkpoints=len(checkpoints), cap_min_m=float(face_cap.min()), cap_max_m=float(face_cap.max()),
            hard_projection_totals={key: sum(row['hard_projection'][key] for row in (*checkpoints, log['terminal']))
                                    for key in ('axis_hits','point_hits','calls_with_hits')})

    input_path = pin(run/'data/training-inputs.json', replay['training_inputs_sha256'])
    camera_path = pin(out/'training-cameras.json', replay['training_camera_sha256'])
    rows, camera = read(input_path)['records'], read(camera_path)['records']
    ids = [row['frame'] for row in rows]
    require(ids == list(mesh['frame_ids']) and len(set(ids)) == len(ids) and set(ids).isdisjoint(replay['retained_holdouts']),
            'World projection audit must use only the frozen training roster.')
    projected = projections(mesh, faces, camera, rows, face_scales, owner, array(local, ('x','y','z')),
        Rotation.from_quat(array(local, QUAT_KEYS)).as_matrix(), sigma, active, strict_head, reference, cap) if project_views else None
    return dict(round=round_number, subject=subject, commit=COMMITS[round_number], points=len(world), result_sha256=sha(result_path),
        portable_ply_sha256=result['sha256'], world_ply_sha256=parity['sha256'], bound_ply_sha256=bound['bound_ply_sha256'],
        fit_sha256=sha(fit_path), posed_mesh_sha256=sha(posed_path), reference_mesh_sha256=sha(ref_mesh),
        input_sha256=sha(input_path), training_camera_sha256=sha(camera_path), export_camera_sha256=sha(run/'export/cameras.json'),
        initial_gaussian_parameter_sha256=integration['local_parameter_sha256'], per_frame_updates=train['per_frame_updates'],
        source_bin_draws=train['source_bin_draws'], reference_slot=reference, poses=len(rows), extent_m=extent,
        retained_holdouts=replay['retained_holdouts'],
        exported_scale_and_opacity_bitwise_equal=True, local_to_world_max_log_sigma_error=max_log_error,
        metrics=stats, opacity_p05_p50_p95=opacity_quantiles, projected_widths=projected, world_scale_log=scale_report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    director = read(pin(DIRECTOR, DIRECTOR_SHA))
    subjects = {}
    for subject in ('016','006'):
        before = measure(subject, 5, director, True)
        subjects[subject] = dict(round5=before)
        if not args.baseline_only:
            after = measure(subject, 6, director, True)
            exact = ('fit_sha256','posed_mesh_sha256','reference_mesh_sha256','input_sha256','training_camera_sha256',
                     'export_camera_sha256','initial_gaussian_parameter_sha256','per_frame_updates','source_bin_draws','retained_holdouts')
            require(all(before[key] == after[key] for key in exact), 'R6 frozen geometry, cameras, initialization or sampling changed.')
            subjects[subject].update(round6=after, exact_round5_round6_fields=list(exact))
    payload = dict(schema='shuolee-smpl-round6-independent-world-cpu-v1', state='baseline_verified' if args.baseline_only else 'verified',
        units='model-conditioned world metres, principal axis one-sigma', director_review_sha256=DIRECTOR_SHA,
        opacity_active_threshold=ACTIVE, gpu_calls=0, quality_accepted=False, subjects=subjects)
    data = json.dumps(payload, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(data)
    print(data, end='')


if __name__ == '__main__':
    main()
