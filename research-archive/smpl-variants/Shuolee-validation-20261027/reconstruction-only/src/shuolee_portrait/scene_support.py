"""Local scene observability, with no camera fit or residual-based selection.

Derivatives are evaluated at the frozen, conditional triangulation. Background
ROI labels do not establish staticity. A full-rank local model is necessary in
that model, not evidence of a physically correct camera or static subject.
"""
from collections import Counter

import numpy as np

from .fixed_tracks import cameras_from_records, _summary

# Fixed engineering budget: never silently subsample the frozen bank to fit it.
MAX_JOINT_COLUMNS = 1200
MAX_JOINT_ENTRIES = 8_000_000
RANK_RTOL = 1e-8
RANK_ATOL = 1e-9
K_NAMES = ['fx_1percent', 'fy_1percent', 'cx_1pixel', 'cy_1pixel']
POSE_NAMES = ['rx_1degree', 'ry_1degree', 'rz_1degree',
              'tx_1percent_baseline', 'ty_1percent_baseline', 'tz_1percent_baseline']


def specification():
    return dict(pixel_domain='undistorted_integer_centers',
        camera_model='The existing rectified pinhole model only; no raw distortion changes.',
        k_parameters=K_NAMES, pose_parameters=POSE_NAMES,
        point_unit='One reference camera baseline, in arbitrary reconstruction scale.',
        gauge='First camera fixed. Farthest camera from the first fixes one baseline length; its translation is tangent to that baseline.',
        reference_baseline_selection='Maximum distance from first camera, first index on ties; no residual selection.',
        rank_relative_tolerance=RANK_RTOL, rank_absolute_tolerance=RANK_ATOL,
        max_joint_columns=MAX_JOINT_COLUMNS, max_joint_entries=MAX_JOINT_ENTRIES,
        no_optimization=True, no_residual_filtering=True,
        noise_interpretation='Pixel sensitivity only. No noise distribution is measured and no statistical confidence interval is claimed.')


def _skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def projection_jacobians(camera, xyz, baseline):
    """d(integer-centre pixel)/d(scaled K, left camera SE3, world XYZ).

The constant -0.5 pixel offset has zero derivative. A left rotation acts on
both R and t, so its derivative is -skew(R X+t); it does not move the camera
centre. Translation derivatives are in camera coordinates.
"""
    K, E = camera['K'], camera['E']
    y = E[:3, :3] @ np.asarray(xyz) + E[:3, 3]
    if not np.isfinite(y).all() or y[2] <= 0 or not np.isfinite(baseline) or baseline <= 0:
        raise ValueError('Positive finite depth and reference baseline required.')
    x, v, z = y;fx, fy = K[0, 0], K[1, 1]
    image = np.array([[fx/z, 0., -fx*x/z**2], [0., fy/z, -fy*v/z**2]])
    intrinsic = np.array([[.01*fx*x/z, 0., 1., 0.], [0., .01*fy*v/z, 0., 1.]])
    pose = image @ np.c_[-_skew(y)*np.pi/180., np.eye(3)*(.01*baseline)]
    point = image @ E[:3, :3] * baseline
    return intrinsic, pose, point


def _spectrum(a, names, reference=None):
    """Numerical local rank in declared units; missing information is not zero error."""
    n = len(names)
    if len(a) == 0:
        return dict(rows=0, parameters=n, numerical_rank=0, nullity=n,
                    singular_values=[], condition_number=None, least_sensitive_direction=None,
                    unit_pixel_l2_parameter_amplification=None, amplification_unbounded=True)
    padded = np.pad(a, ((0, max(0, n-len(a))), (0, 0)))
    _, s, v = np.linalg.svd(padded, full_matrices=False)
    reference = float(s[0] if reference is None else reference)
    tol = max(RANK_ATOL, RANK_RTOL*reference)
    rank = int(np.count_nonzero(s > tol))
    return dict(rows=len(a), parameters=n, numerical_rank=rank, nullity=n-rank,
        singular_values=s.tolist(), rank_threshold=tol,
        condition_number=float(s[0]/s[-1]) if rank == n else None,
        least_sensitive_direction=dict(zip(names, map(float, v[-1]))),
        unit_pixel_l2_parameter_amplification=float(1/s[-1]) if rank == n else None,
        amplification_unbounded=rank != n,
        null_direction_basis_not_unique=rank < n)


def _tangent(normal):
    normal = normal/np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    a = np.cross(normal, axis);a /= np.linalg.norm(a)
    return np.c_[a, np.cross(normal, a)]


def _layout(cameras, anchor):
    frames = list(cameras);first = frames[0];origin = cameras[first]['center']
    matrices = {};offsets = {};names = list(K_NAMES);offset = 4
    for frame in frames[1:]:
        basis = np.eye(6)
        labels = POSE_NAMES
        if frame == anchor:
            direction = cameras[frame]['E'][:3, :3] @ (cameras[frame]['center']-origin)
            basis = np.zeros((6, 5));basis[:3, :3] = np.eye(3)
            basis[3:, 3:] = _tangent(direction)
            labels = POSE_NAMES[:3] + ['tangent_a_1percent_baseline', 'tangent_b_1percent_baseline']
        matrices[frame] = basis;offsets[frame] = (offset, offset+basis.shape[1])
        names.extend(f'{frame}:{label}' for label in labels);offset += basis.shape[1]
    return matrices, offsets, names


def _local_information(cameras, tracks, results, baseline, anchor, shared_k):
    """Eliminate point nuisance directions, then pose nuisance directions.

This is a Jacobian projection only: no measured pixel, point, K or E is fitted
or updated. Both triangulation and held views contribute design information;
this information calculation is not an independent held-error score.
"""
    bases, offsets, names = _layout(cameras, anchor)
    selected = [t for t in tracks if results[t['track_id']]['status'] == 'measured']
    upper_rows = sum(2*len(t['observations']) for t in selected)
    within_budget = len(names) <= MAX_JOINT_COLUMNS and upper_rows*len(names) <= MAX_JOINT_ENTRIES
    poses = {f: [] for f in cameras};conditional_blocks = [];joint_blocks = []
    point_ranks = Counter();unprofiled_k = []
    for track in selected:
        point = results[track['track_id']]['xyz'];jk = [];jx = [];jp = []
        for obs in track['observations']:
            k, pose, x = projection_jacobians(cameras[obs['frame']], point, baseline)
            jk.append(k);jx.append(x);jp.append((obs['frame'], pose))
            poses[obs['frame']].append(pose)
        jk = np.vstack(jk);jx = np.vstack(jx)
        u, singular, _ = np.linalg.svd(jx, full_matrices=True)
        rank = int(np.count_nonzero(singular > max(RANK_ATOL, RANK_RTOL*singular[0])))
        point_ranks[rank] += 1
        complement = u[:, rank:].T
        if shared_k:
            conditional_blocks.append(complement @ jk);unprofiled_k.append(jk)
            if within_budget:
                block = np.zeros((len(jk), len(names)));block[:, :4] = jk
                for i, (frame, pose) in enumerate(jp):
                    if frame in offsets:
                        start, end = offsets[frame]
                        block[2*i:2*i+2, start:end] = pose @ bases[frame]
                joint_blocks.append(complement @ block)
    per_frame = []
    for frame, blocks in poses.items():
        matrix = np.vstack(blocks) if blocks else np.zeros((0, 6))
        per_frame.append(dict(frame=frame, observations=len(blocks),
            conditional_pose=_spectrum(matrix, POSE_NAMES),
            assumption='All triangulated XYZ and K treated as exact; this is an optimistic conditional calculation.'))
    result = dict(usable_tracks=len(selected), per_frame_pose=per_frame,
        triangulated_point_jacobian_ranks={str(k):v for k,v in point_ranks.items()},
        conditional_k=None, joint_k_and_pose=None,
        independent_validation=False, cameras_optimized=False,
        bounds=dict(columns=len(names), observation_rows_upper_bound=upper_rows,
                    entries_upper_bound=upper_rows*len(names), within_budget=within_budget))
    if not shared_k:
        result['state'] = 'shared_rectified_K_not_supported_by_manifest'
        return result
    conditional = np.vstack(conditional_blocks) if conditional_blocks else np.zeros((0, 4))
    raw = np.vstack(unprofiled_k) if unprofiled_k else np.zeros((0, 4))
    reference = float(np.linalg.norm(raw, ord=2)) if len(raw) else 0.
    result['conditional_k'] = _spectrum(conditional, K_NAMES, reference)
    result['conditional_k']['assumption'] = 'Extrinsics fixed; unknown shared XYZ changes are projected out.'
    if not within_budget:
        result['state'] = 'joint_budget_exceeded_no_subsampling'
        return result
    a = np.vstack(joint_blocks) if joint_blocks else np.zeros((0, len(names)))
    if not len(a):
        result['state'] = 'no_observable_scene_candidates'
        return result
    intrinsic, pose = a[:, :4], a[:, 4:]
    u, singular, _ = np.linalg.svd(pose, full_matrices=False)
    tol = max(RANK_ATOL, RANK_RTOL*float(singular[0]))
    rank_pose = int(np.count_nonzero(singular > tol))
    intrinsic_remaining = intrinsic - u[:, :rank_pose] @ (u[:, :rank_pose].T @ intrinsic)
    spectrum = _spectrum(intrinsic_remaining, K_NAMES, reference)
    result['joint_k_and_pose'] = dict(
        shared_k_after_points_and_poses=spectrum,
        pose_parameters=pose.shape[1], pose_numerical_rank=rank_pose,
        pose_nullity_with_fixed_K=pose.shape[1]-rank_pose,
        pose_rank_threshold=tol,
        joint_parameters=a.shape[1], joint_structured_rank=rank_pose+spectrum['numerical_rank'],
        joint_structured_nullity=a.shape[1]-rank_pose-spectrum['numerical_rank'],
        reference_camera=list(cameras)[0], scale_anchor_camera=anchor,
        gauge_constraints=7, parameter_names=names,
        rank_scope='Local scaled Jacobian after explicit point/pose nuisance projection; not global uniqueness.',
        sensitivity_retained_fraction=float(np.linalg.norm(intrinsic_remaining)/np.linalg.norm(intrinsic)) if np.linalg.norm(intrinsic) > RANK_ATOL else None)
    result['state'] = 'local_design_computed'
    return result


def analyze_scene_support(rows, annotated, measurement):
    cameras = cameras_from_records(rows);frames = list(cameras)
    results = {r['track_id']:r for r in measurement['tracks']}
    if set(results) != {t['track_id'] for t in annotated['tracks']}:
        raise ValueError('Scene analysis must retain exactly the frozen measured-track membership.')
    origin = cameras[frames[0]]['center']
    distances = [np.linalg.norm(c['center']-origin) for c in cameras.values()]
    anchor = frames[int(np.argmax(distances))];baseline = float(max(distances))
    epsilon = np.finfo(float).eps*max(1., np.max(np.abs([c['center'] for c in cameras.values()])))*64
    shared = all(np.array_equal(c['wh'], cameras[frames[0]]['wh']) and
                 np.allclose(c['K'], cameras[frames[0]]['K'], rtol=0., atol=1e-8)
                 for c in cameras.values())
    scene = [t for t in annotated['tracks'] if t['region'] == 'static_scene_candidate']
    report = dict(specification=specification(), scene_candidate_tracks=len(scene),
        original_track_count=len(annotated['tracks']), source_tracks_dropped=0,
        scene_staticity_verified=False, material_identity_verified=False,
        source_cameras_or_pixels_modified=False, physical_calibration_verified=False,
        root_cause_resolved=False, quality_accepted=False, training_admitted=False,
        shared_rectified_intrinsics=dict(equal_within_1e_minus_8=shared,
            hypothesis='A shared rectified K, not evidence for one constant physical sensor K.',
            original_raw_distortion_optimized=False),
        reference_baseline_world_units=baseline, baseline_anchor_frame=anchor,
        information=None, frames=[], tracks=[], subject_depth_comparison=[],
        limits=['Background pixels can be moving people, screens or reflections.',
                'Conditional triangulated depths are not physical depth ground truth.',
                'Local numerical rank cannot prove correct material identity or global camera uniqueness.',
                'The first-camera and baseline gauge remove seven similarity freedoms, not a real calibration reference.',
                'No statistical camera confidence interval is estimated from these measurements.'])
    scene_points = []
    for track in scene:
        r = results[track['track_id']]
        report['tracks'].append(dict(track_id=track['track_id'], status=r['status'], reason=r['reason'],
            held_error_px=r['held_error_px'], triangulation_angle_degrees=r['max_triangulation_angle_degrees']))
        if r['status'] == 'measured':scene_points.append(r['xyz'])
    # All candidate tracks, including unobservable/high-error ones, keep their
    # count and identity. Only a valid conditional XYZ can define a derivative.
    for frame in frames:
        observations = [o for t in scene for o in t['observations'] if o['frame'] == frame]
        depths = [p['depth_world_units']/baseline for t in scene for p in results[t['track_id']].get('projections', [])
                  if p['frame'] == frame and results[t['track_id']]['status'] == 'measured' and baseline > epsilon]
        uv = np.array([o['uv'] for o in observations], float).reshape(-1, 2)
        wh = cameras[frame]['wh']-1
        cells = {tuple(np.clip(np.floor(p/wh*4), 0, 3).astype(int)) for p in uv}
        spread = np.linalg.svd((uv-uv.mean(axis=0))/wh, compute_uv=False).tolist() if len(uv) else []
        report['frames'].append(dict(frame=frame, candidate_observations=len(observations),
            occupied_cells_4x4=len(cells), image_spread_singular_values=spread,
            conditional_depth_over_baseline=_summary(depths),
            depth_p10=float(np.quantile(depths,.1)) if depths else None))
        foreground_depths = [p['depth_world_units']/baseline for t in annotated['tracks'] if t['region'] in ('face','body')
            for p in results[t['track_id']].get('projections', []) if p['frame'] == frame and results[t['track_id']]['status'] == 'measured' and baseline > epsilon]
        report['subject_depth_comparison'].append(dict(frame=frame,
            scene=_summary(depths), foreground=_summary(foreground_depths),
            comparable=bool(depths and foreground_depths), metric_depth_verified=False))
    report['frames_without_scene_observations'] = [r['frame'] for r in report['frames'] if not r['candidate_observations']]
    report['scene_point_spread_over_baseline'] = (
        np.linalg.svd((np.asarray(scene_points)-np.mean(scene_points,axis=0))/baseline, compute_uv=False).tolist()
        if scene_points and baseline > epsilon else [])
    if baseline <= epsilon:
        report['state'] = 'no_nonzero_camera_baseline'
    else:
        report['information'] = _local_information(cameras,scene,results,baseline,anchor,shared)
        report['state'] = report['information']['state']
    return report
