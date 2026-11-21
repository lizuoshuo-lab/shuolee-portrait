"""Fixed-camera, held-view point diagnostics, not a calibration solver.

Input correspondences must already have a declared identity and split. No image
matching, camera optimization, residual-based selection, or training admission
is performed here. A held observation is not independent camera validation if
it helped estimate those cameras. Even a zero residual does not prove material
identity, subject staticity, or physical calibration.
"""
import numpy as np


PIXEL_DOMAIN = 'undistorted_integer_centers'


def cameras_from_records(records):
    cameras = {}
    for row in records:
        frame = row['frame']
        if not isinstance(frame, str) or not frame or frame in cameras:
            raise ValueError('Camera frame IDs must be unique nonempty strings.')
        K, E = np.asarray(row['K'], float), np.asarray(row['w2c'], float)
        wh = np.asarray(row['wh'], float)
        if K.shape != (3, 3) or E.shape != (4, 4) or wh.shape != (2,):
            raise ValueError(f'Invalid camera shape: {frame}')
        if not all(np.isfinite(x).all() for x in (K, E, wh)):
            raise ValueError(f'Non-finite camera: {frame}')
        if (wh < 2).any() or (wh != np.floor(wh)).any():
            raise ValueError(f'Invalid image dimensions: {frame}')
        if (K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose(K[2], [0, 0, 1], atol=1e-10, rtol=0) or
                abs(K[0, 1]) > 1e-10 or abs(K[1, 0]) > 1e-10):
            raise ValueError(f'Expected an undistorted pinhole K: {frame}')
        if (not np.allclose(E[3], [0, 0, 0, 1], atol=1e-10, rtol=0) or
                not np.allclose(E[:3, :3].T @ E[:3, :3], np.eye(3), atol=1e-6, rtol=0) or
                abs(np.linalg.det(E[:3, :3]) - 1) > 1e-6):
            raise ValueError(f'Expected a rigid world-to-camera matrix: {frame}')
        image_hash = row.get('source_sha256')
        if not isinstance(image_hash, str) or len(image_hash) != 64:
            raise ValueError(f'Missing source image SHA256: {frame}')
        try:
            int(image_hash, 16)
        except ValueError as exc:
            raise ValueError(f'Invalid image SHA256: {frame}') from exc
        cameras[frame] = dict(K=K, E=E, wh=wh, source_sha256=image_hash,
                              center=-E[:3, :3].T @ E[:3, 3])
    if len(cameras) < 3:
        raise ValueError('Held-view diagnostics need at least three cameras.')
    return cameras


def project(camera, xyz):
    """COLMAP corner-origin K to integer-centre pixels: subtract 0.5 once."""
    point = camera['E'][:3, :3] @ np.asarray(xyz, float) + camera['E'][:3, 3]
    depth = float(point[2])
    if not np.isfinite(point).all() or depth <= 0:
        return None, depth
    pixel = camera['K'] @ point
    return pixel[:2] / pixel[2] - 0.5, depth


def _summary(values):
    a = np.asarray(values, float)
    if not len(a):
        return dict(count=0, median=None, p90=None, maximum=None)
    if not np.isfinite(a).all():
        raise ValueError('A non-finite residual cannot enter aggregate statistics.')
    return dict(count=len(a), median=float(np.median(a)),
                p90=float(np.quantile(a, .9)), maximum=float(a.max()))


def _track(cameras, track, min_parallax_degrees, camera_fit_ids):
    observations = track['observations']
    frames = [o['frame'] for o in observations]
    ids = [o['observation_id'] for o in observations]
    fit_frames = track['triangulation_frames']
    if len(frames) < 3 or len(set(frames)) != len(frames):
        raise ValueError(f'Track {track["track_id"]} needs distinct observations in at least three views.')
    if len(fit_frames) < 2 or len(set(fit_frames)) != len(fit_frames):
        raise ValueError('Triangulation frames must be a distinct preregistered subset.')
    if not set(fit_frames) < set(frames):
        raise ValueError('At least one view must be held out of point triangulation.')
    if track.get('identity_basis') not in ('material_candidate', 'semantic_landmark', 'unknown'):
        raise ValueError('Declare material_candidate, semantic_landmark or unknown identity basis.')
    if track.get('region') not in ('static_scene_candidate', 'face', 'clothing', 'body', 'unknown'):
        raise ValueError('Declare the region without assuming that all background is static.')
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Observation IDs must be unique nonempty strings within a track.')
    by_frame = {}
    for observation in observations:
        frame = observation['frame']
        if frame not in cameras:
            raise ValueError(f'Unknown camera frame {frame}')
        camera = cameras[frame]
        if observation['image_sha256'] != camera['source_sha256']:
            raise ValueError(f'Observation image identity mismatch: {frame}')
        uv = np.asarray(observation['uv'], float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError(f'Invalid source pixels: {frame}')
        if (uv < 0).any() or (uv > camera['wh'] - 1).any():
            raise ValueError(f'Observation outside integer-centre image domain: {frame}')
        by_frame[frame] = uv
    overlap = None if camera_fit_ids is None else sorted(set(ids) & camera_fit_ids)
    result = dict(track_id=track['track_id'], region=track['region'],
                  identity_basis=track['identity_basis'],
                  triangulation_frames=list(fit_frames),
                  held_frames=[f for f in frames if f not in fit_frames],
                  camera_fit_observation_overlap=overlap,
                  camera_validation_independence='unknown' if overlap is None else
                      ('not_independent' if overlap else 'id_disjoint_only_unverified'),
                  physical_material_identity_verified=False,
                  cause_identified=False, status='not_observable',
                  xyz=None, fit_error_px=[], held_error_px=[])
    A, rays, centers = [], [], []
    for frame in fit_frames:
        camera = cameras[frame]
        ray = np.linalg.solve(camera['K'], np.r_[by_frame[frame] + .5, 1.])
        ray /= ray[2]
        E = camera['E'][:3]
        A.extend([ray[0] * E[2] - E[0], ray[1] * E[2] - E[1]])
        world_ray = camera['E'][:3, :3].T @ ray
        rays.append(world_ray / np.linalg.norm(world_ray))
        centers.append(camera['center'])
    angles, baselines = [], []
    for i in range(len(rays)):
        for j in range(i):
            angles.append(np.degrees(np.arctan2(np.linalg.norm(np.cross(rays[i], rays[j])),
                                                np.dot(rays[i], rays[j]))))
            baselines.append(np.linalg.norm(centers[i] - centers[j]))
    result['max_triangulation_angle_degrees'] = float(max(angles))
    result['max_camera_baseline_world_units'] = float(max(baselines))
    epsilon = np.finfo(float).eps * max(1., np.max(np.abs(centers))) * 64
    if max(baselines) <= epsilon:
        result['reason'] = 'coincident_camera_centers'
        return result
    if max(angles) < min_parallax_degrees:
        result['reason'] = 'insufficient_triangulation_angle'
        return result
    _, singular, vectors = np.linalg.svd(np.asarray(A), full_matrices=False)
    point = vectors[-1]
    if abs(point[3]) < np.finfo(float).eps * 64:
        result['reason'] = 'point_at_infinity'
        return result
    xyz = point[:3] / point[3]
    result['dlt_singular_values'] = singular.tolist()
    result['xyz'] = xyz.tolist()
    projections = []
    for frame in frames:
        uv, depth = project(cameras[frame], xyz)
        projections.append(dict(frame=frame, depth_world_units=depth,
                                observed_uv=by_frame[frame].tolist(),
                                predicted_uv=None if uv is None else uv.tolist(),
                                residual_px=None if uv is None else float(np.linalg.norm(uv - by_frame[frame]))))
    result['projections'] = projections
    if any(p['predicted_uv'] is None for p in projections):
        result['reason'] = 'non_positive_depth'
        return result
    result.update(status='measured', reason='conditional_on_given_cameras_and_correspondence',
                  fit_error_px=[p['residual_px'] for p in projections if p['frame'] in fit_frames],
                  held_error_px=[p['residual_px'] for p in projections if p['frame'] not in fit_frames])
    return result


def evaluate_fixed_tracks(camera_records, bank, min_parallax_degrees=1.0):
    """No threshold tuning or residual pruning; all input track outcomes persist."""
    if bank.get('schema') != 'shuolee-fixed-tracks-v1' or bank.get('pixel_domain') != PIXEL_DOMAIN:
        raise ValueError('Explicit v1 track schema and undistorted integer-centre domain required.')
    if not np.isfinite(min_parallax_degrees) or not 0 < min_parallax_degrees < 90:
        raise ValueError('A declared positive triangulation-angle minimum below 90 degrees is required.')
    cameras = cameras_from_records(camera_records)
    tracks = bank['tracks']
    if not tracks:
        raise ValueError('An empty bank cannot produce a passing diagnostic.')
    track_ids = [t['track_id'] for t in tracks]
    if any(not isinstance(i, str) or not i for i in track_ids) or len(set(track_ids)) != len(track_ids):
        raise ValueError('Track IDs must be unique nonempty strings.')
    all_observations = [o['observation_id'] for t in tracks for o in t['observations']]
    if len(set(all_observations)) != len(all_observations):
        raise ValueError('The same observation was duplicated across tracks.')
    source_locations = [(o['frame'], tuple(o['uv'])) for t in tracks for o in t['observations']]
    if len(set(source_locations)) != len(source_locations):
        raise ValueError('One source pixel was duplicated under different observation IDs.')
    fit_ids = bank.get('camera_fit_observation_ids')
    if fit_ids is not None:
        if not isinstance(fit_ids, list) or any(not isinstance(i, str) or not i for i in fit_ids):
            raise ValueError('Camera-fit usage must be an explicit ID list or null for unknown.')
        fit_ids = set(fit_ids)
    results = [_track(cameras, t, min_parallax_degrees, fit_ids) for t in tracks]
    measured = [t for t in results if t['status'] == 'measured']
    return dict(schema='shuolee-fixed-track-diagnostic-v1', pixel_domain=PIXEL_DOMAIN,
                camera_count=len(cameras), track_count=len(tracks), measured_tracks=len(measured),
                not_observable_tracks=len(results) - len(measured),
                min_parallax_degrees=float(min_parallax_degrees), tracks=results,
                held_error_px=_summary([e for t in measured for e in t['held_error_px']]),
                shared_point_error_definition='Point fitted only in the listed triangulation views; prediction in the held views. Cameras stay unchanged.',
                independent_camera_validation=False, quality_accepted=False,
                root_cause_resolved=False, training_admitted=False, cameras_optimized=False,
                limits=[
                    'Source hashes and ID disjointness do not prove that two observations are the same material point.',
                    'Point-view holdout does not remove leakage through cameras estimated from the same observations.',
                    'Camera uncertainty, wrong matches and subject motion can yield indistinguishable residuals.',
                    'No zero-residual result establishes physical K, world scale or a ghosting-free reconstruction.'])
