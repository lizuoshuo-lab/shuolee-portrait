"""C0a: read-only, bounded Jacobian analysis of the failed 121 camera graphs.

This does not solve for, save, or apply any camera/point/intrinsic update.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

BASE = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current')
R = BASE / 'robust-portrait-122'
OLD = BASE / 'raw-domain-121'
DATA = Path('/opt/gs-slam-lab/shuolee-raw-domain-121')
OUT = R / 'camera-observability'
SCALES = [0.001, 0.001, 1.0, 1.0]
NAMES = ['log_fx', 'log_fy', 'cx', 'cy']
TOL = 1e-10


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def imports():
    global np, svd, Rotation
    import numpy as np
    from scipy.linalg import svd
    from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def project(K, rot, center, point):
    q = rot @ (point - center)
    return np.array([K[0, 0] * q[0] / q[2] + K[0, 2], K[1, 1] * q[1] / q[2] + K[1, 2]])


def derivatives(K, rot, center, point):
    q = rot @ (point - center)
    x, y, z = q
    require(z > 0 and np.all(np.isfinite(q)), 'Nonpositive or nonfinite observed point depth')
    jp = np.array([[K[0, 0]/z, 0., -K[0, 0]*x/z**2], [0., K[1, 1]/z, -K[1, 1]*y/z**2]])
    jr = jp @ -skew(q)
    jc = -jp @ rot
    jx = jp @ rot
    jk = np.array([[K[0, 0]*x/z, 0., 1., 0.], [0., K[1, 1]*y/z, 0., 1.]])
    return jr, jc, jx, jk, z


def fixture():
    # Fixed before any real-data Jacobian is evaluated; no epsilon/tolerance sweep.
    K = np.array([[1120., 0., 800.3], [0., 1060., 640.8], [0., 0., 1.]])
    rot = Rotation.from_rotvec([0.12, -0.17, 0.06]).as_matrix()
    center = np.array([0.2, -0.1, 0.3])
    point = np.array([0.6, 0.4, 4.2])
    jr, jc, jx, jk, _ = derivatives(K, rot, center, point)
    analytic = np.column_stack([jr, jc, jx, jk])
    numeric = np.zeros_like(analytic)
    epsilons = [1e-6]*11 + [1e-4]*2
    for axis, eps in enumerate(epsilons):
        ys = []
        for sign in [-1., 1.]:
            kk, rr, cc, xx = K.copy(), rot.copy(), center.copy(), point.copy()
            v = sign*eps
            if axis < 3:
                d = np.zeros(3); d[axis] = v
                rr = Rotation.from_rotvec(d).as_matrix() @ rot
            elif axis < 6:
                cc[axis-3] += v
            elif axis < 9:
                xx[axis-9] += v
            elif axis < 11:
                kk[axis-9, axis-9] *= np.exp(v)
            else:
                kk[axis-11, 2] += v
            ys.append(project(kk, rr, cc, xx))
        numeric[:, axis] = (ys[1]-ys[0])/(2*eps)
    errors = np.max(np.abs(numeric-analytic), axis=0)
    result = dict(updated=now(), passed=bool(errors.max() < 1e-5),
                  convention='R_prime=Exp(delta_rotation) R; additive camera center and world point; integer pixel centers',
                  epsilons=epsilons, maximum_absolute_derivative_error=float(errors.max()),
                  error_by_column=errors.tolist(), limit=1e-5,
                  columns=['rx', 'ry', 'rz', 'Cx', 'Cy', 'Cz', 'X', 'Y', 'Z']+NAMES,
                  cameras_changed=False, real_calibration_proven=False)
    save(OUT/'jacobian-fixture.json', result)
    require(result['passed'], 'Analytic Jacobian fixture failed; real-data diagnostic not started')


def rank(s):
    return int(np.sum(s > TOL*s[0])) if len(s) and s[0] > 0 else 0


def analyze(subject, result, paths):
    source = DATA/subject
    paths.update({
        'observations': source/'roi-supplement/observations.json',
        'contract': source/'global-fixedk/input-contract.json',
        'cameras': source/'global-fixedk/cameras-diagnostic.json',
        'structure': source/'global-fixedk/background-structure.npz',
        'baseline_result': source/'global-fixedk/result.json',
        'family_audit': OLD/f'supplement-{subject}-track-family-audit.json',
        'global_plan': OLD/'global-fixedk-plan.json',
        'global_source': Path(__file__).parent/'global_fixedk121.py',
    })
    result['input_hashes_before'] = {k: sha(p) for k, p in paths.items()}
    obs, contract, cameras, baseline, family = [read(paths[k]) for k in ['observations', 'contract', 'cameras', 'baseline_result', 'family_audit']]
    require(contract['input_sha256']['observations'] == result['input_hashes_before']['observations'], 'Frozen observation SHA mismatch')
    require(contract['input_sha256']['family_audit'] == result['input_hashes_before']['family_audit'], 'Frozen family audit SHA mismatch')
    require(contract['plan_sha256'] == result['input_hashes_before']['global_plan'], 'Frozen 121 plan SHA mismatch')
    require(contract['script_sha256'] == result['input_hashes_before']['global_source'], 'Frozen 121 source SHA mismatch')
    require(obs['pixel_domain'] == 'native_already_processed_export_integer_centers', 'Unsupported pixel domain')
    require(not obs['feature_images_resized'], 'Resized feature observations violate contract')
    records = obs['records']; tracks = {t['track_id']: t for t in obs['tracks']}
    require(len(tracks) == len(obs['tracks']), 'Duplicate track identity')
    frame_lookup = {r['frame']: i for i, r in enumerate(records)}
    require(len(frame_lookup) == len(records), 'Duplicate source frame identity')
    frozen = contract['fit_track_ids']; retained = set(family['retained_track_ids']); quarantined = set(family['quarantined_track_ids'])
    require(len(frozen) == len(set(frozen)), 'Duplicate frozen FIT identity')
    require(set(frozen) <= retained and not set(frozen) & quarantined, 'Frozen FIT/quarantine mismatch')
    require(all(t in tracks and tracks[t]['split'] == 'fit' for t in frozen), 'Heldout or missing identity in frozen FIT')
    with np.load(paths['structure'], allow_pickle=False) as data:
        tids, xyz = data['track_ids'].astype(str).tolist(), data['xyz'].copy()
    require(len(tids) == len(set(tids)) and set(tids) <= set(frozen), 'Structure not unique frozen FIT only')
    require(xyz.shape == (len(tids), 3) and np.isfinite(xyz).all(), 'Invalid saved structure')
    require([c['frame'] for c in cameras] == baseline['registered_frames'], 'Registered frame identity mismatch')
    require(len(cameras) == baseline['registered_count'] and len(tids) == baseline['background_points'], 'Baseline cardinality mismatch')
    K = np.array(contract['K_integer_centers'], dtype=float)
    require(K.shape == (3, 3) and np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0, 'Invalid K')
    require(K[0, 1] == 0 and K[1, 0] == 0 and np.array_equal(K[2], [0, 0, 1]), 'Non-pinhole K')
    require(np.array_equal(K, baseline['K_integer_centers']), 'Baseline K mismatch')
    indices = [c['frame_index'] for c in cameras]
    require(indices == sorted(set(indices)), 'Camera indices duplicate or unsorted')
    poses = {}; image_hashes = {}
    for camera in cameras:
        i = camera['frame_index']; record = records[i]
        require(record['frame'] == camera['frame'] and not record['additional_geometric_resampling'], 'Frame/pixel identity mismatch')
        require(np.array_equal(camera['K_integer_centers'], K), 'Camera K differs')
        rr, cc = np.array(camera['R_w2c']), np.array(camera['C_world'])
        require(rr.shape == (3, 3) and cc.shape == (3,) and np.isfinite(rr).all() and np.isfinite(cc).all(), 'Invalid saved pose')
        require(np.max(np.abs(rr @ rr.T-np.eye(3))) < 1e-8 and abs(np.linalg.det(rr)-1) < 1e-8, 'Invalid saved SO3')
        poses[i] = (rr, cc)
        key = 'source_image_'+record['frame']; paths[key] = Path(record['image'])
        image_hashes[key] = sha(paths[key])
        require(image_hashes[key] == record['source_sha256'], 'Registered source image SHA mismatch')
    result['input_hashes_before'].update(image_hashes)
    gauge = baseline.get('bundle_adjustment', {}).get('gauge')
    require(gauge is not None, 'Saved baseline gauge missing; no invented gauge')
    first, seed, axis = gauge['first_frame'], gauge['seed_frame'], gauge['seed_center_component']
    require(first == 0 and first in poses and seed in poses and axis in [0, 1, 2], 'Unsupported saved gauge')
    require(np.array_equal(poses[first][0], np.eye(3)) and np.array_equal(poses[first][1], np.zeros(3)), 'First pose was not fixed')
    require(poses[seed][1][axis] == gauge['fixed_value'] and abs(gauge['fixed_value']) > 0.1, 'Saved scale gauge mismatch')
    columns = [(i, j) for i in indices for j in range(6) if i != first and not (i == seed and j == 3+axis)]
    column_lookup = {v: k for k, v in enumerate(columns)}
    all_a, all_b, raw_b, depths, residuals, track_rows, point_ranks = [], [], [], [], [], [], []
    for tid, point in zip(tids, xyz):
        track = tracks[tid]
        require(len({o['frame'] for o in track['observations']}) == len(track['observations']), 'Duplicate frame in track')
        observations = [o for o in track['observations'] if frame_lookup[o['frame']] in poses]
        require(len(observations) >= 2, 'Saved point has insufficient registered observations')
        aa = np.zeros((2*len(observations), len(columns))); bb = np.zeros((len(aa), 4)); xx = np.zeros((len(aa), 3))
        for row, observation in enumerate(observations):
            i = frame_lookup[observation['frame']]; record = records[i]; rr, cc = poses[i]
            uv = np.array(observation['uv'])
            require(uv.shape == (2,) and np.isfinite(uv).all(), 'Nonfinite native observation')
            require(np.all(uv >= 0) and np.all(uv < np.array(record['wh'])), 'Observation outside native frame')
            require(observation['image_sha256'] == record['source_sha256'], 'Observation image identity mismatch')
            jr, jc, jx, jk, depth = derivatives(K, rr, cc, point)
            camera_jac = np.column_stack([jr, jc]); rows = slice(2*row, 2*row+2)
            for j in range(6):
                if (i, j) in column_lookup:
                    aa[rows, column_lookup[(i, j)]] = camera_jac[:, j]
            bb[rows] = jk*np.array(SCALES); xx[rows] = jx
            depths.append(float(depth)); residuals.append(float(np.linalg.norm(project(K, rr, cc, point)-uv)))
        u, s, _ = svd(xx, full_matrices=False); rrk = rank(s); basis = u[:, :rrk]
        all_a.append(aa-basis @ (basis.T @ aa)); all_b.append(bb-basis @ (basis.T @ bb)); raw_b.append(bb)
        point_ranks.append(rrk)
        track_rows.append(dict(track_id=tid, split='fit', registered_frames=[o['frame'] for o in observations], observation_count=len(observations)))
    A, B, raw = np.vstack(all_a), np.vstack(all_b), np.vstack(raw_b)
    column_norms = np.linalg.norm(A, axis=0)
    require(np.all(column_norms > 0), 'Zero-support free camera direction; no silent removal')
    normalized = A/column_norms
    u, s, _ = svd(normalized, full_matrices=False); camera_rank = rank(s); basis = u[:, :camera_rank]
    remainder = B-basis @ (basis.T @ B)
    orthogonality = float(np.max(np.abs(basis.T @ remainder)))
    require(orthogonality < 1e-8*max(1., float(np.linalg.norm(B))), 'Projection orthogonality control failed')
    require(np.linalg.norm(remainder) <= np.linalg.norm(B)*(1+1e-10) and np.linalg.norm(B) <= np.linalg.norm(raw)*(1+1e-10), 'Projection increased norm')
    nobs = len(depths); ds = []
    for i, name in enumerate(NAMES):
        nr, npnt, nall = [float(np.linalg.norm(j[:, i])) for j in [raw, B, remainder]]
        ds.append(dict(parameter=name, parameter_delta=SCALES[i], raw_l2_px=nr, after_point_l2_px=npnt,
                       after_point_pose_l2_px=nall, raw_rms_vector_px=nr/np.sqrt(nobs),
                       after_point_pose_rms_vector_px=nall/np.sqrt(nobs), remaining_amplitude_fraction=nall/nr,
                       remaining_energy_fraction=(nall/nr)**2))
    _, ks, kv = svd(remainder, full_matrices=False)
    result.update(state='completed_local_linear_diagnostic',
                  identity_checks_passed=True, gauge=gauge, saved_K_integer_centers=K.tolist(),
                  registered_frames=[c['frame'] for c in cameras], frame_count=len(cameras),
                  full_video_registration=False, diagnostic_subset_frame_count=len(records),
                  frozen_fit_tracks=len(frozen), saved_fit_points=len(tids), observation_count=nobs,
                  included_all_registered_observations=True, excluded_observations_by_residual=0, heldout_observations_used=0,
                  depth_range=[min(depths), max(depths)],
                  saved_fit_residual_px={'max':max(residuals), 'rms':float(np.sqrt(np.mean(np.square(residuals)))), 'independent_validation':False},
                  physical_material_identity_certified=False,
                  material_warning='These are saved candidate background appearances. Sampled review is not complete physical identity truth.',
                  source_optimizer_converged=baseline['bundle_adjustment']['optimizer_success'],
                  point_jacobian_ranks={str(k):point_ranks.count(k) for k in sorted(set(point_ranks))},
                  camera_nuisance_columns=len(columns), camera_nuisance_rank=camera_rank,
                  normalized_camera_singular_extrema=[float(s[0]), float(s[-1])],
                  camera_column_norm_range=[float(column_norms.min()),float(column_norms.max())],
                  projection_orthogonality_max=orthogonality,
                  directions=ds, scaled_intrinsic_singular_values=ks.tolist(), scaled_intrinsic_rank=rank(ks),
                  scaled_intrinsic_right_vectors=kv.tolist(), smallest_to_largest_signature=float(ks[-1]/ks[0]),
                  interpretation='Fraction remaining is a local derivative amplitude after nuisance projection, not a measured error reduction, probability, or calibrated uncertainty. No parameter increment is estimated or applied.',
                  parameter_updates_written=False, face_subpixel_admission=False)
    save(OUT/subject/'used-identities.json', track_rows)


def worker(subject):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
    imports(); start = time.monotonic(); paths = {}
    result = dict(subject=subject, started=now(), pid=os.getpid(), attempt=1,
                  script_sha256=sha(Path(__file__)), plan_sha256=sha(R/'camera-observability-plan.json'),
                  implementation_sha256=sha(R/'camera-observability-implementation.json'),
                  training_started=False, K_changed=False, cameras_changed=False, points_changed=False)
    try:
        analyze(subject, result, paths)
    except Exception as e:
        result.update(state='stopped_with_evidence', error=repr(e), traceback=traceback.format_exc())
    finally:
        before = result.get('input_hashes_before', {})
        after = {k:sha(p) for k,p in paths.items() if p.is_file()}
        result['input_hashes_after'] = after
        result['inputs_unchanged'] = bool(before and all(after.get(k) == v for k,v in before.items()))
        if not result['inputs_unchanged']:
            result['state'] = 'stopped_input_hash_validation'
        result.update(updated=now(), elapsed_seconds=time.monotonic()-start)
        save(OUT/subject/'result.json', result)
    print(json.dumps({k:result.get(k) for k in ['subject','state','error','frame_count','saved_fit_points','observation_count','directions','scaled_intrinsic_singular_values','elapsed_seconds','inputs_unchanged']}, allow_nan=False), flush=True)
    return 0 if result['state'] == 'completed_local_linear_diagnostic' else 2


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', choices=['sofa','portrait'])
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    plan = read(R/'camera-observability-plan.json'); implementation = read(R/'camera-observability-implementation.json')
    require(implementation['worker_sha256'] == sha(Path(__file__)), 'Implementation source changed after registration')
    require(sha(OLD/'global-material-review.json') == plan['source_material_review_sha256'], 'Material review changed')
    require(sha(OLD/'global-fixedk-analysis.json') == plan['source_analysis_sha256'], 'Baseline analysis changed')
    OUT.mkdir(exist_ok=False)
    progress = dict(state='running', pid=os.getpid(), command=sys.argv, started=now(), GPU=False, training_started=False, subjects={})
    save(OUT/'progress.json', progress)
    imports(); fixture()
    results = []
    env = os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='')
    for subject in plan['subjects']:
        (OUT/subject).mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', subject]
        with (OUT/subject/'run.log').open('wb') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            progress['subjects'][subject] = dict(state='running', pid=proc.pid, command=command, started=now())
            save(OUT/'progress.json', progress)
            try:
                rc = proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait(); rc = -9
                # Preserve the attempt and prohibit reruns if a budget is exhausted.
                if not (OUT/subject/'result.json').exists():
                    save(OUT/subject/'result.json', dict(subject=subject, state='stopped_120_second_budget', no_retry=True, training_started=False))
        require((OUT/subject/'result.json').exists(), f'{subject} worker exited without result: {rc}; inspect retained log')
        result = read(OUT/subject/'result.json'); results.append(result)
        progress['subjects'][subject].update(state=result['state'], returncode=rc)
        save(OUT/'progress.json', progress)
    for subject in ['grass','indoor']:
        results.append(dict(subject=subject, state='unsupported_no_valid_camera_graph', no_invented_pose=True, training_started=False))
    save(OUT/'result.json', dict(updated=now(), stage='C0a', subjects=results, training_started=False,
         conclusion_scope='Read-only local observability at failed121 saved estimates; no K/camera candidate or independent physical calibration.'))
    progress.update(state='completed_review_pending', completed=now()); save(OUT/'progress.json', progress)
    print(json.dumps({'stage':'C0a','subjects':[{k:r.get(k) for k in ['subject','state','observation_count','directions','scaled_intrinsic_singular_values','inputs_unchanged']} for r in results]}, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
