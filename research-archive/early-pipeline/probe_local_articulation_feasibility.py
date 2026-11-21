"""Read-only local VPoser capacity diagnostic for two frozen stock portrait fits."""

import argparse
from copy import deepcopy
import fcntl
from hashlib import sha256
from importlib.util import find_spec
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation


LAB = Path('/opt/gs-slam-lab')
PROJECTS = Path('/mnt/d/Projects')
PROJECT = PROJECTS / 'Shuolee-smpl-source-visibility'
OUTPUT = LAB / 'shuolee-local-articulation-feasibility-005-008'
LOCK = Path('/tmp/shuolee-native122-gpu.lock')
LATENT_LIMIT = .025
ROTATION_LIMIT_DEG = 2.
RESIDUAL_SLACK_PX = .10
FACE_MOVE_LIMIT_PX = .50
OTHER_MOVE_LIMIT_PX = 1.
FD_STEP = .001
DEADLINE_SECONDS = 150
FIELDS = ('result.json', 'config.json', 'fit/official-fit-result.json', 'fit/selection.json',
          'fit/shared-body.json', 'run.json', '../preregistration.json', '../camera.json')
SOURCES = {
    '005': dict(run='coverage-005', project='view-coverage', protected=[11, 27, 42, 55],
        witness='pose-coverage-review/result.json',
        witness_sha='26c48a5831edf428ed63f3320d8aa3c57ce77320a9114b70c024a1615e526139',
        digests=(
            'f55b70846760238b595cf1cb00803254991d98bffd6caaa730c9007d38e71946',
            'ef46a20e1bacb905ef3d3a66a2749b2b88a046dfbd900ffd52a43208bc7c85bd',
            '08f8e551d9c06d5aeb8593018d54c429fb913bee8158ca95d9ea9a0f9d77ab13',
            '56d9af357c8442c23a6b0f8cfba2d1909cc0eacd855782e7cd33445d21dd3251',
            'd762b97063e9d4c1e4eb925f366ca50f6ee16d7da5e48427f7906249239073f8',
            'd9b693de3f5e4fe0476ef054b19ea6cc25a31e47793d00bc01d538c70d352673',
            '17cf148806f5186d6c9a26cd83f5617e845fb8e783f451508d9eb4d8962cfc1a',
            '65bc4ccfbf6d53f57cf7f47cc782ffe793d9000cab3580d7aa896c00d764fdd4')),
    '008': dict(run='visibility-008', project='source-visibility', protected=[15, 43, 71, 98],
        witness='../shuolee-smpl-visibility-008-upperbody/shoulder-span-readonly.json',
        witness_sha='e3f794f28f35f296cdb05db094224044838504e0c2dd2e8a68572eb9c6094480',
        digests=(
            '4f2605fca0b011798308fa414de604c3c3dc00dfcb5d4779d31d5338dc89aa84',
            '6ae1e685be00858eb0083b4dbff005a7b7a62ed3ca29ee2fa13011b706d397b8',
            'ead9039816741cb6c780ecc888fc139c6f826c7580a851344e12bfca45925ef0',
            '44e85e52a01642cda7c294f9afbb8ef4520f3e31b53cf218fadca0dcf5c0950d',
            '72672b67af8b3fcd6865df2b5343cdd900b6ddc228bc157c8522cd17abbeffc9',
            '1c9f671f046b659d980240ebe8326aa1e7dd58b6241a49192c3043b8f02fc77f',
            '06b7e045856c042efea1d235e34efc1d1e81b71381c7e59545ffba40955a5e1c',
            'b4e7a76076da210b57371ccc30d338655ddda2732b53b8823d607e6ab7385741')),
}
# Subject, sample, target COCO25 slot, admitted control, exact frozen winner.
CASES = (
    ('005', 13, 3, False, '654f853e0e4c6a78025d7b8efbe5d627441782f6b93aec94e7ce1e7cbc25ca4b'),
    ('005', 22, 3, False, 'c7f5a183efb388a51b718845cd7a5637db7871fb6bad6eb21a5b6d39d3a0e4f3'),
    ('005', 20, 3, True,  '2a304c4f59d31cd0217cd8d811d1a8fb01132f3a33f6f92a9047dc9432b079d3'),
    ('008', 11, 2, False, '77ddbec2d37f03eb1615b56a056cb3e316cf33a7b4660b8856e9339ba3cf02b8'),
    ('008', 106, 5, False, '7c650cc603b33a07404d40bf06c1258fbd101184bb5563bc4523a30aa3fe6aa5'),
    ('008', 4, 2, True,   'f94ab4d1062e22c4d8aedbaf40dab974a248f36fa40aa42f9267134927bc08c8'),
    ('008', 52, 5, True,  '9a56399098f9a22afa355ba021402e8ef8bfb1788f5cc000453971f613185e79'),
)
sys.path.insert(0, str(PROJECT / 'src'))
from shuolee_portrait import official_sequence as seq, stable_selection as screen


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frame_name(number):
    return f'frame_{number:06d}'


def prepare():
    """Bind sources and confirm original screen without executing the body model."""
    pins = {Path(__file__).resolve(): seq.rt.sha(__file__)}
    packed = []
    reports = []
    model_pairs = set()
    model_sources = ('src/avatar_core/shuolee_vposer.py',
                     'third_party/human_body_prior/human_body_prior/models/vposer_model.py')
    for subject, spec in SOURCES.items():
        base = LAB / ('shuolee-smpl-' + spec['run'])
        run = base / 'run'
        original_project = PROJECTS / ('Shuolee-smpl-' + spec['project'])
        bound = {(run / name).resolve(): digest for name, digest in zip(FIELDS, spec['digests'], strict=True)}
        witness_path = (base / spec['witness']).resolve()
        bound[witness_path] = spec['witness_sha']
        seq.check_bound(bound)
        terminal, cfg, fit, selection, shared, run_cfg, launch, camera = (
            seq.rt.read(run / name) for name in FIELDS)
        official = Path(cfg['official_smplifyx']['source_root'])
        assets = Path(run_cfg['assets'])
        model_pairs.add((official, assets))
        require(terminal['state'] == 'completed' and terminal['steps'] == 10_000
                and fit['new_fitting_started'] and fit['shared_identity'] and fit['selection'] == selection
                and fit['shared_body_sha256'] == bound[run / 'fit/shared-body.json']
                and cfg['capture_layout'] == camera['capture_layout'],
                f'{subject}: frozen result, fit or camera config changed.')
        bound.update({Path(name): digest for name, digest in fit['source']['input_files'].items()})
        bound.update({original_project / name: digest for name, digest in launch['code_sha256'].items()})
        bound.update({official / name: digest for name, digest in seq.official.PINNED.items()})
        bound.update({assets / row['path']: row['sha256']
                      for row in seq.rt.read(original_project / 'configs/server-assets.json')['files']
                      if row['path'].startswith(('smplx/', 'vposer/'))})
        bound[Path(launch['video'])] = launch['source_sha256']
        bound[run / 'fit/fitted.npz'] = fit['fit_sha256']
        bound[run / terminal['ply']] = terminal['sha256']
        for name in model_sources:
            bound[original_project / name] = seq.rt.sha(original_project / name)
            bound[PROJECT / name] = bound[original_project / name]
        seq.check_bound(bound)
        inputs = seq.load_inputs(run / 'data/inputs.json', run / 'data/observations.npz',
                                 camera_length_contract=cfg['camera_length_contract'],
                                 reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
        holds = [frame_name(n) for n in spec['protected']]
        require(inputs['confidence'] == fit['source']['confidence']
                and inputs['scope']['retained_holdouts'] == selection['retained_holdouts']
                == launch['expected_holdouts'] == holds
                and all(not inputs['targets'][inputs['ids'].index(f)].any() for f in holds),
                f'{subject}: original confidence or protected roster changed.')
        beta = np.asarray(shared['betas'], dtype=np.float32).reshape(1, 10)
        shared_latent = np.asarray(shared['pose_latent'], dtype=np.float32).reshape(1, 32)
        witness = seq.rt.read(witness_path)
        witnessed = {row['frame']: row for row in witness['frames' if subject == '005' else 'rows']}
        for code, sample, slot, control, winner_sha in CASES:
            if code != subject:
                continue
            frame = frame_name(sample)
            idx = inputs['ids'].index(frame)
            row = inputs['rows'][idx]
            target, K = inputs['targets'][idx], inputs['K'][idx]
            winner = run / 'fit/frames' / frame / 'winner-camera-space.npz'
            receipt = winner.with_name('result.json')
            bound.update({winner: winner_sha, receipt: seq.rt.sha(receipt),
                          Path(row['image']): row['source_sha256'], Path(row['mask']): row['mask_sha256']})
            seq.check_bound({winner: winner_sha})
            with np.load(winner, allow_pickle=False) as saved:
                arrays = dict(saved)
            report = seq.rt.read(receipt)
            seq.check_result(report, row)
            scores, flags, _, _ = screen._candidate(row, arrays, target, K, beta, None, report, True)
            old = selection['reasons'][frame]
            require(frame not in holds and frame in inputs['scope']['fit_frame_ids'] and slot in screen.BODY
                    and target[slot, 2] > 0 and scores == old['scores'] and flags == old['screen_flags'],
                    f'{subject}/{frame}: observed target or original gate differs.')
            if control:
                require(not flags and frame in selection['selected_frames'], 'Original control must be admitted.')
            else:
                evidence = witnessed[frame]
                check = (evidence['body_point'] == 'R_elbow' and evidence['winner_sha256'] == winner_sha
                         and evidence['source_sha256'] == row['source_sha256']) if subject == '005' else (
                    evidence['worst'] == slot and evidence['fit_winner_sha256'] == winner_sha
                    and evidence['sha256'] == row['source_sha256'])
                require(flags and check and old['status'] == 'pose_screen_rejection',
                        'Rejected target must match its original source witness.')
                if subject == '005':
                    bound[witness_path.parent / f'{frame}.jpg'] = evidence['image_sha256']
            packed.append((subject, row, target, K, arrays, report, beta, shared_latent, slot, control, scores, flags))
            reports.append(dict(subject=subject, frame=frame, source_index=row['source_frame_index'],
                                target_slot=slot, admitted_control=control, winner_sha256=winner_sha,
                                source_sha256=row['source_sha256'], target_sha256=sha256(target.tobytes()).hexdigest(),
                                native_K=K.tolist(), source_w2c=row['w2c'], protected_frames=holds,
                                support='Native source witness supports visible limb; exact anatomical center remains unverified.'
                                        if not control else 'Stock pose admitted under the unchanged screen.'))
        seq.check_bound(bound)
        for path, digest in bound.items():
            require(path not in pins or pins[path] == digest, f'Shared source differs: {path}')
            pins[path] = digest
    require(len(model_pairs) == 1 and len(packed) == len(CASES), 'Expected one common stock model source.')
    smplx_dir = Path(find_spec('smplx').origin).parent
    for name in ('body_models.py', 'lbs.py', 'vertex_joint_selector.py'):
        pins[smplx_dir / name] = seq.rt.sha(smplx_dir / name)
    seq.check_bound(pins)
    plan = dict(
        cases=reports, input_sha256={str(path): digest for path, digest in pins.items()},
        variable='32D stock VPoser latent only. Freeze all body named parameters, complete K and root translation.',
        protections='Per supported non-target detector point, allow residual increase at most 0.10 native px. '
                    'Cap non-target face displacement at 0.50 px and all other supported point displacement at 1.0 px. '
                    'Use the original body/feet gate verbatim. Face and hand limits are diagnostic protection. '
                    'Unsupported measurements, unknown depth and hidden anatomy remain outside physical validation.',
        trust=dict(latent_linf=LATENT_LIMIT, body_joint_rotation_deg=ROTATION_LIMIT_DEG,
                   finite_difference_max_step=FD_STEP, nonlinear_radial_prediction_max_error_px=.50),
        derivative_check='Exact FP64 promotion for the Jacobian and its finite difference. '
                         'Use the original direction and at most one quarter of the nearest decoder activation '
                         'crossing along it. Verify the endpoints remain in the same activation cell. '
                         'Check the original frozen replay and nonlinear trial with the stock FP32 model.',
        limits=dict(cases=len(CASES), cpu_seconds=DEADLINE_SECONDS, jacobians_per_case=1, linear_programs_per_case=2,
                    nonlinear_trial_forwards_per_case=1, optimizer_steps=0, gpu_calls=0),
        output_contract='Read-only source and stock training. Emit numeric observations outside training inputs.')
    return plan, packed, next(iter(model_pairs))


def local_step(projected, observed, jacobian, pose_jacobian, target, active):
    """One linear radial descent direction under fixed bounded support constraints."""
    residual = projected[active] - observed[active, :2]
    norms = np.linalg.norm(residual, axis=1)
    target_idx = int(np.where(active == target)[0][0])
    require(norms[target_idx] > 1e-6, 'Target residual has no defined radial direction.')
    radial = np.einsum('ni,nij->nj', residual / np.maximum(norms[:, None], 1e-6), jacobian)
    objective = radial[target_idx]
    # A componentwise axis-angle bound keeps each first-order 3-vector inside
    # 90% of the rotation cap. The nonlinear check uses geodesic rotation.
    angular = np.r_[pose_jacobian, -pose_jacobian]
    angular_limit = np.full(len(angular), .9 * np.deg2rad(ROTATION_LIMIT_DEG) / np.sqrt(3))
    bounds = [(-LATENT_LIMIT, LATENT_LIMIT)] * 32
    relaxed = linprog(objective, A_ub=angular, b_ub=angular_limit, bounds=bounds, method='highs')
    other = active != target
    protected = jacobian[other].reshape(-1, 32)
    move = np.where(active[other] >= 67, FACE_MOVE_LIMIT_PX, OTHER_MOVE_LIMIT_PX)
    # A zero residual has no radial gradient. Apply a stricter displacement bound there.
    cap = np.repeat(np.minimum(move, np.where(norms[other] < .10, .05, move)) / np.sqrt(2), 2)
    constrained = linprog(objective,
        A_ub=np.r_[angular, protected, -protected, radial[other]],
        b_ub=np.r_[angular_limit, cap, cap, np.full(other.sum(), .5 * RESIDUAL_SLACK_PX)],
        bounds=bounds, method='highs')
    require(relaxed.success and constrained.success, 'Bounded local linear feasibility problem failed.')
    return constrained.x, -float(relaxed.fun), -float(constrained.fun)


def project(body, vposer, intrinsics, translate, latent):
    angles = vposer.decode(latent.reshape(1, 32))['pose_body'].reshape(1, 63)
    mesh = body(body_pose=angles, return_verts=True, return_full_pose=True)
    vertices = mesh.vertices + translate[:, None]
    joints = mesh.joints + translate[:, None]
    q = joints @ intrinsics.T
    return angles, vertices, joints, q[..., :2] / q[..., 2:]


def replay_error(values, saved):
    keys = ('pose_rad', 'vertex_m', 'joint_m', 'uv_px')
    names = ('body_pose_axis_angle', 'vertices_camera', 'joints_camera', 'joints_projected_native')
    return {key: float(np.max(abs(value - saved[name])))
            for key, value, name in zip(keys, values, names, strict=True)}


def checked_directional_derivative(observation, vposer, origin, J, label):
    """Check one exact direction wholly within its stock decoder activation cell."""
    import torch

    prefix = list(vposer.decoder_net)[:4]
    require(tuple(type(layer).__name__ for layer in list(vposer.decoder_net)[:5])
            == ('Linear', 'LeakyReLU', 'Dropout', 'Linear', 'LeakyReLU')
            and not vposer.training, 'Expected the pinned VPoser eval decoder activation structure.')

    def preactivation(latent):
        first = prefix[0](latent.reshape(1, 32))
        second = prefix[3](prefix[2](prefix[1](first)))
        return torch.cat((first, second), dim=-1).reshape(-1)

    direction = torch.sin(torch.arange(1, 33, dtype=torch.float32))
    direction = (direction / direction.norm()).double()
    center, rate = torch.autograd.functional.jvp(preactivation, origin, direction)
    changing = rate != 0
    crossing = (center[changing] / rate[changing]).abs()
    nearest = float(crossing.min()) if crossing.numel() else None
    step = min(FD_STEP, .25 * nearest) if nearest is not None else FD_STEP
    require(step >= 1e-9, f'{label}: activation crossing too close for this derivative: '
                          f'nearest={nearest}, step={step}.')

    def crossings(h):
        sign = center > 0
        minus = (preactivation(origin - h * direction) > 0) != sign
        plus = (preactivation(origin + h * direction) > 0) != sign
        return int((minus | plus).sum())

    with torch.no_grad():
        nominal_crossings, chosen_crossings = crossings(FD_STEP), crossings(step)
        require(chosen_crossings == 0, f'{label}: derivative endpoints left the activation cell: '
                                      f'nearest={nearest}, step={step}, crossings={chosen_crossings}.')
        finite_difference = (observation(origin + step * direction)
                             - observation(origin - step * direction)).numpy() / (2 * step)
    analytical = J @ direction.numpy()
    error = abs(finite_difference - analytical)
    relative = float(np.linalg.norm(error) / max(1e-8, np.linalg.norm(analytical)))
    max_abs = float(error.max())
    check = dict(relative_error=relative, max_abs_error=max_abs, max_error_output=int(error.argmax()),
                 step=step, nearest_activation_crossing=nearest, nominal_crossings=nominal_crossings,
                 chosen_crossings=chosen_crossings, analytic_norm=float(np.linalg.norm(analytical)))
    require(relative <= .03 and max_abs <= .25, f'{label}: centered directional derivative differs: {check}')
    return check


def audit_case(body, vposer, diagnostic_body, diagnostic_vposer, case):
    import torch

    subject, row, target, K, arrays, report, beta, shared_latent, slot, control, base_scores, flags = case
    frame = row['frame']
    label = f'{subject}/{frame}'
    parameters = dict(body.named_parameters())
    frozen = {name: arrays[name] for name in parameters}
    body.reset_params(**frozen)
    diagnostic_body.reset_params(**frozen)
    origin = torch.from_numpy(arrays['pose_embedding']).reshape(32)
    intrinsics = torch.from_numpy(arrays['K_compute_float32'])
    translate = torch.from_numpy(arrays['camera_translation'])
    require(torch.equal(intrinsics, torch.as_tensor(K, dtype=torch.float32)), 'Frozen computation K differs.')
    with torch.no_grad():
        shared = vposer.decode(torch.from_numpy(shared_latent))['pose_body'].reshape(1, 63)
    require(sha256(shared.numpy().tobytes()).hexdigest() == report['shared_body_target_sha256'],
            'Frozen shared VPoser target does not replay.')
    active = np.flatnonzero(target[:, 2] > 0)
    active_tensor = torch.from_numpy(active)
    precise_origin = origin.double()
    precise_intrinsics = intrinsics.double()
    precise_translate = translate.double()

    def observation(latent):
        angles, _, _, uv = project(diagnostic_body, diagnostic_vposer, precise_intrinsics,
                                    precise_translate, latent)
        return torch.cat((uv[0].index_select(0, active_tensor).reshape(-1), angles.reshape(-1)))

    with torch.no_grad():
        stock_values = tuple(value.numpy() for value in project(body, vposer, intrinsics, translate, origin))
        precise_values = tuple(value.numpy() for value in project(
            diagnostic_body, diagnostic_vposer, precise_intrinsics, precise_translate, precise_origin))
    replay = dict(stock_fp32=replay_error(stock_values, arrays), diagnostic_fp64=replay_error(precise_values, arrays))
    for dtype, error in replay.items():
        require(error['uv_px'] <= .005 and max(error['vertex_m'], error['joint_m'], error['pose_rad']) <= 2e-6,
                f'{label}: {dtype} replay differs: {error}')
    angle0, _, _, uv0 = stock_values
    J = torch.autograd.functional.jacobian(observation, precise_origin, vectorize=False).numpy()
    require(J.shape == (2 * len(active) + 63, 32) and np.isfinite(J).all(), 'Unexpected VPoser Jacobian.')
    derivative = checked_directional_derivative(observation, diagnostic_vposer, precise_origin, J, label)
    J_uv = J[:-63].reshape(len(active), 2, 32)
    dx, relaxed, radial = local_step(uv0[0], target, J_uv, J[-63:], slot, active)
    with torch.no_grad():
        angle1, vert1, joint1, uv1 = (value.numpy() for value in project(
            body, vposer, intrinsics, translate, origin + torch.from_numpy(dx.astype(np.float32))))
    require(all(np.array_equal(parameters[name].detach().numpy(), arrays[name]) for name in parameters),
            'Frozen named body parameter changed.')
    pose0 = Rotation.from_rotvec(angle0.reshape(21, 3))
    pose1 = Rotation.from_rotvec(angle1.reshape(21, 3))
    moved = float(np.rad2deg((pose1 * pose0.inv()).magnitude()).max())
    e0 = np.linalg.norm(uv0[0] - target[:, :2], axis=1)
    e1 = np.linalg.norm(uv1[0] - target[:, :2], axis=1)
    movement = np.linalg.norm(uv1[0] - uv0[0], axis=1)
    other = active[active != slot]
    face = other[other >= 67]
    remaining = other[other < 67]
    max_other_growth = float((e1[other] - e0[other]).max())
    max_face_move = float(movement[face].max(initial=0.))
    max_other_move = float(movement[remaining].max(initial=0.))
    joint_candidate = dict(arrays, vertices_camera=vert1, joints_camera=joint1, joints_projected_native=uv1,
                           body_pose_axis_angle=angle1)
    scores1, flags1, _, _ = screen._candidate(row, joint_candidate, target, K, beta, None, report, True)
    actual = float(e0[slot] - e1[slot])
    preserves = (moved <= ROTATION_LIMIT_DEG and max_other_growth <= RESIDUAL_SLACK_PX
                 and max_face_move <= FACE_MOVE_LIMIT_PX and max_other_move <= OTHER_MOVE_LIMIT_PX
                 and abs(actual - radial) <= .50)
    height = row['initial_box_xyxy'][3] - row['initial_box_xyxy'][1]
    shared_rot = Rotation.from_rotvec(shared.numpy().reshape(21, 3)).as_matrix()
    pose_prior = [float(30. * .5 * np.square(x.as_matrix() - shared_rot).sum()) for x in (pose0, pose1)]
    return dict(subject=subject, frame=frame, admitted_control=control, target_slot=slot,
                target_residual_stock_trial_px=[float(e0[slot]), float(e1[slot])],
                target_confidence=float(target[slot, 2]), original_body_threshold_px=.03 * height,
                relaxed_first_order_radial_gain_px=relaxed, protected_first_order_radial_gain_px=radial,
                realized_radial_gain_px=actual, nonlinear_protections_passed=preserves,
                new_pose_screen_pass_under_protections=bool(flags and not flags1 and preserves),
                stock_screen_flags=flags, trial_screen_flags=flags1, stock_region_scores=base_scores, trial_region_scores=scores1,
                largest_supported_non_target_residual_growth_px=max_other_growth,
                max_supported_face_movement_px=max_face_move, max_other_supported_movement_px=max_other_move,
                max_body_joint_rotation_deg=moved, latent_linf=float(np.max(abs(dx))),
                latent_l2=float(np.linalg.norm(dx)), shared_rotation_penalty_stock_trial=pose_prior,
                replay_error=replay, directional_jacobian_relative_error=derivative['relative_error'],
                directional_jacobian_check=derivative,
                jacobian_sha256=sha256(J.tobytes()).hexdigest())


def stop_on_budget(_signal, _frame):
    raise TimeoutError('Local articulation CPU diagnostic reached its 150-second budget.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'probe'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == LAB and not output.exists(), 'Use a new output directly under the local lab.')
    plan, packed, (official, assets) = prepare()
    if args.mode == 'check':
        print(json.dumps(dict(state='source_and_stock_pins_verified', cases=plan['cases'],
                              pins=len(plan['input_sha256']), body_forward_calls=0, gpu_calls=0)))
        return
    import torch

    with LOCK.open('a+b') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir()
        seq.rt.save(output / 'preregistration.json', plan)
        start = perf_counter()
        records = []
        previous = signal.signal(signal.SIGALRM, stop_on_budget)
        try:
            signal.alarm(DEADLINE_SECONDS)
            seq.rt.configure(assets)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            body, vposer, _ = seq.official.create_models(
                official_root=official, assets=assets, device='cpu', static_topology=False)
            body.eval().requires_grad_(False)
            diagnostic_body, diagnostic_vposer = deepcopy(body).double(), deepcopy(vposer).double()
            for case in packed:
                records.append(audit_case(body, vposer, diagnostic_body, diagnostic_vposer, case))
                print('CASE_COMPLETE', records[-1]['subject'], records[-1]['frame'], flush=True)
            seq.check_bound(plan['input_sha256'])
        except BaseException as exc:
            seq.rt.save(output / 'failure.json', dict(state='failed', error=str(exc), records=records,
                                                     elapsed_seconds=perf_counter()-start))
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        seq.rt.save(output / 'result.json', dict(
            state='local_feasibility_measured', elapsed_seconds=perf_counter()-start, records=records,
            preregistration_sha256=seq.rt.sha(output / 'preregistration.json'), optimizer_steps=0, gpu_calls=0,
            fit_artifacts_created=0, quality_accepted=False))


if __name__ == '__main__':
    main()
