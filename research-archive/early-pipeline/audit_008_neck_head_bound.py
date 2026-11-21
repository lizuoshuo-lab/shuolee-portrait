"""Frozen stock SMPL-X sensitivity to a bounded neck/head pose correction."""

from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

import audit_008_left_ear_local_bound as prior
from audit_008_head_ownership import FLAME, MAP


head = prior.head
OUT = head.LAB / 'shuolee-smpl-neck-head-bound-008'
BASE = head.LAB / 'shuolee-smpl-head-local-bound-008/result.json'
JOINT_SLICES = (slice(33, 36), head.AXIS)
HEAD_OBSERVATIONS = {0, 15, 16, 17, 18}
LIMIT_RAD = np.deg2rad(15.)
DIFFERENCE = .002
REGION_MARGIN = .85
OTHER_POINT_LIMIT = 2.


def forward(model, pose, K, step):
    import torch

    aa = pose['body_pose_axis_angle'].copy()
    for s, d in zip(JOINT_SLICES, step.reshape(2, 3), strict=True):
        if d.any():
            aa[0, s] = (Rotation.from_rotvec(d) * Rotation.from_rotvec(aa[0, s])).as_rotvec()
    args = {k: torch.from_numpy(pose[k]) for k in (*head.POSE_FIELDS[:7], 'betas')}
    with torch.no_grad():
        out = model(**args, body_pose=torch.from_numpy(aa), return_verts=True)
        translation = torch.from_numpy(pose['camera_translation'])[:, None]
        vertices = (out.vertices + translation).numpy()
        joints = (out.joints + translation).numpy()
    head.need(vertices[..., 2].min() > .05 and joints[..., 2].min() > .05, 'Non-positive replay depth.')
    project = lambda p: (p[..., :2] / p[..., 2:])[0].astype(np.float64)
    return aa, vertices.astype(np.float64), project(vertices @ K.T), joints, project(joints @ K.T)


def central(model, pose, K, directions, h):
    columns = []
    for direction in directions:
        plus, minus = (forward(model, pose, K, sign * h * direction) for sign in (1., -1.))
        columns.append([(plus[i] - minus[i]) / (2 * h) for i in (1, 2, 4)])
    return tuple(np.stack(block, axis=-1) for block in zip(*columns, strict=True))


def solve(objective, geom, pixel, other, dimensions, conservative):
    scale = (np.sqrt(3), np.sqrt(2)) if conservative else (1., 1.)
    a = [geom.reshape(-1, 6) * scale[0] / head.PROTECTED_METERS,
         pixel.reshape(-1, 6) * scale[1] / head.PROTECTED_PIXELS]
    if conservative:
        a.append(other.reshape(-1, 6) * np.sqrt(2) / OTHER_POINT_LIMIT)
    a = np.concatenate(a)
    a = np.concatenate((a, -a))
    limit = REGION_MARGIN if conservative else 1.
    boxes = [(-LIMIT_RAD, LIMIT_RAD) if i in dimensions else (0., 0.) for i in range(6)]
    sol = linprog(objective, A_ub=a, b_ub=np.full(len(a), limit), bounds=boxes, method='highs')
    head.need(sol.success and (a @ sol.x).max() <= limit + 1e-7, 'Local linear solve failed.')
    return sol.x, -float(sol.fun)


def measure(case, model, protected, screen):
    pose, row, target, K, report, reason = case
    ear = prior.EAR_SLOT
    base = forward(model, pose, K, np.zeros(6))
    replay_m = np.abs(base[1] - pose['vertices_camera']).max()
    replay_px = np.abs(base[4] - pose['joints_projected_native'][0]).max()
    head.need(replay_m < 2e-6 and replay_px < .001, 'Official pose did not replay.')
    residual = base[4][ear] - target[ear, :2]
    observed = np.flatnonzero(target[:, 2] > 0)
    body = np.intersect1d(observed, np.r_[screen.BODY, screen.FEET])
    other = np.array([i for i in body if i not in HEAD_OBSERVATIONS])
    th = .03 * (row['initial_box_xyxy'][3] - row['initial_box_xyxy'][1])
    errors = np.linalg.norm(base[4] - target[:, :2], axis=-1)
    head.need(body[errors[body] > th].tolist() == [ear], 'Stock active rejection is not solely the left ear.')

    big = central(model, pose, K, np.eye(6), DIFFERENCE)
    small = central(model, pose, K, np.eye(6), DIFFERENCE / 2)
    mixed = np.array([1., -1., 1., -1., 1., -1.]) / np.sqrt(6)
    mixed_measured = central(model, pose, K, mixed[None], DIFFERENCE)
    convergence = [float(np.max(np.abs(a - b)) * LIMIT_RAD) for a, b in zip(big, small, strict=True)]
    directional = [float(np.max(np.abs((a @ mixed) - b[..., 0])) * LIMIT_RAD)
                   for a, b in zip(small, mixed_measured, strict=True)]
    head.need(all(x < lim for values in (convergence, directional) for x, lim in zip(values, (.00025, .25, .25), strict=True)),
              'Stock Jacobian did not converge under the registered numerical check.')
    jac_m, jac_px, jac_joint = small
    geom, pixel, other_jac = jac_m[0, protected], jac_px[protected], jac_joint[other]
    radial = residual @ jac_joint[ear] / errors[ear]
    _, head_bound = solve(radial, geom, pixel, other_jac, range(3, 6), False)
    _, coupled_bound = solve(radial, geom, pixel, other_jac, range(6), False)
    step, prediction = solve(radial, geom, pixel, other_jac, range(6), True)
    exact = forward(model, pose, K, step)
    winner = dict(pose, body_pose_axis_angle=exact[0], vertices_camera=exact[1].astype(np.float32),
                  joints_camera=exact[3], joints_projected_native=exact[4][None].astype(np.float32))
    _, flags, _, _ = screen._candidate(row, winner, target, K, pose['betas'], None, report, True)
    m = float(np.linalg.norm((exact[1] - base[1])[0, protected], axis=-1).max())
    p = float(np.linalg.norm((exact[2] - base[2])[protected], axis=-1).max())
    other_move = float(np.linalg.norm((exact[4] - base[4])[other], axis=-1).max())
    protected_ok = m < head.PROTECTED_METERS and p < head.PROTECTED_PIXELS and other_move < OTHER_POINT_LIMIT
    return dict(frame=row['frame'], stock_ear_px=float(errors[ear]), original_gate_px=th,
                required_radial_gain_px=float(errors[ear] - th), replay_vertex_m=float(replay_m), replay_joint_px=float(replay_px),
                jacobian_step_convergence=convergence, jacobian_mixed_direction_error=directional,
                ear_jacobian_singular_values=np.linalg.svd(jac_joint[ear], compute_uv=False).tolist(),
                optimistic_head_only_radial_gain_px=head_bound, optimistic_neck_head_radial_gain_px=coupled_bound,
                conservative_prediction_px=prediction, tested_angles_deg=np.rad2deg(step).tolist(),
                exact_ear_px=float(np.linalg.norm(exact[4][ear] - target[ear, :2])), exact_protected_max_m=m,
                exact_protected_max_px=p, exact_other_body_joint_max_px=other_move, original_flags=reason['screen_flags'],
                exact_flags=flags, exact_pose_numeric_pass=not flags, exact_nonhead_protection_pass=protected_ok,
                exact_feasibility_witness=bool(not flags and protected_ok), fit_or_train_admitted=False)


def main():
    import torch

    sys.path.insert(0, str(head.PROJECT / 'src'))
    from shuolee_portrait import official_sequence as seq, stable_selection as screen

    head.need(head.sha(BASE) == 'f3dac18737d671306fd07ace2767bc02f740f4664d1fd380f172bc7c1656d2d0', 'Changed earlier audit.')
    run = head.RUN
    cfg = head.read(run / 'config.json')
    fit = head.read(run / 'fit/official-fit-result.json')
    sel = head.read(run / 'fit/selection.json')
    stock_receipt = head.read(prior.PREVIOUS)
    head.need(fit['selection'] == sel and head.sha(prior.PREVIOUS) == '2cf9a1ad565c0b6ac2d71ee09db07846f195d6142c35b552857a82f90d9bf39a',
              'Changed original comparison.')
    pins = dict(stock_receipt['source_pins'])
    for path in (BASE, prior.PREVIOUS, Path(__file__), Path(head.__file__), Path(prior.__file__), head.ASSET, FLAME, MAP):
        pins[str(path)] = head.sha(path)
    inputs = seq.load_inputs(run / 'data/inputs.json', run / 'data/observations.npz',
                            camera_length_contract=cfg['camera_length_contract'], reference_frame=sel['reference_frame'],
                            capture_layout=cfg['capture_layout'])
    head.need(inputs['bound'] == fit['source']['input_files'] and inputs['confidence'] == fit['source']['confidence']
              and set(prior.FRAMES).isdisjoint(sel['retained_holdouts'] + sel['selected_frames']), 'Inputs or data split differ.')
    pins.update(inputs['bound'])
    cases = []
    for frame in prior.FRAMES:
        i = inputs['ids'].index(frame)
        path = head.winner(frame)
        with np.load(path, allow_pickle=False) as archive:
            pose = dict(archive)
        row = inputs['rows'][i]
        target, K = inputs['targets'][i], inputs['K'][i]
        report = head.read(path.with_name('result.json'))
        reason = sel['reasons'][frame]
        scores, flags, _, _ = screen._candidate(row, pose, target, K, pose['betas'], None, report, True)
        head.need(scores == reason['scores'] and flags == reason['screen_flags'], 'Original screen differs.')
        pins[row['image']] = row['source_sha256']
        cases.append((pose, row, target, K, report, reason))
    faces = cases[0][0]['faces']
    labels, _ = head.semantics(faces)
    protected = ~np.isin(np.arange(10475), faces[labels != 3].ravel())
    head.need(protected.sum() == 5488, 'Changed protected mesh set.')
    head.verify(pins)
    OUT.mkdir()
    plan = dict(frames=prior.FRAMES, angular_axis_limit_deg=15., joints=[12, 15],
                epsilon_rad=[DIFFERENCE, DIFFERENCE / 2], margin=REGION_MARGIN,
                max_protected_vertex_m=head.PROTECTED_METERS, max_protected_vertex_px=head.PROTECTED_PIXELS,
                max_other_supported_body_joint_px=OTHER_POINT_LIMIT, protected_vertices=int(protected.sum()),
                exact_trials_per_frame=1, holdouts_used=False, optimizer_updates=0, pins=pins,
                scope='Local optimistic first-order component bounds and a conservative exact stock-forward witness; no fit output is written.')
    head.save(OUT / 'preregistration.json', plan)
    torch.set_num_threads(4)
    body, _, _ = seq.official.create_models(official_root=Path(cfg['official_smplifyx']['source_root']),
                                            assets=head.ASSET.parent.parent, device='cpu', static_topology=False)
    body.eval().requires_grad_(False)
    head.need(np.array_equal(body.faces, faces) and body.parents[12] == 9 and body.parents[15] == 12,
              'Changed stock SMPL-X topology.')
    rows = [measure(case, body, protected, screen) for case in cases]
    head.verify(pins)
    head.save(OUT / 'result.json', dict(state='neck_head_local_feasibility_measured', records=rows,
              preregistration_sha256=head.sha(OUT / 'preregistration.json'), fit_calls=0, gpu_calls=0,
              gaussian_updates=0, holdouts_used=False, quality_accepted=False))
    print(head.sha(OUT / 'result.json'))


if __name__ == '__main__':
    main()
