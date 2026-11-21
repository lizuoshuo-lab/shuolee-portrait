"""One CPU local feasibility check for the admitted-guarded 008 rear transition."""

from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

import audit_008_head_joint as head


FRAMES = ('frame_000059', 'frame_000060')
EAR_VERTEX = 6
EAR_SLOT = 18
EPSILON = .001
MAX_COMPONENT_RAD = np.deg2rad(5.)
OUT = head.LAB / 'shuolee-smpl-head-local-bound-008'
PREVIOUS = head.LAB / 'shuolee-smpl-head-joint-008-f059-f060-cpu/result.json'


def perturb(pose, delta):
    axes = pose['body_pose_axis_angle'].copy()
    previous = Rotation.from_rotvec(axes[0, head.AXIS])
    axes[0, head.AXIS] = (Rotation.from_rotvec(delta) * previous).as_rotvec()
    return dict(pose, body_pose_axis_angle=axes)


def optimize(objective, vertex_jacobian, pixel_jacobian, safe):
    scale_m = np.sqrt(3) if safe else 1.
    scale_px = np.sqrt(2) if safe else 1.
    design = np.concatenate((vertex_jacobian.reshape(-1, 3) * scale_m / head.PROTECTED_METERS,
                             pixel_jacobian.reshape(-1, 3) * scale_px / head.PROTECTED_PIXELS))
    constraints = np.concatenate((design, -design))
    fit = linprog(objective, A_ub=constraints, b_ub=np.ones(len(constraints)),
                  bounds=[(-MAX_COMPONENT_RAD, MAX_COMPONENT_RAD)] * 3, method='highs')
    head.need(fit.success and np.max(constraints @ fit.x) <= 1.000001, 'Invalid local optimization receipt.')
    return fit.x, -float(fit.fun)


def measure(case, protected, shape):
    pose, row, observation, expected = case
    directions = np.eye(3) * EPSILON
    variants = [perturb(pose, sign * direction) for direction in directions for sign in (1., -1.)]
    vertex_sets = head.mesh_forward([pose, *variants], shape, pose['faces'])
    stock = vertex_sets[0]
    uv = [head.project(v, row) for v in vertex_sets]
    corner = np.asarray(row['K'], dtype=np.float64)
    integer = np.asarray(row['K_integer_centers'], dtype=np.float64)
    head.need(np.allclose(corner[:2, 2] - integer[:2, 2], .5, atol=1e-7, rtol=0), 'Changed pixel convention.')
    stock_error = np.max(np.abs(stock - pose['vertices_camera']))
    p = uv[0][EAR_VERTEX] - .5
    head.need(stock_error < 2e-6 and np.linalg.norm(p - pose['joints_projected_native'][0, EAR_SLOT]) < .002,
              'Stock vertex or ear replay differs.')
    residual = p - observation[:2]
    norm = np.linalg.norm(residual)
    head.need(abs(norm - expected['ear18_left_px'][0]) < .002, 'Stock ear residual differs.')
    pixel_j = np.stack([(uv[i] - uv[i + 1]) / (2 * EPSILON) for i in (1, 3, 5)], axis=-1)
    vertex_j = np.stack([(vertex_sets[i][0] - vertex_sets[i + 1][0]) / (2 * EPSILON)
                         for i in (1, 3, 5)], axis=-1)
    objective = residual @ pixel_j[EAR_VERTEX] / norm
    _, optimistic = optimize(objective, vertex_j[protected], pixel_j[protected], False)
    delta, conservative = optimize(objective, vertex_j[protected], pixel_j[protected], True)
    trial = head.mesh_forward([perturb(pose, delta)], shape, pose['faces'])[0]
    trial_uv = head.project(trial, row)
    moved_m = np.linalg.norm(trial[0, protected] - stock[0, protected], axis=1).max()
    moved_px = np.linalg.norm(trial_uv[protected] - uv[0][protected], axis=1).max()
    trial_error = np.linalg.norm(trial_uv[EAR_VERTEX] - .5 - observation[:2])
    return dict(frame=row['frame'], stock_ear_px=float(norm), stock_vertex_replay_m=float(stock_error),
                original_gate_px=expected['limit_px'], required_improvement_px=float(norm - expected['limit_px']),
                optimistic_first_order_gain_px=optimistic, conservative_first_order_gain_px=conservative,
                exact_trial_delta_rad=delta.tolist(), exact_trial_ear_px=float(trial_error),
                exact_trial_protected_max_m=float(moved_m), exact_trial_protected_max_px=float(moved_px),
                exact_trial_ear_pass=bool(trial_error < expected['limit_px']),
                exact_trial_protection_pass=bool(moved_m < head.PROTECTED_METERS and moved_px < head.PROTECTED_PIXELS))


def main():
    sys.path.insert(0, str(head.PROJECT / 'src'))
    from shuolee_portrait import official_sequence as seq

    run = head.RUN
    previous = head.read(PREVIOUS)
    head.need(head.sha(PREVIOUS) == '2cf9a1ad565c0b6ac2d71ee09db07846f195d6142c35b552857a82f90d9bf39a',
              'Changed opposite-side comparison.')
    selection = head.read(run / 'fit/selection.json')
    official = head.read(run / 'fit/official-fit-result.json')
    cfg = head.read(run / 'config.json')
    shared = np.array(head.read(run / 'fit/shared-body.json')['betas'], dtype=np.float32)
    inputs = seq.load_inputs(run / 'data/inputs.json', run / 'data/observations.npz',
                             camera_length_contract=cfg['camera_length_contract'],
                             reference_frame=selection['reference_frame'], capture_layout=cfg['capture_layout'])
    head.need(official['source']['input_files'] == inputs['bound'] and official['source']['confidence'] == inputs['confidence']
              and set(FRAMES).isdisjoint(selection['retained_holdouts']), 'Stock inputs or holdout separation changed.')
    cases, pins = [], {str(PREVIOUS): head.sha(PREVIOUS), str(Path(__file__)): head.sha(Path(__file__)),
                       str(Path(head.__file__)): head.sha(Path(head.__file__)), str(head.ASSET): head.sha(head.ASSET)}
    for frame, old in zip(FRAMES, previous['records'], strict=True):
        path = head.winner(frame)
        with np.load(path, allow_pickle=False) as blob:
            pose = dict(blob)
        i = inputs['ids'].index(frame)
        row, observation = inputs['rows'][i], inputs['targets'][i, EAR_SLOT]
        head.need(old['frame'] == frame and old['source_sha256'] == row['source_sha256']
                  and old['rejected_body_or_feet_ids'][0] == [EAR_SLOT] and observation[2] > 0
                  and np.array_equal(shared, pose['betas']) and np.array_equal(row['w2c'], np.eye(4))
                  and np.array_equal(inputs['K'][i], pose['K_integer_centers']),
                  'Changed source, camera or sole ear rejection.')
        cases.append((pose, row, observation, old))
        pins[str(path)] = head.sha(path)
        pins[str(row['image'])] = row['source_sha256']
    faces = cases[0][0]['faces']
    labels, _ = head.semantics(faces)
    protected = ~np.isin(np.arange(10475), faces[labels != 3].ravel())
    head.need(protected.sum() == 5488, 'Changed protected geometry.')
    head.verify(pins)
    OUT.mkdir()
    plan = dict(frames=FRAMES, protected_vertices=int(protected.sum()), axis_angle_component_limit_rad=MAX_COMPONENT_RAD,
                finite_difference_rad=EPSILON, optimizer_updates=0, fit_calls=0, gpu_calls=0, pins=pins,
                scope='One first-order three-axis head-joint bound; relaxed componentwise protection is optimistic. '
                      'A separate conservative linear witness has one exact stock forward per frame. No global certificate.')
    head.save(OUT / 'preregistration.json', plan)
    rows = [measure(case, protected, shared) for case in cases]
    head.verify(pins)
    head.save(OUT / 'result.json', dict(state='local_head_bound_measured', records=rows,
        preregistration_sha256=head.sha(OUT / 'preregistration.json'), holdouts_used=False,
        fit_calls=0, gaussian_updates=0, gpu_calls=0, quality_accepted=False))
    print(head.sha(OUT / 'result.json'))


if __name__ == '__main__':
    main()
