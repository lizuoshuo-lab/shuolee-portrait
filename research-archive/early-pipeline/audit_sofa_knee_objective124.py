"""Read-only saved-state diagnosis of the rejected temporal stage-entry fit."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from run_official_gaussians123 import read, save, sha, now, PROJECT, ASSETS
import official_smplifyx_entry123 as entry
from smplify_score_contract123 import relative_confidence
from smplify_observation_validity124 import mask_outside_raster
from smplify_temporal124 import rotation_energy

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/local-clothed-surface-123/cross-subject-sh0-124')
RUN = ROOT / 'sofa-temporal-warmstart-v2'
OLD = ROOT / 'sofa-temporal'
SRC = ROOT / 'sofa-input'
OUT = RUN / 'knee-objective-audit'
OFFICIAL = ROOT.parent / 'smplifyx-reference-control/official'


def frame(run, index, name='winner-camera-space.npz'):
    return run / f'official-fit/frames/frame_{index:06d}' / name


def prepare():
    assert not OUT.exists(), 'One registered audit only.'
    stop = read(RUN / 'quality-stop-decision.json')
    assert stop['state'] == 'stopped_quality_failed' and stop['pid'] is None
    inputs = [SRC / 'data/inputs.json', SRC / 'data/observations.npz', RUN / 'plan.json',
              RUN / 'quality-stop-decision.json', frame(RUN, 16), frame(RUN, 17),
              frame(OLD, 16), frame(RUN, 16, 'result.json')]
    sources = [Path(__file__), Path(entry.__file__), Path(__file__).with_name('smplify_temporal124.py'),
               Path(__file__).with_name('smplify_score_contract123.py'),
               Path(__file__).with_name('smplify_observation_validity124.py'),
               OFFICIAL / 'smplifyx/fitting.py', OFFICIAL / 'cfg_files/fit_smplx.yaml']
    OUT.mkdir()
    save(OUT / 'plan.json', dict(updated=now(),
         hypothesis='Evaluate the rejected16, old16, and actual17 initializer under the same saved current17 temporal target. Determine whether the final objective prefers the knee regression or a lower-energy saved state exists. This is not a new initialization or pose selection.',
         states=['rejected16', 'old16_complete_diagnostic_only', 'actual_seed17'],
         original_stages=[3, 4],
         budget=dict(cpu_only=True, states=3, stage_evaluations=6, max_seconds=120, optimizer_steps=0),
         protections=['same native observations', 'exact K and translation parity', 'same shared beta',
                      'saved geometry and winner loss replay parity', 'fixed current17 temporal target'],
         forbidden=['new fit', 'threshold or weight scan', 'old state as new initializer', 'replace poses', 'Gaussian training'],
         sources={str(p): sha(p) for p in sources}, inputs={str(p): sha(p) for p in inputs}))
    print('SOFA_KNEE_AUDIT_REGISTERED', flush=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def run():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not (OUT / 'result.json').exists()
    plan = read(OUT / 'plan.json')
    for path, digest in {**plan['sources'], **plan['inputs']}.items():
        assert sha(path) == digest, path
    import torch
    import smplx
    torch.set_num_threads(1)
    start = time.monotonic()
    rows = read(SRC / 'data/inputs.json')['records']
    row, neighbour_row = rows[15], rows[16]
    assert row['frame'] == 'frame_000016' and neighbour_row['frame'] == 'frame_000017'
    obs = load_npz(SRC / 'data/observations.npz')
    confidence, _ = relative_confidence(obs['scores'], obs['boxes'], obs['flip_disagreement'])
    target = np.zeros((1, 118, 3), np.float32)
    dst = np.r_[entry.COCO, 19:25, 25:67, 67:118]
    source = np.r_[0:23, 91:133, 40:91]
    target[0, dst, :2] = obs['keypoints'][15, source]
    target[0, dst, 2] = confidence[15, source]
    target, _ = mask_outside_raster(target, row['wh'])
    utils, prior, fitting, _ = entry.load_official(OFFICIAL)
    config = entry.load_official_config(OFFICIAL)
    mapper = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                   use_face_contour=False, openpose_format='coco25')
    body = smplx.SMPLX(str(ASSETS / 'smplx/SMPLX_NEUTRAL.npz'), batch_size=1, num_betas=10,
                      num_expression_coeffs=10, use_pca=True, num_pca_comps=12, flat_hand_mean=False,
                      use_face_contour=False, create_body_pose=False, create_transl=False,
                      joint_mapper=utils.JointMapper(mapper), dtype=torch.float32)
    loader = entry.module(PROJECT / 'src/avatar_core/shuolee_vposer.py', '_knee124_audit_vposer')
    loader.MODEL = ASSETS / 'vposer/V02_05'
    loader.CODE = PROJECT / 'third_party/human_body_prior'
    vposer, _, _ = loader.load_vposer('cpu')
    vposer.eval()
    current, old, previous = load_npz(frame(RUN, 16)), load_npz(frame(OLD, 16)), load_npz(frame(RUN, 17))
    receipt = read(frame(RUN, 16, 'result.json'))
    parity_fields = ['K_compute_float32', 'camera_translation', 'betas']
    assert all(np.array_equal(current[k], old[k]) for k in parity_fields)
    assert np.array_equal(current['betas'], previous['betas'])
    assert np.array_equal(receipt['continuation']['initial_body_latent'], previous['pose_embedding'])
    assert np.array_equal(receipt['continuation']['initial_state']['body_fit_initial_root'], previous['global_orient'])
    assert receipt['original_stage_indices'] == [3, 4]
    reference = torch.zeros(1, 165)
    reference[:, :3] = torch.from_numpy(previous['global_orient'])
    reference[:, 3:66] = torch.from_numpy(previous['body_pose_axis_angle'])
    dt = abs(row['time_seconds'] - neighbour_row['time_seconds'])
    nominal = read(RUN / 'plan.json')['nominal_dt_seconds']
    K, translation = torch.from_numpy(current['K_compute_float32']), torch.from_numpy(current['camera_translation'])
    gt, conf = torch.from_numpy(target[:, :, :2]), torch.from_numpy(target[:, :, 2])
    records, geometry_parity = [], {}

    def camera(points):
        q = (points + translation[:, None]) @ K.T
        return q[..., :2] / q[..., 2:]

    for state, saved in [('rejected16', current), ('old16_complete_diagnostic_only', old), ('actual_seed17', previous)]:
        params = {k: saved[k] for k, _ in body.named_parameters()}
        if state == 'actual_seed17':
            params = dict(betas=current['betas'], global_orient=previous['global_orient'])
        body.reset_params(**params)
        latent = torch.from_numpy(saved['pose_embedding'].copy()).requires_grad_()
        for stage in (3, 4):
            assert time.monotonic() - start < 120
            output = body(body_pose=vposer.decode(latent)['pose_body'].reshape(1, 63),
                          return_verts=True, return_full_pose=True)
            projected = camera(output.joints)
            if stage == 3 and state != 'actual_seed17':
                checks = dict(vertices_max=float((output.vertices + translation[:, None] - torch.from_numpy(saved['vertices_camera'])).abs().max()),
                              projection_max=float((projected - torch.from_numpy(saved['joints_projected_native'])).abs().max()))
                assert checks['vertices_max'] < 1e-5 and checks['projection_max'] < .02, checks
                geometry_parity[state] = checks
            jw = torch.ones(1, 118)
            jw[:, [1, 9, 12]] = 0
            jw[:, 25:67] = config['hand_joints_weights'][stage]
            jw[:, 67:] = config['face_joints_weights'][stage]
            weights = dict(data_weight=1000 / row['wh'][1], body_pose_weight=config['body_pose_prior_weights'][stage],
                           shape_weight=config['shape_weights'][stage], bending_prior_weight=3.17 * config['body_pose_prior_weights'][stage],
                           hand_prior_weight=config['hand_pose_prior_weights'][stage], expr_prior_weight=config['expr_weights'][stage],
                           jaw_prior_weight=[float(x) for x in config['jaw_pose_prior_weights'][stage].split(',')])
            l2 = prior.L2Prior
            loss = fitting.SMPLifyLoss(interpenetration=False, rho=config['rho'], use_hands=True, use_face=True,
                    body_pose_prior=l2(), shape_prior=l2(), angle_prior=prior.SMPLifyAnglePrior(),
                    left_hand_prior=l2(), right_hand_prior=l2(), expr_prior=l2(), jaw_prior=l2(), **weights)
            original = loss(output, camera, gt, conf, None, jw, use_vposer=True, pose_embedding=latent)
            temporal = rotation_energy(output.full_pose, reference, dt, nominal)
            data = (jw[..., None] * conf[..., None]).square() * loss.robustifier(gt - projected) * weights['data_weight']**2
            terms = dict(original=original, temporal=temporal, data=data.sum(), vposer=latent.square().sum()*weights['body_pose_weight']**2)
            gradients = {}
            for term, value in terms.items():
                gs = torch.autograd.grad(value, (latent, body.global_orient), retain_graph=True, allow_unused=True)
                gradients[term] = {k: float(g.norm()) if g is not None else 0. for k, g in zip(('latent_l2', 'root_l2'), gs)}
            residuals = (projected - gt).norm(dim=-1)
            records.append(dict(state=state, stage=stage, weights=weights,
                losses={k: float(v.detach()) for k, v in terms.items()}, total=float((original+temporal).detach()),
                gradient_norms=gradients,
                left_knee=dict(openpose_index=13, observed_xy=target[0,13,:2].tolist(), projected_xy=projected[0,13].detach().tolist(),
                               confidence=float(conf[0,13]), pixel_distance=float(residuals[0,13]), data_loss=float(data[0,13].sum())),
                data_groups={label: float(data[:,lo:hi].sum()) for label,lo,hi in [('body_feet',0,25),('hands',25,67),('face',67,118)]}))
    final = next(r for r in records if r['state'] == 'rejected16' and r['stage'] == 4)
    rel = abs(final['total'] - receipt['winning_loss']) / max(1., abs(receipt['winning_loss']))
    result = dict(updated=now(), elapsed_seconds=time.monotonic()-start, records=records,
                  geometry_parity=geometry_parity, exact_input_fields=parity_fields,
                  saved_winner_loss=receipt['winning_loss'], saved_loss_relative_difference=rel,
                  replay_final_loss_accepted=rel < 1e-4, source_hashes_verified=True,
                  optimization_steps=0, cpu_only=True, old_states_diagnostic_only=True,
                  causal_conclusion_reviewed=False, quality_accepted=False, gaussian_admitted=False)
    save(OUT / 'result.json', result)
    assert rel < 1e-4, rel
    print('SOFA_KNEE_AUDIT_COMPLETE', rel, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else run)()
