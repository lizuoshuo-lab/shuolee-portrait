"""One all59-frame, zero-update test of the original bending prior interface.

Use the frozen VPoser checkpoint already audited for RGB/mask/landmark forces.
No extra training, parameter sweep, new detections or source repository edits.
"""
import fcntl
import json
import os
import time
import traceback
import numpy as np
import torch
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
import vhap_smplify_angle_prior123 as adapter
from official_smplifyx_entry123 import load_official_config
from run_official_gaussians123 import read, save, sha, now

ROOT = base.mask.joint.ROOT
RUN = ROOT / 'smplify-vhap-normalized-laplacian'
OUT = ROOT / 'vhap-smplify-prior-audit-corrected'
PLAN = ROOT / 'vhap-smplify-prior-audit-corrected-plan.json'
OFFICIAL = ROOT.parent / 'smplifyx-reference-control/official'


def stats(values):
    x = np.asarray(values, dtype=float)
    return dict(minimum=float(x.min()), median=float(np.median(x)), maximum=float(x.max()))


def cosine(x, y):
    denominator = float(x.norm() * y.norm())
    return None if denominator < 1e-15 else float(torch.dot(x.reshape(-1), y.reshape(-1)) / denominator)


def synthetic_contract(source):
    prior = adapter.load_original_prior(source, torch.float64, 'cpu')
    p = torch.linspace(-.6, .6, 126, dtype=torch.float64).reshape(2, 63).requires_grad_()
    body = prior(p)
    full = prior(torch.cat([torch.ones((2, 3), dtype=p.dtype), p], -1), with_global_pose=True)
    assert torch.equal(body, full)
    ids = [52, 55, 9, 12]
    signs = torch.tensor([1, -1, -1, -1], dtype=p.dtype)
    expected = torch.exp(p[:, ids] * signs).square()
    assert torch.equal(body, expected)
    assert torch.autograd.gradcheck(lambda x: prior(x).sum(), (p,), eps=1e-6, atol=1e-6, rtol=1e-5)
    gradient, = torch.autograd.grad(body.sum(), p)
    support = torch.where(gradient.abs().sum(0) > 0)[0].tolist()
    assert support == sorted(ids)
    shoulder = p.detach().clone()
    shoulder[:, 45:51] += 1.2  # SMPL-X left/right shoulder, excluding root.
    assert torch.equal(prior(shoulder), body.detach())
    bent = torch.zeros((2, 63), dtype=p.dtype)
    bent[0, ids] = -signs
    bent[1, ids] = signs
    assert torch.all(prior(bent)[0] < prior(bent)[1])
    return dict(body_full_index_parity=True, analytic_formula_parity=True, finite_difference_passed=True,
                body_axis_angle_radians=True, nonzero_gradient_indices=support,
                shoulders_unconstrained_by_this_prior=True, signed_bending_control_passed=True)


def main():
    plan = read(PLAN)
    assert plan['source_sha256'] == sha(__file__) and plan['optimizer_updates'] == 0
    assert not OUT.exists(), 'Read the existing result; never repeat this audit.'
    for path, digest in plan['input_hashes'].items():
        assert sha(path) == digest, path
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32 = False
    source = OFFICIAL / 'smplifyx/prior.py'
    contract = synthetic_contract(source)
    save(OUT / 'synthetic-contract.json', dict(updated=now(), **contract))
    t, rows = base.make_tracker(OUT)
    checkpoint = RUN / 'epoch-30/tracked-smplx-params.npz'
    forces = read(RUN / 'joint-force-audit/result.json')
    assert forces['checkpoint_sha256'] == sha(checkpoint) and forces['frames'] == 59
    force_rows = {r['frame']: r for r in forces['records']}
    with np.load(checkpoint, allow_pickle=False) as z:
        saved = dict(z)
    assert list(saved['frame_ids']) == [r['frame'] for r in rows] == plan['roster']
    with torch.no_grad():
        for key in bridge.PARAMS:
            getattr(t, key).copy_(torch.as_tensor(saved[key], device='cuda'))
    for key in bridge.PARAMS:
        getattr(t, key).requires_grad_(key == 'pose_embedding')
    t.vposer.eval()
    t.vposer.requires_grad_(False)
    before = {key: bridge.digest_tensor(getattr(t, key)) for key in bridge.PARAMS}
    config = load_official_config(OFFICIAL)
    original_body_weight = config['body_pose_prior_weights'][-1]
    coefficient = adapter.vposer_ratio_coefficient(t.cfg.w.reg_vposer, 32, original_body_weight)
    prior = adapter.load_original_prior(source, torch.float32, 'cuda')
    records = []
    for i, row in enumerate(rows):
        assert time.monotonic() - start < plan['max_seconds']
        ids = torch.tensor([i], device='cuda')
        pose = adapter.decoded_body(t, ids)
        components = prior(pose)
        bend = adapter.bending_energy(t, ids, prior, 1.)
        grad, = torch.autograd.grad(bend, t.pose_embedding, retain_graph=True)
        grad = grad[i]
        assert torch.isfinite(grad).all() and grad.norm() > 0
        verts, _, joints, _, _ = t.forward_smplx(ids)
        K = torch.as_tensor(row['K_integer_centers'], device='cuda', dtype=joints.dtype)
        q = joints @ K.T
        uv = q[..., :2] / q[..., 2:]
        parity = float((uv - torch.as_tensor(saved['joints_projected_native'][i], device='cuda')).abs().max())
        vparity = float((verts[0] - torch.as_tensor(saved['vertices_camera'][i], device='cuda')).abs().max())
        assert parity < .002 and vparity < 5e-6
        sample = t.dataset.one(i)
        target = sample['lmk2d_body'].to(uv).clone()
        target[..., :2] -= .5  # NativeSamples stores corner-based pixels.
        raw = torch.as_tensor(t.dataset.obs['keypoints'][i, :17], device='cuda')
        assert float((target[0, t.dataset.COCO, :2] - raw).abs().max()) < 1e-4
        probes = {}
        for name, index in [('left_elbow', 6), ('right_elbow', 3), ('left_knee', 13), ('right_knee', 10)]:
            error2 = (uv[0, index] - target[0, index, :2]).square().sum()
            eg, = torch.autograd.grad(error2, t.pose_embedding, retain_graph=True)
            probes[name] = dict(error_px=float(error2.detach().sqrt()),
                               cosine_bending_vs_error_gradient=cosine(grad, eg[i]),
                               error2_derivative_under_negative_bending_gradient=float(-torch.dot(eg[i], grad)))
        previous = force_rows[row['frame']]['losses']
        norms = {name: float(previous[name]['gradient_l2']['pose_embedding']) for name in
                 ('photo', 'silhouette', 'lmk_body', 'lmk_face', 'smooth_joint', 'reg_joint')}
        converted_norm = float(grad.norm()) * coefficient
        record = dict(frame=row['frame'],projection_parity_px=parity,vertices_parity_m=vparity,
                      original_unweighted_bending_components=components[0].detach().cpu().tolist(),
                      original_unweighted_sum=float(bend.detach()), latent_gradient_l2=float(grad.norm()),
                      ratio_only_coefficient=coefficient, ratio_only_loss=float(bend.detach()) * coefficient,
                      ratio_only_latent_gradient_l2=converted_norm,
                      existing_same_checkpoint_latent_gradient_l2=norms,
                      ratio_only_vs_existing_gradient={name:converted_norm / value if value > 0 else None for name,value in norms.items()},
                      joint_error_direction=probes)
        records.append(record)
        save(OUT / 'progress.json', dict(updated=now(), state='auditing', pid=os.getpid(),
                                       frames=i+1, optimizer_updates=0, seconds=time.monotonic()-start))
    assert all(bridge.digest_tensor(getattr(t, key)) == digest for key, digest in before.items())
    for path, digest in plan['input_hashes'].items():
        assert sha(path) == digest, path
    summary = dict(bending_sum=stats([r['original_unweighted_sum'] for r in records]),
                   ratio_only_latent_gradient=stats([r['ratio_only_latent_gradient_l2'] for r in records]),
                   ratio_to_existing_gradients={name:dict(**stats([r['ratio_only_vs_existing_gradient'][name] for r in records if r['ratio_only_vs_existing_gradient'][name] is not None]),
                        undefined_zero_denominator_frames=[r['frame'] for r in records if r['ratio_only_vs_existing_gradient'][name] is None]) for name in norms},
                   bending_opposes_detector_error_descent={name:sum((r['joint_error_direction'][name]['cosine_bending_vs_error_gradient'] or 0) < 0 for r in records) for name in probes})
    result = dict(updated=now(), state='complete_read_only', frames=len(records), optimizer_updates=0,
                  parameters_unchanged=True, checkpoint_sha256=sha(checkpoint), source_sha256=sha(__file__),
                  measurement_correction='Detector probes explicitly subtract NativeSamples half-pixel offset and verify raw-observation parity. Earlier audit preserved.',
                  official_prior_sha256=sha(source), synthetic_contract=contract,
                  coefficient_contract=dict(vhap_vposer_weight=t.cfg.w.reg_vposer, latent_dimensions=32,
                    smplify_final_body_weight=original_body_weight, smplify_bending_weight=3.17*original_body_weight,
                    equivalent_bending_coefficient=coefficient,
                    purpose='Algebraic preservation of bending/VPoser ratio only; not calibrated against VHAP image terms.'),
                  records=records, summary=summary, seconds=time.monotonic()-start,
                  quality_accepted=False, gaussian_training_started=False,
                  limits=['No optimization or new mesh; this validates an optional original-prior interface.',
                    'The original prior penalizes four signed axis-angle components; it is neither joint limits nor collision handling.',
                    'A valid human pose may still be the wrong observed pose; detector observations are not independent anatomical truth.',
                    'Prior gradient norms use the same 32D latent and identical checkpoint; they are not Adam update magnitudes.',
                    'Do not copy raw SMPLify weights into VHAP or infer a training coefficient from the rejected output.'])
    save(OUT / 'result.json', result)
    save(OUT / 'progress.json', dict(updated=now(), state='complete_read_only', pid=None, frames=len(records), optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records', 'limits')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save(OUT / 'failure.json', dict(updated=now(), error=repr(error), traceback=traceback.format_exc()))
            save(OUT / 'progress.json', dict(updated=now(), state='failed', pid=None, optimizer_updates=0))
        raise
