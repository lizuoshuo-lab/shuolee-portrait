"""Frozen all-frame hand correspondence and local-capacity audit; no fitting.

Use the original SMPL-X model twice at exactly the same geometry. The second
instance exposes its supported 45D hand axis-angle interface solely for a
Jacobian measurement. Never apply a solve/update to a saved fit.
"""
import copy
import fcntl
import json
import os
import time
import traceback

import numpy as np
import torch
from torch.func import functional_call

import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
from run_official_gaussians123 import read, save, sha, now

ROOT = base.mask.joint.ROOT
CONTROL = ROOT / 'smplify-vhap-normalized-laplacian'
OUT = ROOT / 'smplify-vhap-hand-capacity-audit'
PLAN = ROOT / 'smplify-vhap-hand-capacity-plan.json'


def statistics(values):
    x = np.asarray(values, dtype=float)
    return dict(median=float(np.median(x)), p90=float(np.quantile(x, .9)), maximum=float(x.max()))


def span_residual(jacobian, residual, weights):
    """Optimistic weighted-L2 tangent-space bound, NOT the L1 training loss."""
    a = np.asarray(jacobian, dtype=np.float64) * weights[:, None]
    b = np.asarray(residual, dtype=np.float64) * weights
    scales = np.linalg.norm(a, axis=0)
    active = scales > 1e-12
    a = a[:, active] / scales[active]
    if not a.size:
        return dict(rank=0, residual_energy_fraction=1., inferred_delta_norm=None)
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    keep = s > s[0] * 1e-5
    fitted = u[:, keep] @ (u[:, keep].T @ b)
    delta = (vh[keep].T @ ((u[:, keep].T @ b) / s[keep])) / scales[active]
    return dict(rank=int(keep.sum()), residual_energy_fraction=float(np.sum((b-fitted)**2) / max(np.sum(b*b), 1e-24)),
                inferred_delta_norm=float(np.linalg.norm(delta)),
                inferred_delta_max_abs=float(np.abs(delta).max()),
                retained_condition_number=float(s[0]/s[keep][-1]))


def main():
    plan = read(PLAN)
    assert plan['source_sha256'] == sha(__file__) and plan['optimizer_updates'] == 0
    assert not OUT.exists(), 'Do not repeat a finished audit or overwrite failure evidence.'
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir()
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tracker, rows = base.make_tracker(OUT)
    saved_path = CONTROL / 'epoch-30/tracked-smplx-params.npz'
    with np.load(saved_path, allow_pickle=False) as z, torch.no_grad():
        assert list(z['frame_ids']) == [r['frame'] for r in rows] == plan['roster']
        for key in bridge.PARAMS:
            getattr(tracker, key).copy_(torch.as_tensor(z[key], device='cuda'))
        saved_projection = z['joints_projected_native'].copy()
    tracker.vposer.eval()
    tracker.vposer.requires_grad_(False)
    for key in bridge.PARAMS:
        getattr(tracker, key).requires_grad_(key == 'pose_embedding')
    before = {k: bridge.digest_tensor(getattr(tracker, k)) for k in bridge.PARAMS}
    full_model = copy.deepcopy(tracker.smplx)
    full_model.use_pca = False  # Read-only alternate official input interface.
    full_model.requires_grad_(False)
    assert tracker.smplx.use_pca and tracker.left_hand_pose.shape[-1] == 12
    mapping = tracker.smplx.joint_mapper.joint_maps.detach().cpu().numpy()
    assert mapping[7] == mapping[25] == 20 and mapping[4] == mapping[46] == 21
    components = dict(left=tracker.smplx.left_hand_components.detach(), right=tracker.smplx.right_hand_components.detach())
    records = []
    for slot, row in enumerate(rows):
        assert time.monotonic()-start < plan['max_seconds']
        tracker.clear_cache()
        sample = tracker.dataset.one(slot)
        ids = sample['timestep_index']
        K = torch.tensor(row['K_integer_centers'], device='cuda', dtype=torch.float32)
        _, _, joints, _, _ = tracker.forward_smplx(ids)
        uv = joints @ K.T
        uv = uv[0, :, :2] / uv[0, :, 2:]
        parity = float(np.max(np.abs(uv.detach().cpu().numpy()-saved_projection[slot])))
        assert parity < .002, (row['frame'], parity)
        body = sample['lmk2d_body'][0].cuda()
        elbow_errors = {side: ((uv[index]+.5-body[index, :2])**2).sum()
                        for side, index in [('left', 6), ('right', 3)]}
        elbow_grad = {side: torch.autograd.grad(v, tracker.pose_embedding, retain_graph=True)[0][slot].detach()
                      for side, v in elbow_errors.items()}
        body_pose = tracker.vposer.decode(tracker.pose_embedding[ids])['pose_body'].reshape(1, 63).detach()
        parameters = {k: getattr(tracker, k)[ids].detach() for k in
                      ('expression', 'jaw_pose', 'global_orient', 'transl', 'leye_pose', 'reye_pose')}
        parameters.update(betas=tracker.betas[None].detach(), body_pose=body_pose,
                          left_hand_pose=(tracker.left_hand_pose[ids] @ components['left']).detach(),
                          right_hand_pose=(tracker.right_hand_pose[ids] @ components['right']).detach())
        template = (tracker.smplx.v_template + tracker.static_offset[0]).detach()
        item = dict(frame=row['frame'], saved_projection_parity_px=parity, hands={})
        for side, hand_start, wrist_index, wrist_pose_start in [('left', 25, 7, 57), ('right', 46, 4, 60)]:
            field = 'lmk2d_lhand' if side == 'left' else 'lmk2d_rhand'
            target = sample[field][0].cuda()
            predicted = uv[hand_start:hand_start+21]
            residual = (predicted + .5 - target[:, :2])
            scale = torch.tensor([2/row['wh'][0], 2/row['wh'][1]], device='cuda')
            per_point = ((residual*scale).abs().sum(-1)*target[:, 2]
                         * tracker.cfg.w.landmark * tracker.cfg.w.landmark_hand / 21)
            gradients = {}
            for label, value in [('root', per_point[0]), ('fingers', per_point[1:].sum()), ('all', per_point.sum())]:
                g = torch.autograd.grad(value, tracker.pose_embedding, retain_graph=True)[0][slot].detach()
                eg = elbow_grad[side]
                gradients[label] = dict(loss=float(value.detach()), body_latent_gradient_norm=float(g.norm()),
                    elbow_gradient_dot=float(g @ eg),
                    elbow_gradient_cosine=float((g @ eg) / (g.norm()*eg.norm()).clamp_min(1e-20)))
            # Each hand's 45 local finger axes plus its 3 wrist axes. Other poses,
            # shape, offset, cameras, target pixels and confidence remain fixed.
            initial = torch.cat([parameters[side+'_hand_pose'][0], body_pose[0, wrist_pose_start:wrist_pose_start+3]])
            def forward(q):
                p = dict(parameters)
                p[side+'_hand_pose'] = q[None, :45]
                p['body_pose'] = torch.cat([body_pose[:, :wrist_pose_start], q[None, 45:],
                                            body_pose[:, wrist_pose_start+3:]], dim=-1)
                output = functional_call(full_model, {'v_template': template}, (), p)
                projected = output.joints[0, hand_start:hand_start+21] @ K.T
                return (projected[:, :2]/projected[:, 2:]).flatten()
            initial = initial.detach().requires_grad_(True)
            alternate = forward(initial)
            representation_parity = float((alternate-predicted.detach().flatten()).abs().max())
            assert representation_parity < .002, (row['frame'], side, representation_parity)
            jac = torch.autograd.functional.jacobian(forward, initial, vectorize=True).detach()
            assert torch.isfinite(jac).all()
            # No step is applied. Central difference only verifies this Jacobian.
            direction = torch.sin(torch.arange(48, device='cuda', dtype=torch.float32)+1)
            direction = direction/direction.norm()
            epsilon = .002
            with torch.no_grad():
                numerical = (forward(initial+epsilon*direction)-forward(initial-epsilon*direction))/(2*epsilon)
            analytical = jac @ direction
            derivative_rel = float((numerical-analytical).norm()/analytical.norm().clamp_min(1e-8))
            assert derivative_rel < .03, (row['frame'], side, derivative_rel)
            j = jac.cpu().numpy().astype(float)
            c = components[side].cpu().numpy().astype(float)
            pca = j[:, :45] @ c.T
            weights = np.sqrt(target[:, 2].detach().cpu().numpy())[:, None] * np.array([2/row['wh'][0], 2/row['wh'][1]])
            r = residual.detach().cpu().numpy().flatten()
            translation = np.tile(np.eye(2), (21, 1))
            spans = dict(pca12=span_residual(pca, r, weights.flatten()),
                         full45=span_residual(j[:, :45], r, weights.flatten()),
                         pca12_wrist3=span_residual(np.c_[pca, j[:, 45:]], r, weights.flatten()),
                         full45_wrist3=span_residual(j, r, weights.flatten()),
                         common_image_translation=span_residual(translation, r, weights.flatten()))
            item['hands'][side] = dict(elbow_px=float(elbow_errors[side].detach().sqrt()),
                residual_px=statistics(residual.detach().norm(dim=-1).cpu().numpy()),
                body_wrist_to_hand_root_target_px=float((body[wrist_index, :2]-target[0, :2]).norm()),
                body_wrist_confidence=float(body[wrist_index, 2]), hand_root_confidence=float(target[0, 2]),
                model_wrist_root_parity_px=float((uv[wrist_index]-predicted[0]).detach().abs().max()),
                original_pca_to_full45_parity_px=representation_parity,
                jacobian_directional_check_relative_error=derivative_rel,
                local_span=spans, original_L1_body_gradient=gradients)
        records.append(item)
        save(OUT/'progress.json', dict(updated=now(),state='auditing_read_only',pid=os.getpid(),frames=len(records),
                                        optimizer_updates=0,seconds=time.monotonic()-start))
        del joints, uv, residual, per_point, alternate, jac, initial, elbow_errors, elbow_grad
    assert all(bridge.digest_tensor(getattr(tracker,k))==v for k,v in before.items())
    for p,h in plan['input_hashes'].items():
        assert sha(p)==h,p
    summary = {}
    for side in ('left', 'right'):
        hs = [r['hands'][side] for r in records]
        summary[side] = dict(target_wrist_conflict_px=statistics([h['body_wrist_to_hand_root_target_px'] for h in hs]),
            pca12_tangent_unexplained_fraction=statistics([h['local_span']['pca12']['residual_energy_fraction'] for h in hs]),
            full45_tangent_unexplained_fraction=statistics([h['local_span']['full45']['residual_energy_fraction'] for h in hs]),
            pca12_wrist3_tangent_unexplained_fraction=statistics([h['local_span']['pca12_wrist3']['residual_energy_fraction'] for h in hs]),
            full45_wrist3_tangent_unexplained_fraction=statistics([h['local_span']['full45_wrist3']['residual_energy_fraction'] for h in hs]),
            image_translation_explained_fraction=statistics([1-h['local_span']['common_image_translation']['residual_energy_fraction'] for h in hs]),
            root_opposes_elbow_frames=sum(h['original_L1_body_gradient']['root']['elbow_gradient_dot']<0 for h in hs),
            fingers_oppose_elbow_frames=sum(h['original_L1_body_gradient']['fingers']['elbow_gradient_dot']<0 for h in hs),
            all_hand_opposes_elbow_frames=sum(h['original_L1_body_gradient']['all']['elbow_gradient_dot']<0 for h in hs))
    result=dict(updated=now(),state='complete_read_only',frames=59,optimizer_updates=0,
        parameters_unchanged=True,source_sha256=sha(__file__),input_hashes=plan['input_hashes'],
        seconds=time.monotonic()-start,peak_cuda_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        summary=summary,records=records,quality_accepted=False,production_configuration_changed=False,
        limits=['Weighted L2 tangent projections are optimistic local geometry diagnostics, not actual L1 fits or independent ground truth.',
            'No inferred delta was applied; large proposed angle norms invalidate using the tangent bound as a prediction of convergence.',
            'Hidden hand observations remain unverified; a detector residual is not anatomical truth.',
            'A separate 45D input only tests the official model interface. No production hand dimension, prior, loss or optimizer changed.'])
    save(OUT/'result.json',result)
    save(OUT/'progress.json',dict(updated=now(),state='complete_read_only',pid=None,frames=59,optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','input_hashes')}),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
            save(OUT/'progress.json',dict(updated=now(),state='failed',pid=None,optimizer_updates=0))
        raise
