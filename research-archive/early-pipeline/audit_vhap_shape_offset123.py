"""Frozen-checkpoint component ablation; no fitting or candidate parameters saved."""
import fcntl
import json
import os
import time
import traceback
import numpy as np
import torch
from smplx.lbs import blend_shapes
import fit_smplify_vhap_normalized_laplacian123 as base
import fit_smplify_vhap_surface123 as bridge
from run_official_gaussians123 import read, save, sha, now

ROOT = base.mask.joint.ROOT
RUN = ROOT / 'smplify-vhap-normalized-laplacian'
OUT = ROOT / 'smplify-vhap-shape-offset-audit'
PLAN = ROOT / 'smplify-vhap-shape-offset-plan.json'
LABELS = ('initial_shape_no_offset', 'current_shape_no_offset',
          'initial_shape_current_offset', 'current_shape_current_offset')


def stats(x):
    x = np.asarray(x, dtype=float)
    return dict(minimum=float(x.min()), median=float(np.median(x)),
                p90=float(np.quantile(x, .9)), maximum=float(x.max()))


def main():
    plan = read(PLAN)
    assert plan['source_sha256'] == sha(__file__)
    assert plan['optimizer_updates'] == 0 and not OUT.exists()
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    lock = open('/tmp/shuolee-native122-gpu.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OUT.mkdir()
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(6060)
    torch.backends.cuda.matmul.allow_tf32 = False
    t, rows = base.make_tracker(OUT)
    checkpoint = RUN / 'epoch-30/tracked-smplx-params.npz'
    with np.load(checkpoint, allow_pickle=False) as z:
        saved = {k: z[k].copy() for k in z.files}
    with np.load(RUN / 'initial/tracked-smplx-params.npz', allow_pickle=False) as z:
        beta0 = torch.as_tensor(z['betas'].copy(), device='cuda')
    assert list(saved['frame_ids']) == [r['frame'] for r in rows] == plan['roster']
    with torch.no_grad():
        for k in bridge.PARAMS:
            getattr(t, k).copy_(torch.as_tensor(saved[k], device='cuda'))
    before = {k: bridge.digest_tensor(getattr(t, k)) for k in bridge.PARAMS}
    beta = t.betas.detach().clone()
    offset = t.static_offset.detach().clone()
    beta_delta = blend_shapes((beta - beta0)[None], t.smplx.shapedirs[..., :10])
    reallocated = offset + beta_delta
    variants = ((beta0, torch.zeros_like(offset)), (beta, torch.zeros_like(offset)),
                (beta0, offset), (beta, offset))
    meshes = {label: [] for label in LABELS}
    records = []
    parity = []
    try:
        with torch.no_grad():
            for i, row in enumerate(rows):
                assert time.monotonic() - started < plan['max_seconds']
                assert sha(row['image']) == row['source_sha256']
                assert sha(row['mask']) == row['mask_sha256']
                sample = t.dataset.one(i)
                mask = sample['alpha_map'][0, 0].numpy() > .5
                per_variant = {}
                full_v = full_j = None
                K = torch.tensor(row['K_integer_centers'], device='cuda', dtype=offset.dtype)
                def project(j):
                    q = j @ K.T
                    return q[..., :2] / q[..., 2:]
                for label, (b, off) in zip(LABELS, variants):
                    t.betas.copy_(b)
                    t.static_offset.copy_(off)
                    t.clear_cache()
                    v, _, joints, _, _ = t.forward_smplx(sample['timestep_index'])
                    rast = t.rasterize_smplx(sample, v, t.smplx.faces_tensor, train_mode=False)
                    visible = (rast['rast_out'][0, ..., -1].flip(0) > 0).cpu().numpy()
                    per_variant[label] = dict(
                        mask_iou=float((visible & mask).sum() / max(1, (visible | mask).sum())),
                        target_uncovered=float((mask & ~visible).sum() / max(1, mask.sum())),
                        mesh_outside=float((visible & ~mask).sum() / max(1, visible.sum())))
                    meshes[label].append(v[0].cpu().numpy())
                    if label == LABELS[-1]:
                        full_v, full_j = v.detach().clone(), joints.detach().clone()
                # Exact shape/offset gauge: preserve v_template + S*beta + offset.
                # This is a temporary evaluation only; no candidate checkpoint or optimizer.
                t.betas.copy_(beta0)
                t.static_offset.copy_(reallocated)
                t.clear_cache()
                v, _, joints, _, _ = t.forward_smplx(sample['timestep_index'])
                ep = dict(frame=row['frame'], vertices_m=float((v-full_v).abs().max()),
                          joints_m=float((joints-full_j).abs().max()),
                          projection_px=float((project(joints)-project(full_j)).abs().max()),
                          checkpoint_vertex_m=float(np.max(np.abs(full_v[0].cpu().numpy()-saved['vertices_camera'][i]))))
                assert ep['vertices_m'] < 5e-6 and ep['joints_m'] < 5e-6
                assert ep['projection_px'] < .005 and ep['checkpoint_vertex_m'] < 5e-6
                parity.append(ep)
                records.append(dict(frame=row['frame'], variants=per_variant))
                save(OUT/'progress.json', dict(updated=now(),state='auditing',pid=os.getpid(),
                     frames=i+1,optimizer_updates=0,seconds=time.monotonic()-started))
                del rast, v, joints, full_v, full_j
    finally:
        with torch.no_grad():
            t.betas.copy_(beta)
            t.static_offset.copy_(offset)
        t.clear_cache()
    assert all(bridge.digest_tensor(getattr(t, k)) == h for k, h in before.items())
    np.savez_compressed(OUT/'diagnostic-components.npz',
                        **{k:np.asarray(v) for k,v in meshes.items()},
                        frame_ids=saved['frame_ids'],faces=saved['faces'],K_integer_centers=saved['K_integer_centers'])
    def penalties(b, off):
        lap = torch.sparse.mm(t.smplx_laplacian, off[0])
        return dict(shape=float(t.cfg.w.reg_shape*b.square().mean()),
                    offset=float(t.cfg.w.reg_offset*off.abs().mean()),
                    laplacian=float(t.cfg.w.reg_offset_lap*lap.square().sum(-1).mean()))
    summary = {label:{metric:stats([r['variants'][label][metric] for r in records])
                      for metric in ('mask_iou','target_uncovered','mesh_outside')} for label in LABELS}
    offset_effect = {}
    for metric in ('mask_iou','target_uncovered','mesh_outside'):
        delta=np.array([r['variants'][LABELS[-1]][metric]-r['variants'][LABELS[1]][metric] for r in records])
        offset_effect[metric]=dict(delta=stats(delta),increased=int((delta>0).sum()),decreased=int((delta<0).sum()))
    result = dict(updated=now(),state='complete_read_only_pending_visual_review',frames=59,optimizer_updates=0,
                  parameters_restored=True,checkpoint_sha256=sha(checkpoint),seconds=time.monotonic()-started,
                  pose_and_all_other_variables_fixed=True,records=records,summary=summary,offset_effect_at_current_shape=offset_effect,
                  gauge_parity=parity,gauge_max_projection_px=max(r['projection_px'] for r in parity),
                  original_offset_mm=stats(np.linalg.norm(offset.cpu().numpy()[0],axis=-1)*1000),
                  equivalent_reallocated_offset_mm=stats(np.linalg.norm(reallocated.cpu().numpy()[0],axis=-1)*1000),
                  original_penalties=penalties(beta,offset),equivalent_reallocated_penalties=penalties(beta0,reallocated),
                  source_hashes=plan['input_hashes'],quality_accepted=False,actual_visual_review=False,
                  scope='Full59 frozen-pose counterfactual render. Not a refit, candidate, or improved reconstruction.',
                  limits=['Shape and free offset share an exact linear subspace. Their magnitudes alone do not measure clothing fit.',
                          'Conditional ablations do not decompose nonlinear pose effects or uniquely identify the original fitting failure.',
                          'Mask overlap and keypoint residuals do not establish hidden-surface physical truth.',
                          'No new Gaussian, camera, weight download, input manipulation, or production default change.'])
    for p,h in plan['input_hashes'].items():
        assert sha(p)==h,p
    save(OUT/'result.json',result)
    save(OUT/'progress.json',dict(updated=now(),state='complete_read_only',pid=None,frames=59,optimizer_updates=0))
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','gauge_parity','source_hashes')},ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists():
            save(OUT/'failure.json',dict(updated=now(),error=repr(error),traceback=traceback.format_exc()))
        raise
