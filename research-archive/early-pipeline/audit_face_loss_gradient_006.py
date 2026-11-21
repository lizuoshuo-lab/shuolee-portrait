"""Audit direct derivatives of the registered face term at its saved endpoint."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

import review_face_photometric_006 as endpoint
from review_bound_alpha_ab import pin, read, require, sha
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, save


OUT = LAB/'shuolee-smpl-face-photometric-006-gradient-support'


def inputs():
    pin(endpoint.PLAN, endpoint.PLAN_SHA)
    plan = read(endpoint.PLAN)
    info = endpoint.candidate_info(plan)
    require(info is not None, 'Wait for the registered trial terminal.')
    root = Path(info['run'])
    sys.path.insert(0, str(Path(info['project'])/'src'))
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import _inputs
    from shuolee_portrait.training import _face_interior_pixels, _native_foreground_target

    result = read(root/'result.json')
    replay_path = root/result['bound_replay']
    pin(replay_path, result['bound_replay_sha256'])
    snapshot = replay_path.parents[1]
    bound_path = snapshot/'bound-replay.json'
    pin(bound_path, read(replay_path)['bound_replay_sha256'])
    bound = read(bound_path)
    fit, mesh, facts = _inputs(root, snapshot, ASSET, bound)
    pin(snapshot/bound['bound_ply'], bound['bound_ply_sha256'])
    rows = read(root/'data/training-inputs.json')['records']
    known = read(root/'gaussian-run/config.json')['face_photometric_roi_pixels']
    with np.load(ASSET, allow_pickle=False) as asset:
        dominant = asset['weights'].argmax(axis=1)
        landmarks = np.concatenate((asset['lmk_faces_idx'].ravel(), asset['dynamic_lmk_faces_idx'].ravel()))
    protected = np.isin(dominant, [12,15,22,23,24])
    faces = fit['faces']
    protected[faces[landmarks].ravel()] = True
    head_vertices = protected[faces]
    face_groups = np.select([head_vertices.all(axis=1),head_vertices.any(axis=1)],[0,1],default=2)
    identities = []
    for slot,row in enumerate(rows):
        identity = facts['frames'][slot]
        require(identity['pose_index'] == slot and identity['frame'] == row['frame'] == fit['frame_ids'][slot],
                'Selected source and terminal poses differ.')
        pin(Path(row['image']),row['source_sha256'])
        pin(Path(row['mask']),row['mask_sha256'])
        _, alpha, box = _native_foreground_target(row)
        region = _face_interior_pixels(fit['observed_native_keypoints'][slot,67:118],alpha,box)
        require(region is not None and region.shape[1] == known[row['frame']], 'Face ROI differs from training.')
        identities.append(dict(frame=row['frame'], pose_index=slot, roi_pixels=region.shape[1]))
    preflight = dict(terminal_result_sha256=info['terminal_sha256'],
                    trial_terminal_sha256=info['trial_terminal_sha256'],
                    local_ply_sha256=bound['bound_ply_sha256'], script_sha256=sha(Path(__file__)),
                    groups=['all_three_protected','mixed_protected','none_protected'],
                    grouped_face_counts=np.bincount(face_groups,minlength=3).tolist(),
                    frames=identities, loss_weight=plan['change']['applied_weight'],
                    optimizer_updates=0, refits=0)
    return root, rt, snapshot, fit, mesh, facts, rows, face_groups, preflight


def support(gradient, groups):
    import torch
    if gradient is None or not gradient.numel():
        return None
    gradient = gradient.reshape(len(groups),-1).abs()
    require(torch.isfinite(gradient).all(), 'Nonfinite raster gradient.')
    mass = gradient.sum(dim=1)
    stats = []
    for i in range(3):
        x = mass[groups==i]
        stats.append(dict(rows=len(x), nonzero=int((x>0).sum()), abs_sum=float(x.sum()),
                          row_abs_max=float(x.max()) if len(x) else 0.))
    return stats


def render(data):
    _, rt, snapshot, fit, mesh, facts, rows, groups, plan = data
    from shuolee_portrait.bound_replay import RAW_PARAMETERS, _load_model
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose, _face_interior_pixels, _native_foreground_target
    import torch

    require(not OUT.exists(), 'Preserve an existing audit.')
    OUT.mkdir()
    save(OUT/'plan.json',plan)
    rt.configure(ASSET.parents[1])
    from gaussian_renderer import render as raster
    pipe=SimpleNamespace(convert_SHs_python=False,compute_cov3D_python=False,debug=False)
    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with gpu_lock():
            require(torch.cuda.device_count()==1, 'Expose one CUDA GPU.')
            model, parity = _load_model(snapshot/'bound-local.ply',snapshot,fit,mesh,facts,ASSET)
            parameters = [getattr(model,name) for name in RAW_PARAMETERS]
            before = [p.detach().clone() for p in parameters]
            for p in parameters:
                p.requires_grad_(True)
            group = torch.as_tensor(groups,device='cuda')[model.binding.long()]
            background = torch.zeros(3,device='cuda')
            records=[]
            for slot,row in enumerate(rows):
                source, mask, box = _native_foreground_target(row)
                yy,xx = torch.as_tensor(_face_interior_pixels(fit['observed_native_keypoints'][slot,67:118],
                                                              mask,box),device='cuda')
                cam, _ = camera(row,slot,pose_index=slot)
                color = render_pose(raster,cam,model,pipe,background,slot)
                white = raster(cam,model,pipe,background,override_color=torch.ones_like(model._xyz))
                rgb = color['render'][:,box[1]+yy,box[0]+xx]
                alpha = white['render'][:1,box[1]+yy,box[0]+xx]
                error = rgb - source.cuda()[:,yy,xx]
                loss = .5*plan['loss_weight']*(error.abs()+(error+1-alpha).abs()).mean()
                grad = torch.autograd.grad(loss, [*parameters,color['viewspace_points'],white['viewspace_points']],
                                           allow_unused=True)
                names=(*RAW_PARAMETERS,'rgb_viewspace','alpha_viewspace')
                records.append(dict(frame=row['frame'],weighted_face_loss=float(loss.detach()),
                                    gradients={name:support(value,group) for name,value in zip(names,grad,strict=True)}))
            require(model.optimizer is None and all(torch.equal(p.detach(),b) and p.grad is None
                                                    for p,b in zip(parameters,before,strict=True)),
                    'Gradient support read changed saved model state.')
    save(OUT/'result.json',dict(plan,state='completed',records=records,geometry_readback=parity,
                               scope='Direct gradients at the fixed terminal only; no historical trajectory or mesh-offset gradient.'))
    return dict(result=str(OUT/'result.json'),sha256=sha(OUT/'result.json'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    args=parser.parse_args()
    data=inputs()
    print(json.dumps(dict(data[-1],gpu_calls=0) if args.action=='check' else render(data)))
