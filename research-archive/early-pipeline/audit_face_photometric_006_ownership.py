"""Attribute the cached 006 far spill to bound Gaussian owners without editing the model."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_face_photometric_006_holdouts as held


trial = held.trial
read, require, pin, sha = held.read, held.require, held.pin, held.sha
REVIEW = trial.ROOT/'protected-holdouts/result.json'
REVIEW_SHA = '0e4f6fecccec95d1c395b4c96ae53dd652b791a148764e9545f37802446c4262'
GROUPS = ('strict_protected_head', 'mixed_protected_head', 'arm_hand', 'leg_foot', 'trunk_collar', 'other')
SUPPRESS = ('any_protected_head', 'all_nonhead')


def ownership(faces, protected, dominant):
    face_head = protected[faces].sum(axis=1)
    group = np.full(len(faces), len(GROUPS)-1, np.int64)
    group[face_head == 3] = 0
    group[(face_head > 0) & (face_head < 3)] = 1
    limb_sets = (tuple(range(16,22)) + tuple(range(25,55)),
                 (1,2,4,5,7,8,10,11), (0,3,6,9,13,14))
    for index, joints in enumerate(limb_sets, 2):
        votes = np.isin(dominant[faces], joints).sum(axis=1)
        group[(face_head == 0) & (votes >= 2)] = index
    return group


def rois(mask):
    groups, _ = label(mask, np.ones((3,3), dtype=bool))
    indices = sorted(np.unique(groups[mask]).tolist(), key=lambda i: (-int((groups == i).sum()), i))
    result = [('all_added_far', mask)]
    for rank, index in enumerate(indices, 1):
        result.append((f'component_{rank}', groups == index))
    return result


def bounds(mask):
    y, x = np.where(mask)
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1]


def preflight():
    receipt, views = held.baseline()
    sys.path.insert(0, str(trial.PROJECT/'src'))
    state = held.candidate(receipt, views)
    require(state is not None, 'The registered candidate terminal is unavailable.')
    pin(REVIEW, REVIEW_SHA)
    review = read(REVIEW)
    require(review['state'] == 'protected_gate_complete' and review['frames'] == list(held.FRAMES)
            and review['candidate_terminal_sha256'] == receipt['candidate_terminal_sha256']
            and review['candidate_bound_sha256'] == receipt['candidate_bound_sha256']
            and review['baseline_native_alpha_sha256'] == receipt['baseline_native_alpha_sha256']
            and review['evaluation_fit_sha256'] == receipt['evaluation_fit_sha256']
            and review['optimizer_updates'] == review['refits'] == 0,
            'The cached protected replay does not bind the candidate and baseline.')
    candidate_alpha = REVIEW.with_name('candidate-native-alpha.npz')
    pin(candidate_alpha, review['candidate_alpha_sha256'])
    with np.load(candidate_alpha, allow_pickle=False) as cache:
        require(tuple(cache.files) == held.FRAMES, 'Cached protected raster roster differs.')
        for view, record in zip(state.views, review['records'], strict=True):
            frame = view.spec['frame']
            view.cached = cache[frame]
            require(frame == record['frame'] and record['camera'] == view.spec['camera']
                    and record['source_sha256'] == view.spec['source_sha256']
                    and record['scoring_mask_sha256'] == view.spec['score_mask_sha256']
                    and view.cached.dtype == np.float32 and view.cached.shape == view.visible.shape
                    and np.isfinite(view.cached).all(), 'Cached protected native raster changed.')
            added = (view.cached >= held.ALPHA) & ~view.visible & view.far
            require(int(added.sum()) == record['far_pixels']['added'] and added.any(), 'Cached added far pixels differ.')
            view.rois = rois(added)

    import torch
    from shuolee_portrait.bound_replay import inspect_bound_ply
    from shuolee_portrait.training import _frozen_smpl_surface

    torch.set_num_threads(4)
    base, linear, clothing, protected, error = _frozen_smpl_surface(state.poses, state.asset)
    require(base.device.type == linear.device.type == 'cpu', 'Frozen mesh parity must run on CPU.')
    faces = state.poses['faces']
    with np.load(state.asset, allow_pickle=False) as asset:
        weights = asset['weights']
    require(weights.shape == (len(protected),55) and np.isfinite(weights).all(), 'Unexpected SMPL-X skinning weights.')
    dominant = weights.argmax(axis=1)
    labels = ownership(faces, protected, dominant)
    binding, _ = inspect_bound_ply(state.snapshot/'bound-local.ply', len(faces))
    require(not np.any(state.mesh['static_offset'][~clothing]), 'Candidate offset crossed protected skin.')
    state.base, state.linear = base, linear
    state.binding, state.owners = binding, labels[binding]
    receipt.update(candidate_cached_alpha_sha256=review['candidate_alpha_sha256'],
                   candidate_protected_receipt_sha256=REVIEW_SHA, protected_vertices=int(protected.sum()),
                   clothing_vertices=int(clothing.sum()), mesh_replay_max_abs_m=error,
                   ownership={name: dict(triangles=int((labels == i).sum()),
                       bound_points=int((state.owners == i).sum()),
                       majority_clothing_triangles=int(((labels == i) & (clothing[faces].sum(axis=1) >= 2)).sum()))
                              for i,name in enumerate(GROUPS)},
                   components={view.spec['frame']: [
                       dict(region=name, pixels=int(mask.sum()), native_xyxy=bounds(mask))
                       for name,mask in view.rois] for view in state.views},
                   scope='Protected-vertex triangle partition and majority dominant skinning owner. '
                         'Alpha contribution is composited visibility, with no physical surface or cause inference.')
    return receipt, state


def render_frame(view, model, owners, draw):
    import torch

    white = torch.ones((len(owners),3), device='cuda', dtype=model.get_opacity.dtype)
    stock = draw(white)
    cached = torch.from_numpy(view.cached).to(stock.device)
    error = float((stock-cached).abs().max())
    require(error <= 1e-6 and torch.equal(stock >= held.ALPHA, cached >= held.ALPHA),
            'Stock replay does not reproduce the saved native Alpha and threshold decisions.')
    total = torch.zeros_like(stock)
    regions = {name: dict(pixels=int(mask.sum()), alpha_mass=float(view.cached[mask].sum(dtype=np.float64)),
                         contributions={}, suppression={}) for name,mask in view.rois}
    for index, name in enumerate(GROUPS):
        member = owners == index
        contribution = (draw(member[:,None].to(white.dtype).expand_as(white).contiguous())
                        if member.any() else torch.zeros_like(stock))
        total += contribution
        values = contribution.cpu().numpy()
        for region, mask in view.rois:
            alpha_mass = float(values[mask].sum(dtype=np.float64))
            regions[region]['contributions'][name] = dict(alpha_mass=alpha_mass,
                fraction_of_stock_mass=alpha_mass/regions[region]['alpha_mass'])
    closure = float((total-stock).abs().max())
    require(closure < 2e-5, 'Owner contributions do not reproduce stock compositing.')

    raw = model._opacity.detach().clone()
    for name, selected in zip(SUPPRESS, (owners < 2, owners >= 2), strict=True):
        try:
            model._opacity[selected] = -torch.inf
            require(not torch.count_nonzero(model.get_opacity[selected]), 'Opacity suppression is not exact.')
            suppressed = draw(white).cpu().numpy()
        finally:
            model._opacity.copy_(raw)
        for region, mask in view.rois:
            values = suppressed[mask]
            mass = float(values.sum(dtype=np.float64))
            regions[region]['suppression'][name] = dict(remaining_alpha_mass=mass,
                remaining_at_least_half=int((values >= held.ALPHA).sum()),
                alpha_mass_drop=regions[region]['alpha_mass']-mass)
    require(torch.equal(model._opacity, raw) and torch.equal(draw(white), stock),
            'Restoring raw opacity did not restore the exact stock render.')
    return dict(frame=view.spec['frame'], source_frame_index=view.spec['source_frame_index'],
                source_sha256=view.spec['source_sha256'], scoring_mask_sha256=view.spec['score_mask_sha256'],
                camera=view.spec['camera'], cached_alpha_max_abs=error, full_alpha_additivity_max_abs=closure,
                restored_opacity_and_stock_alpha_exact=True, regions=regions)


def replay(state):
    import torch
    from shuolee_portrait.bound_replay import RAW_PARAMETERS, load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose
    from types import SimpleNamespace

    with trial.common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock(), torch.no_grad():
            require(torch.cuda.device_count() == 1, 'Expose exactly one GPU.')
            state.rt.configure(state.asset.parents[1])
            from gaussian_renderer import render as raster

            model, _ = load_bound_snapshot(state.run, state.snapshot, asset_path=state.asset)
            require(np.array_equal(model.binding.cpu().numpy(), state.binding), 'Loaded triangle binding changed.')
            raw = {name: getattr(model,name).detach().clone() for name in RAW_PARAMETERS}
            offset = torch.from_numpy(state.mesh['static_offset']).cuda()[...,None]
            posed = torch.stack([v+(a@offset).squeeze(-1) for v,a in
                                zip(state.base.cuda(), state.linear.cuda(), strict=True)])
            model.vertices_sequence = torch.cat((model.vertices_sequence, posed))
            model.num_timesteps = len(model.vertices_sequence)
            owners = torch.from_numpy(state.owners).cuda()
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            black = torch.zeros(3, device='cuda')
            records = []
            for view in state.views:
                cam, actual = camera(view.row, view.camera_index, pose_index=view.pose_index)
                require(actual == view.spec['camera'], 'Candidate attribution camera changed.')
                def draw(colors):
                    return render_pose(raster, cam, model, pipe, black, view.pose_index, scaling_modifier=1.,
                                       override_color=colors)['render'][0]
                records.append(render_frame(view, model, owners, draw))
            require(model.optimizer is None and all(torch.equal(getattr(model,k), v) for k,v in raw.items())
                    and np.array_equal(model.binding.cpu().numpy(), state.binding), 'Replay changed the model.')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'render'))
    parser.add_argument('--output', type=Path, default=trial.ROOT/'protected-owner-attribution')
    args = parser.parse_args()
    if args.action == 'check':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    receipt, state = preflight()
    if args.action == 'check':
        print(json.dumps(dict(receipt, state='owner_attribution_inputs_verified', gpu_calls=0)))
        return
    output = args.output.resolve()
    require(output.parent == trial.ROOT and not output.exists(), 'Use a fresh separate trial output directory.')
    records = replay(state)
    output.mkdir(exist_ok=False)
    path = output/'result.json'
    held.spatial.save(path, dict(receipt, state='owner_attribution_complete', records=records,
                                model_changed=False, gpu_alpha_renders=int((4+len(np.unique(state.owners)))*len(records)),
                                reviewer_sha256=sha(Path(__file__))))
    print(json.dumps(dict(receipt=str(path), sha256=sha(path))))


if __name__ == '__main__':
    main()
