"""Measure stock SH0 RGB-only versus joint RGB/alpha screen gradients with no updates."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch_unified_cold_016_006 import git_state
from review_unified_cold_views import ASSET, GPU_LEASE, LAB, pin, read, require, save, sha


PROJECT = Path('/mnt/d/Projects/Shuolee-smpl-unified-20261110')
COMMIT = '79b906fff9b62b6833041194fcc2ea3f64a1d464'
SCOPES = {
    '006': ('18333c3592caa278dfdd9cb8c11ee3ec2b561e94688178587824b31208c6faf7',
            '68e6af7e75b73b81def26a4647ff43721209235b8a7a63babb136b54a15bc867', (33, 12, 44)),
    '005': ('8663c18be59e5d6024ca83b0aaa3df5f153374ea4439e55b053e5dd972b4b2f9',
            '6860a91d48ca2e83c37729704a80e735bd7652ac85413f8cd3a32f60fc639f5d', (20, 18, 47)),
    '008': ('a06d2fb717aceea07a7d2af4adbd8267ac8521cf31c5ba5f3da11d329d4b6093',
            '85db35734fd9570edf39a1d6f631924b06eb7efc7230370505ee74182d764452', (58,)),
}
REAR_PANEL = LAB/'shuolee-smpl-unified-008-matched-views/late-new-native.png'
REAR_PANEL_SHA = 'ee62e9673362406d73e0d3153d983ca0a4e5e9b79f6f3633ececbe41b98be942'
RASTER = LAB/'shuolee-portrait-clean-001/env/lib/python3.10/site-packages/diff_gaussian_rasterization'
RASTER_PINS = {
    RASTER/'_C.cpython-310-x86_64-linux-gnu.so': '5161c09abd032639e582230874df6042a4bdd2b62f5137733386520c2a05e113',
    RASTER/'__init__.py': 'eafa9c3f670258b34535b7ee2832e6d8410d6a68762037216f04935ccdafba34',
    PROJECT/'src/avatar_core/submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu':
        '068eb4b0c20af25d22f274443e3fa6e24368418f691f099183e84109faccd6f7',
}
KEYS = ('frame', 'pose_index', 'K', 'w2c', 'wh', 'pixel_convention')
SCALES = (1., .5)


def screen_norms(rgb, alpha, area_weight):
    return np.linalg.norm(rgb[..., :2]*area_weight, axis=-1), np.linalg.norm((rgb+alpha)[..., :2]*area_weight, axis=-1)


def synthetic():
    import torch

    rgb = np.array([[3., 4., 0.], [3., 4., 5.], [3., 0., 10.], [0., 0., 0.]])
    alpha = np.array([[-3., -4., 100.], [3., 4., 0.], [0., 4., 0.], [0., 4., 0.]])
    stock, joint = screen_norms(rgb, alpha, .5)
    require(np.array_equal(stock, [2.5, 2.5, 1.5, 0.]) and np.array_equal(joint, [0., 5., 2.5, 2.]),
            'Joint gradient is not summed before its XY norm and area correction.')
    threshold = 2.
    require(np.array_equal(stock >= threshold, [True, True, False, False]) and
            np.array_equal(joint >= threshold, [False, True, True, True]), 'Synthetic threshold crossings differ.')
    separate_rgb = torch.ones((4, 3), dtype=torch.float64, requires_grad=True)
    separate_alpha = torch.ones_like(separate_rgb, requires_grad=True)
    color_vector, alpha_vector = (torch.from_numpy(v) for v in (rgb, alpha))
    diff = (separate_rgb*color_vector + separate_alpha*alpha_vector).sum()
    a, b = torch.autograd.grad(diff, (separate_rgb, separate_alpha))
    shared = torch.ones_like(separate_rgb, requires_grad=True)
    (correct,) = torch.autograd.grad((shared*color_vector + shared*alpha_vector).sum(), (shared,))
    require(torch.equal(a+b, correct), 'Synthetic shared-coordinate autograd sum differs.')


def rear_component(view, mesh, faces):
    pin(REAR_PANEL, REAR_PANEL_SHA)
    require(view.row['mask_sha256'] == '5aed7cd4a5fdcff95d0b8535ca317fabde494edc38f2fc9d3f45a7471029375d',
            'Unexpected 008 f058 training matte.')
    with Image.open(REAR_PANEL) as panel:
        score = np.asarray(panel.convert('L').crop((834, 21, 1668, 1199)))
        alpha = np.asarray(panel.convert('L').crop((0, 1220, 834, 2398)))
    require(np.array_equal(score, view.mask[342:1520, 940:1774]), 'Archived f058 score matte differs.')
    foreground = (score >= 253) & (alpha < 128) & ((np.arange(1178)[:, None]+342 >= 850) &
                                                  (np.arange(1178)[:, None]+342 <= 1250))
    regions, _ = label(foreground, structure=np.ones((3, 3), dtype=bool))
    index = regions[1209-342, 1305-940]
    require(index > 0, 'The registered f058 hole seed has no component.')
    crop = regions == index
    y, x = np.nonzero(crop)
    xy = np.stack((x+940, y+342), axis=1)
    require(len(xy) == 1363 and list(xy.min(axis=0)) == [1263, 1198] and list(xy.max(axis=0)+1) == [1371, 1222]
            and float(np.median(alpha[crop])) == 11., 'The source-anchored hole component differs.')
    region = np.zeros(view.mask.shape, dtype=bool)
    region[342:1520, 940:1774] = crop

    E, K = np.asarray(view.row['w2c']), np.asarray(view.row['K'])
    xyz = mesh @ E[:3, :3].T + E[:3, 3]
    projected = xyz @ K.T
    uv = projected[:, :2]/projected[:, 2:]-.5
    tri = uv[faces]
    inside_box = ((tri.min(axis=1) <= xy.max(axis=0)) & (tri.max(axis=1) >= xy.min(axis=0))).all(axis=1)
    ids = np.flatnonzero(inside_box & (xyz[faces, 2] > .2).all(axis=1))
    a, b, c = tri[ids].transpose(1, 0, 2)
    p = xy[:, None] - a
    u, v = b-a, c-a
    den = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        l1 = (p[..., 0]*v[:, 1]-p[..., 1]*v[:, 0])/den
        l2 = (u[:, 0]*p[..., 1]-u[:, 1]*p[..., 0])/den
        bary = np.stack((1-l1-l2, l1, l2), axis=-1)
        depths = np.where((bary >= -1e-7).all(axis=-1), 1/(bary/xyz[faces[ids], 2]).sum(axis=-1), np.inf)
    nearest = depths.argmin(axis=1)
    covered = np.isfinite(depths[np.arange(len(xy)), nearest])
    owners = np.unique(ids[nearest[covered]])
    return region, owners, dict(mesh_ray_hit_pixels=int(covered.sum()), roi_pixels=int(region.sum()),
                                legacy_median_alpha_8bit=11, exact_nearest_faces=owners.tolist(),
                                panel_sha256=REAR_PANEL_SHA, native_bbox_xyxy=[1263,1198,1371,1222])


def preflight(subject):
    synthetic()
    git_state(COMMIT)
    for path, digest in RASTER_PINS.items():
        pin(path, digest)
    sys.path.insert(0, str(PROJECT/'src'))
    from shuolee_portrait.bound_replay import _inputs, inspect_bound_ply
    from shuolee_portrait.training import camera

    result_sha, config_sha, frame_numbers = SCOPES[subject]
    run = LAB/f'shuolee-smpl-unified-{subject}/run'
    pin(run/'result.json', result_sha)
    pin(run/'gaussian-run/config.json', config_sha)
    final, cfg = read(run/'result.json'), read(run/'gaussian-run/config.json')
    require(final['state'] == 'completed' and final['steps'] == 10000 and final['format_passed'] is True and
            cfg['sh_degree'] == 0 and cfg['alpha_loss_weight'] == .1 and cfg['alpha_loss_normalization'] == 'foreground_area',
            'Unexpected source SH0 checkpoint/loss profile.')
    snapshot = run/'gaussian-run/step-010000'
    receipt_path = snapshot/'posed-replay/result.json'
    require(run/final['bound_replay'] == receipt_path, 'Use the original stock posed replay.')
    pin(receipt_path, final['bound_replay_sha256'])
    replay = read(receipt_path)
    manifest_path = snapshot/'bound-replay.json'
    pin(manifest_path, replay['bound_replay_sha256'])
    bound = read(manifest_path)
    fit, _, facts = _inputs(run, snapshot, ASSET, bound)
    ply = snapshot/'bound-local.ply'
    pin(ply, bound['bound_ply_sha256'])
    binding, _ = inspect_bound_ply(ply, len(fit['faces']))
    rows = read(run/'data/training-inputs.json')['records']
    held = read(run/'fit/selection.json')['retained_holdouts']
    camera_path = run/'gaussian-run/training-cameras.json'
    pin(camera_path, replay['training_camera_sha256'])
    cameras = read(camera_path)
    require(cameras['source_sha256'] == bound['inputs_sha256']['training_rows'] == replay['training_inputs_sha256']
            and held == replay['retained_holdouts'] and bound['frames'] == facts['frames'], 'Source row lineage differs.')

    faces = fit['faces']
    with np.load(ASSET, allow_pickle=False) as asset:
        require(np.array_equal(asset['f'], faces), 'Stock SMPL-X topology differs.')
        dominant = asset['weights'].argmax(axis=-1)
        protected = np.isin(dominant, [12,15,22,23,24])
        protected[faces[np.concatenate((asset['lmk_faces_idx'].ravel(), asset['dynamic_lmk_faces_idx'].ravel()))].ravel()] = True
    garment = np.isin(dominant, [0,1,2,3,4,5,6,9,13,14,16,17,18,19]) & ~protected
    groups = dict(strict_head=protected[faces[binding]].all(axis=1),
                  clothing_capable=garment[faces[binding]].all(axis=1),
                  arm=np.isin(dominant[faces[binding]], [16,17,18,19,20,21]).all(axis=1),
                  pelvis=(dominant[faces[binding]] == 0).all(axis=1))
    groups['other_or_mixed'] = ~(groups['strict_head'] | groups['clothing_capable'])
    views = []
    rear_evidence = None
    for number in frame_numbers:
        frame = f'frame_{number:06d}'
        require(frame not in held, 'A protected frame entered the training-only probe.')
        slot = [row['frame'] for row in rows].index(frame)
        row, actual = rows[slot], replay['records'][slot]
        named = cameras['records'][slot]
        _, regenerated = camera(row, slot, pose_index=slot)
        require(all(named[k] == actual['camera'][k] == regenerated[k] for k in KEYS)
                and actual['frame'] == bound['frames'][slot]['frame'] == frame and
                all(row[k] == actual[k] for k in ('source_sha256','source_frame_index','mask_sha256')),
                'Frozen source, pose or full native camera differs.')
        for path, expected in ((row['image'], row['source_sha256']), (row['mask'], row['mask_sha256'])):
            pin(Path(path), expected)
        require(actual['image'] == frame+'.png', 'Unexpected archived native raster filename.')
        archived = receipt_path.parent/actual['image']
        pin(archived, actual['image_sha256'])
        with Image.open(row['mask']) as matte:
            mask = np.asarray(matte.convert('L'))
        require(mask.shape == tuple(row['wh'][::-1]), 'Native target raster differs.')
        inside, outside = distance_transform_edt(mask >= 128), distance_transform_edt(mask < 128)
        regions = dict(inner_contour=(mask >= 253) & (inside > 2) & (inside <= 24),
                       outer_contour=(mask <= 2) & (outside > 2) & (outside <= 24))
        require(all(region.any() for region in regions.values()), 'A fixed native contour region is empty.')
        view = SimpleNamespace(row=row, slot=slot, camera=named, archived=archived, mask=mask, regions=regions)
        if subject == '008':
            mesh_path = receipt_path.parent/replay['mesh_file']
            pin(mesh_path, replay['mesh_sha256'])
            with np.load(mesh_path, allow_pickle=False) as posed:
                require(posed['frame_ids'][slot] == frame and np.array_equal(posed['faces'], faces), 'The rear posed mesh differs.')
                region, owners, rear_evidence = rear_component(view, posed['vertices_world'][slot], faces)
            view.regions['registered_hem_hole'] = region
            groups['f058_nearest_hem_faces'] = np.isin(binding, owners)
            rear_evidence['mesh_sha256'] = replay['mesh_sha256']
            rear_evidence['pelvis_owner_faces'] = int((dominant[faces[owners]] == 0).all(axis=1).sum())
        views.append(view)
    threshold = cfg['optimization']['densify_grad_threshold']
    require(threshold > 0, 'A positive stock densification threshold is required.')
    plan = dict(state='cpu_preflight_passed', subject=subject, commit=COMMIT, script_sha256=sha(Path(__file__)),
                terminal_sha256=result_sha, config_sha256=config_sha, replay_sha256=final['bound_replay_sha256'],
                bound_manifest_sha256=replay['bound_replay_sha256'], bound_ply_sha256=bound['bound_ply_sha256'],
                fitting_sha256=bound['inputs_sha256']['fitted'], training_camera_sha256=replay['training_camera_sha256'],
                raster_sha256={str(path):digest for path,digest in RASTER_PINS.items()},
                scales=SCALES, threshold=threshold, matched_views=[dict(frame=v.row['frame'], pose_index=v.slot,
                    source_frame_index=v.row['source_frame_index'], source_sha256=v.row['source_sha256'],
                    mask_sha256=v.row['mask_sha256'], camera=v.camera,
                    regions={k:dict(pixels=int(m.sum()), sha256=hashlib.sha256(m.tobytes()).hexdigest())
                             for k,m in v.regions.items()}) for v in views], rear_evidence=rear_evidence,
                anatomical_owner_counts={k:int(v.sum()) for k,v in groups.items()},
                scope='Same-state terminal proxy. XY vector sum then unchanged area weight and norm; .5 is sensitivity only.',
                regions='Target hard foreground/background 2–24 native px inside/outside; exact archived f058 component separate.',
                max_raster_forwards=len(views)*len(SCALES)*4, optimizer_updates=0, refits=0, export=False, quality_accepted=False)
    return run, snapshot, cfg, views, groups, plan


def statistics(rgb, alpha, weight, visible, groups, threshold):
    stock, combined = screen_norms(rgb, alpha, weight)
    added, removed = (stock < threshold) & (combined >= threshold), (stock >= threshold) & (combined < threshold)
    measurements = {}
    for name, selected in dict(all=visible, **{key:visible & mask for key,mask in groups.items()}).items():
        if not selected.any():
            continue
        measurements[name] = dict(visible=int(selected.sum()),
            rgb_norm_quantiles=np.quantile(stock[selected],[.50,.90,.99]).tolist(),
            joint_norm_quantiles=np.quantile(combined[selected],[.50,.90,.99]).tolist(),
            alpha_branch_norm_sum=float(np.linalg.norm(alpha[selected,:2]*weight,axis=-1).sum(dtype=np.float64)),
            rgb_threshold=int((selected & (stock >= threshold)).sum()), joint_threshold=int((selected & (combined >= threshold)).sum()),
            added=int((selected & added).sum()), removed=int((selected & removed).sum()))
    return measurements, stock, combined


def render(data):
    run, snapshot, cfg, views, groups, plan = data
    import torch
    from shuolee_portrait import runtime as rt
    from shuolee_portrait.bound_replay import RAW_PARAMETERS, load_bound_snapshot
    from shuolee_portrait.cli import gpu_lock
    from shuolee_portrait.training import camera, render_pose, _native_foreground_target, _sh0_foreground_alpha_loss

    rt.configure(ASSET.parents[1])
    from gaussian_renderer import render as raster
    import diff_gaussian_rasterization as kernel

    require(Path(kernel.__file__).resolve() == (RASTER/'__init__.py').resolve() and
            Path(kernel._C.__file__).resolve() in RASTER_PINS, 'Active native rasterizer differs from the pinned kernel.')
    output = LAB/f"shuolee-smpl-joint-screen-gradient-{plan['subject']}"
    require(not output.exists(), 'Preserve the existing screen-gradient audit.')
    records = []
    pooled = {scale:np.zeros((len(groups['strict_head']),3), dtype=np.float64) for scale in SCALES}
    with GPU_LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with gpu_lock():
            require(torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU.')
            output.mkdir()
            save(output/'plan.json', plan)
            model, bound = load_bound_snapshot(run, snapshot, asset_path=ASSET)
            parameters = [getattr(model, name).requires_grad_(True) for name in RAW_PARAMETERS]
            before = [p.detach().clone() for p in parameters]
            binding_before = model.binding.clone()
            geometry_before = model.vertices_sequence.clone()
            pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
            bg = torch.zeros(3, device='cuda')
            for view in views:
                pose = view.slot
                cam, _ = camera(view.row, pose, pose_index=pose)
                gt, target, box = _native_foreground_target(view.row)
                weight = (target > .99).count_nonzero().item()/target.numel()
                gt, target = gt.cuda(), target.cuda()
                for scale in SCALES:
                    repeats, region_values, regional_norms = [], {}, {}
                    for repeat in range(2):
                        color = render_pose(raster, cam, model, pipe, bg, pose, scaling_modifier=scale)
                        matte = raster(cam, model, pipe, bg, scaling_modifier=scale, override_color=torch.ones_like(model._xyz))
                        rgb, alpha = color['render'], matte['render'][:1]
                        a, b = color['viewspace_points'], matte['viewspace_points']
                        require(a is not b and a.data_ptr() != b.data_ptr() and a.shape == b.shape and
                                torch.equal(a, b) and torch.equal(color['radii'], matte['radii']) and
                                torch.equal(color['visibility_filter'], matte['visibility_filter']), 'RGB/alpha screens or visibility differ.')
                        visible = color['visibility_filter'].cpu().numpy()
                        photo, _, _, coverage = _sh0_foreground_alpha_loss(rgb, alpha, gt, target, box,
                            alpha_floor=cfg['straight_rgb_alpha_floor'])
                        loss = photo + cfg['alpha_loss_weight']*coverage
                        values = torch.autograd.grad(loss, [a,b,*parameters], retain_graph=not repeat, allow_unused=True)
                        require(values[0] is not None and values[1] is not None and all(v is None or torch.isfinite(v).all() for v in values),
                                'Missing or nonfinite gradient for the fixed render.')
                        grads = [None if value is None else value.detach().cpu().numpy() for value in values]
                        for item in grads:
                            if item is not None:
                                item.setflags(write=False)
                        repeats.append((grads, visible))
                        if repeat:
                            continue
                        if scale == 1.:
                            with Image.open(view.archived) as saved:
                                expected = np.asarray(saved.convert('RGB'))
                            actual = np.rint(rgb.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                            require(np.array_equal(actual, expected), 'Native stock RGB differs from archived replay.')
                        alpha_only, = torch.autograd.grad(cfg['alpha_loss_weight']*coverage, b, retain_graph=True)
                        for i,(name, region) in enumerate(view.regions.items()):
                            chosen = torch.as_tensor(region, device='cuda')
                            prediction, observation = alpha[0][chosen], target[0][chosen]
                            local_loss = .5*cfg['alpha_loss_weight']*(prediction-observation).abs().sum()/target.sum()
                            regional, = torch.autograd.grad(local_loss, b, retain_graph=i < len(view.regions)-1)
                            direct = regional.detach().cpu().numpy()
                            require(np.isfinite(direct).all(), 'Regional alpha-screen gradient is nonfinite.')
                            regional_norms[name] = np.linalg.norm(direct[:,:2]*weight, axis=-1)
                            support = {g:float(regional_norms[name][m & visible].sum(dtype=np.float64))
                                       for g,m in dict(all=visible, **groups).items()}
                            region_values[name] = dict(pixels=int(region.sum()), mean_alpha=float(prediction.mean()),
                                fraction_alpha_below_half=float((prediction < .5).float().mean()), alpha_only_screen_mass=support)
                        branch = alpha_only.detach().cpu().numpy()
                        alpha_terms = dict(mask_loss_screen_xy_norm=float(np.linalg.norm(branch[:,:2])),
                            straight_rgb_denominator_screen_xy_norm=float(np.linalg.norm((grads[1]-branch)[:,:2])))
                    (base, visible), (repeat, again) = repeats
                    require(np.array_equal(visible, again), 'Stock repeat changed the visible set.')
                    scale_large = model.get_scaling.detach().amax(dim=1).cpu().numpy() > (
                        cfg['optimization']['percent_dense']*cfg['reference_extent_m'])
                    pose_groups = dict(groups, clone_size=~scale_large, split_size=scale_large)
                    same_stats, stock, joint = statistics(base[0],base[1],weight,visible,pose_groups,plan['threshold'])
                    repeat_stats, stock_repeat, joint_repeat = statistics(
                        repeat[0],repeat[1],weight,visible,pose_groups,plan['threshold'])
                    pooled[scale][visible] += np.stack((stock[visible],joint[visible],np.ones(visible.sum())),axis=1)
                    rgb_on, joint_on = stock >= plan['threshold'], joint >= plan['threshold']
                    for name, norms in regional_norms.items():
                        region_values[name].update(added_owner_gradient_mass=float(norms[visible & ~rgb_on & joint_on].sum(dtype=np.float64)),
                            removed_owner_gradient_mass=float(norms[visible & rgb_on & ~joint_on].sum(dtype=np.float64)))
                    gradients = {}
                    for name, a, b in zip(('rgb_screen','alpha_screen',*RAW_PARAMETERS),base,repeat,strict=True):
                        require((a is None) == (b is None), f'Stock repeat changed gradient presence: {name}')
                        gradients[name] = None if a is None else float(np.abs(a-b).max(initial=0))
                    crossed = lambda x,y: int((visible & ((x >= plan['threshold']) != (y >= plan['threshold']))).sum())
                    records.append(dict(frame=view.row['frame'], pose=pose, scale=scale, raster_area_weight=weight,
                        forward_stock_rgb_exact=scale == 1., gradient_repeat_max_abs=gradients,
                        rgb_vs_joint_threshold_changes=crossed(stock,joint),
                        stock_vs_repeat_threshold_changes=crossed(stock,stock_repeat),
                        joint_vs_repeat_threshold_changes=crossed(joint,joint_repeat),
                        screen_stats=same_stats, repeat_screen_stats=repeat_stats, alpha_branch_terms=alpha_terms, regions=region_values))
            require(model.optimizer is None and torch.equal(model.binding,binding_before) and torch.equal(model.vertices_sequence,geometry_before)
                    and all(torch.equal(p.detach(),old) and p.grad is None for p,old in zip(parameters,before,strict=True)),
                    'This read-only gradient probe changed model parameters, posed geometry or optimizer state.')
    pool = {}
    for scale, acc in pooled.items():
        valid = acc[:,2] > 0
        avg = np.divide(acc[:,:2],acc[:,2:],out=np.zeros_like(acc[:,:2]),where=valid[:,None])
        pool[str(scale)] = {key:dict(rgb=int(((avg[:,0] >= plan['threshold']) & ids).sum()),
            joint=int(((avg[:,1] >= plan['threshold']) & ids).sum()), visible=int((valid & ids).sum()))
            for key,ids in dict(all=valid, **groups).items()}
    result = output/'result.json'
    save(result, dict(plan,state='completed',records=records,sampled_visibility_average=pool,
        bound_after_sha256=pin(snapshot/'bound-local.ply',bound['bound_ply_sha256']),
        parameter_values_and_mesh_bitwise=True, statistics_do_not_feed_optimizer=True,
        no_pruning=True, sampled_statistics_are_not_training_window=True))
    print(json.dumps(dict(result=str(result),sha256=sha(result),sampled_visibility_average=pool)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','render'))
    parser.add_argument('subject',choices=tuple(SCOPES))
    args = parser.parse_args()
    data = preflight(args.subject)
    if args.action == 'check':
        print(json.dumps(dict(data[-1], gpu_calls=0)))
    else:
        render(data)
