"""Exercise the replacement policy on the two pinned R8 terminal models."""

import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_smpl_delivery_round9 as launch


OUTPUT = launch.DELIVERY.parent.parent/'smpl-delivery-round9/endpoint-check'


def control(model, constraints, ref, options):
    import torch
    from gaussian_renderer import render
    from shuolee_portrait.render_support import _coverage_metrics, _rgb_error

    i = constraints.slots.index(ref)
    camera, view = constraints.cameras[i], constraints.views[i]
    model.select_mesh_by_timestep(ref)
    rgb = render(camera, model, constraints.pipe, constraints.black)['render']
    alpha = render(camera, model, constraints.pipe, constraints.black,
                   override_color=torch.ones_like(model._xyz))['render'][0]
    error = _rgb_error(rgb, alpha, view)
    regions = dict(constraints.protected_regions[i])
    x0, y0, _, _ = view.box
    if view.face is not None:
        regions['face'] = view.face+view.face.new_tensor([[y0], [x0]])
    if 'face' in regions:
        y, x = regions['face']
        valid = view.inside[y, x] & (alpha[y, x] >= .99)
        coords = torch.stack((y[valid], x[valid]), dim=1)
    else:
        coords = (view.inside & (alpha >= .99)).nonzero()
    launch.need(len(coords) > 0, 'No reliable opaque pixel in the endpoint control.')
    y, x = coords[len(coords)//2].tolist()
    target, low = constraints.options['foreground_target'], constraints.options['low_alpha']

    def metrics(a, e):
        return _coverage_metrics(alpha, a, error, e, view, regions, target, low, 2., 2., options)

    unchanged = metrics(alpha, error)
    redundant = alpha.clone()
    redundant[y, x] -= .005
    alpha_only = metrics(redundant, error)
    hole = alpha.clone()
    hole[y, x] = .5
    negative = metrics(hole, error)
    # Increase measured color error inside the protected face without modifying
    # opacity. This control supplies deliberate input error, not a training view.
    bad_color = error.clone()
    if 'face' in regions:
        ys, xs = regions['face']
        bad_color[ys-y0, xs-x0] += .05
    else:
        bad_color[view.inside[y0:y0+error.shape[0], x0:x0+error.shape[1]]] += .05
    color_only = metrics(alpha, bad_color)
    launch.need(not unchanged['failures'] and not alpha_only['failures']
                and alpha_only['foreground_target_deficit_px'] == 0
                and alpha_only['redundant_alpha_loss_px'] > 0
                and negative['new_low_alpha_holes'] == 1
                and any(item['item'] == 'new_low_alpha_holes' for item in negative['failures'])
                and any('rgb' in item['item'] for item in color_only['failures']),
                'Endpoint controls did not distinguish redundancy, holes and original RGB errors.')
    if 'face' in regions:
        launch.need(any(item.get('region') == 'face' for item in negative['failures'])
                    and any(item.get('region') == 'face' for item in color_only['failures']),
                    'Known endpoint face regressions were missed.')
    return dict(frame=constraints.rows[i]['frame'], native_pixel_xy=[x, y], unchanged=unchanged,
                redundant_alpha=alpha_only, induced_hole=negative, induced_rgb_error=color_only)


def audit(subject):
    import torch
    from shuolee_portrait.bound_replay import load_bound_snapshot
    from shuolee_portrait.sh0_main import Constraints
    from shuolee_portrait.stable_selection import source_bin_training_order
    from shuolee_portrait.training import (_native_foreground_target, camera, named_pose_layout,
                                           smpl_sh0_effective_config)
    from shuolee_portrait.world_scale import WorldScale

    plan, _, _ = launch.baseline(subject)
    run = Path(plan['baseline_run'])
    snapshot = run/'gaussian-run'/launch.round5.FINAL
    input_paths = [snapshot/name for name in ('bound-local.ply', 'bound-replay.json', 'world.ply', 'reference-mesh.npz')]
    frozen = {str(p):launch.sha(p) for p in input_paths}
    model, manifest = load_bound_snapshot(run, snapshot, asset_path=launch.common.ASSETS/'smplx/SMPLX_NEUTRAL.npz')
    rows = launch.read(run/'data/training-inputs.json')['records']
    old_cameras = launch.read(run/'gaussian-run/training-cameras.json')['records']
    with np.load(run/'fit/fitted.npz', allow_pickle=False) as archive:
        fit = dict(archive)
    layout = named_pose_layout(fit, rows)
    slots, ref = layout['pose_indices'], layout['reference_pose_index']
    cameras, records = zip(*(camera(row, i, pose_index=slot) for i, (row, slot) in enumerate(zip(rows, slots, strict=True))))
    launch.need(list(records) == old_cameras and len(rows) == 24
                and {row['frame'] for row in rows}.isdisjoint(plan['retained_holdouts'])
                and model.optimizer is None, 'Endpoint source cameras, roster or optimizer differ.')
    profile = smpl_sh0_effective_config()
    schedule = profile['unified_main']
    extent = float(np.linalg.norm(np.ptp(fit['vertices_world'][ref], axis=0)))
    bins, _ = source_bin_training_order(rows, launch.read(run/'fit/official-fit-result.json')['selection'], seed=profile['seed'])
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    black = torch.zeros(3, device='cuda')
    with torch.no_grad():
        constraints = Constraints(model, rows, slots, cameras, fit, pipe, black, None, bins, extent, schedule)
        controls = control(model, constraints, ref, schedule['world_scale']['render_support'])
        # The replay object owns private in-memory copies. Give stock topology
        # operations an empty optimizer, with no optimizer.step in this check.
        model.posed_vertices = model.vertices_sequence.__getitem__
        model.binding = model.binding.long()
        model.spatial_lr_scale = 1.
        model.max_radii2D = torch.zeros(len(model.binding), device='cuda')
        model.training_setup(SimpleNamespace(**profile['optimization']))
        scale = WorldScale(model, extent, ref, schedule['world_scale'])
        initial = len(model.binding)
        started = time.monotonic()
        scale.repair(0, constraints)
        event, limits = scale.splits[0], scale.limits.summary()
        for name, expected in frozen.items():
            launch.pin(Path(name), expected)
        repaired = event['split_parents']+event['needle']['support_replaced']
        launch.need(not model.optimizer.state and len(model.binding) == initial+repaired
                    and limits['inherited_children'] == 2*repaired and int(model.binding_counter.min()) >= 1,
                    'Endpoint topology operations changed the optimizer or lost bindings.')
        chosen = {
            name: (len(detail['selected_candidate_ids']) if detail else 0)
            for name, detail in event['render_support'].items()
            if name in ('coverage_trial', 'near_limit_trial')
        }
        launch.need(sum(chosen.values()) == repaired, 'Endpoint commit does not match validated subsets.')
    return dict(state='verified', subject=subject, optimizer_updates=0, frozen_on_disk_sha256=frozen,
                reference_frame=manifest['reference_frame_id'], points_before=initial, points_after=len(model.binding),
                protected_roi_view_counts={k:sum(k in r for r in constraints.protected_regions)
                    for k in ('left_eye', 'right_eye', 'left_ear', 'right_ear')},
                real_endpoint_controls=controls, replacement_event=event, limits=limits,
                fresh_training_initialization_used=False, probe_seconds=time.monotonic()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    launch.check_source(a.commit)
    launch.imports()
    import torch
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with launch.common.LEASE.open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = {}
        for subject in ('016', '006'):
            result = audit(subject)
            launch.common.write(OUTPUT/f'{subject}.json', result)
            results[subject] = {k:result[k] for k in ('state', 'optimizer_updates', 'points_before',
                                'points_after', 'limits', 'probe_seconds')}
            results[subject].update(file=str(OUTPUT/f'{subject}.json'), sha256=launch.sha(OUTPUT/f'{subject}.json'))
            print(json.dumps(dict(subject=subject, **results[subject])), flush=True)
            torch.cuda.empty_cache()
    launch.common.write(OUTPUT/'receipt.json', dict(state='completed', candidate_commit=a.commit,
        baseline_commit=launch.BASE, baseline_receipt_sha256=launch.DELIVERY_SHA, subjects=results,
        method='exact R8 terminal replay with observed native RGB/mask, three controlled perturbations and dry-run stock topology',
        optimizer_updates=0, previous_run_artifacts_changed=False, quality_accepted=False))


if __name__ == '__main__':
    main()
