"""Optimize a fixed Gaussian head against native face observations after main training."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from . import runtime as rt
from .face_supplement import REGIONS
from .render_support import _coverage_metrics, _rgb_error, _tile_sum
from .sh0_main import DCColorProjection
from .size_child_optimization import _preserve_pose, _view_cycle


CONFIG = dict(max_cycles=6, min_cycles=4, stable_cycles=2, convergence_delta=1e-5,
              conservative_lr_factor=.25, minimum_gain=.0001,
              color_lr=.004, opacity_lr=.010, cycle_lr_decay=.75,
              color_trust=.060, opacity_trust=.160, original_weight=.60,
              near_hole_alpha=.95, dark_luma=.20, min_region_pixels=8,
              min_landmark_confidence=.50, soft_alpha_weight=.50, epsilon=1e-6)
GROUPS = {'f_dc': '_features_dc', 'opacity': '_opacity'}
MOMENTS = ('exp_avg', 'exp_avg_sq')
FIXED_MODEL = ('_xyz', '_scaling', '_rotation', '_features_rest', 'binding', 'binding_counter',
               'densification_ids', 'max_radii2D', 'xyz_gradient_accum', 'denom', 'static_offset')


@dataclass
class _FaceView:
    frame: str
    role: str
    source_bin: int
    slot: int
    camera: object
    view: object
    target_alpha: torch.Tensor
    face: torch.Tensor
    regions: dict
    eligibility: dict
    baseline_alpha: torch.Tensor | None = None
    baseline_error: torch.Tensor | None = None

    @property
    def crop(self):
        x0, y0, x1, y1 = self.view.box
        return slice(y0, y1), slice(x0, x1)


def _digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _landmark_region(view, observed, group):
    indices = REGIONS[group]
    points = observed[indices]
    usable = np.isfinite(points).all(1) & (points[:, 2] >= CONFIG['min_landmark_confidence'])
    points = points[usable, :2]
    if len(points) < (1 if group.endswith('_ear') else 3):
        return None
    x0, y0, x1, y1 = view.box
    if group.endswith('_ear'):
        inner = observed[67:118]
        inner = inner[np.isfinite(inner).all(1) & (inner[:, 2] >= CONFIG['min_landmark_confidence']), :2]
        if len(inner) < 8:
            return None
        pad = max(3, round(.12*float(np.ptp(inner, axis=0).max())))
    else:
        pad = max(2, round(.25*float(np.ptp(points, axis=0).max())))
    left, top = np.maximum(np.floor(points.min(0)).astype(int)-pad, [x0, y0])
    right, bottom = np.minimum(np.ceil(points.max(0)).astype(int)+pad+1, [x1, y1])
    if left >= right or top >= bottom:
        return None
    return view.inside[top:bottom, left:right].nonzero().T + view.inside.new_tensor(
        [[top], [left]], dtype=torch.long)


def _float_rgb(view):
    return view.rgb.float().div(255) if view.rgb.dtype == torch.uint8 else view.rgb


def _views(constraints, supplemental, observed):
    views = []
    for i, (view, camera, slot, row, source_bin) in enumerate(zip(
            constraints.views, constraints.cameras, constraints.slots, constraints.rows,
            constraints.bins, strict=True)):
        points = observed[slot]
        x0, y0, x1, y1 = view.box
        y, x = slice(y0, y1), slice(x0, x1)
        origin = view.inside.new_tensor([[y0], [x0]], dtype=torch.long)
        regions = dict(getattr(constraints, 'protected_regions', [{} for _ in constraints.views])[i])
        for group in REGIONS:
            if group not in regions:
                region = _landmark_region(view, points, group)
                if region is not None:
                    regions[group] = region
        inside = view.inside[y, x]
        face = torch.zeros_like(inside)
        if view.face is not None:
            fy, fx = view.face
            face[fy, fx] = inside[fy, fx]
        if not bool(face.any()):
            # Older detectors can provide usable landmark neighborhoods with
            # no complete face interior. Preserve only those observed pixels.
            for region in regions.values():
                ry, rx = region-origin
                face[ry, rx] = inside[ry, rx]
        luma = (_float_rgb(view)*view.inside.new_tensor([.2126, .7152, .0722], dtype=torch.float32)[:, None, None]).sum(0)
        head = face.clone()
        box = np.asarray(row.get('face_box_xyxy', []))
        if box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]):
            span = max(box[2:]-box[:2])
            left, top = np.maximum(np.floor(box[:2]-[.12*span, .18*span]).astype(int), [x0, y0])
            right, bottom = np.minimum(np.ceil(box[2:]+.12*span).astype(int)+1, [x1, y1])
            if left < right and top < bottom:
                head[top-y0:bottom-y0, left-x0:right-x0] = inside[top-y0:bottom-y0, left-x0:right-x0]
        face_dark = head & (luma <= CONFIG['dark_luma'])
        regions['face'] = face.nonzero().T+origin
        regions['dark_face_foreground'] = face_dark.nonzero().T+origin
        regions = {key: value for key, value in regions.items() if value.shape[1] >= CONFIG['min_region_pixels']}
        eligibility = {key: int(regions[key].shape[1]) if key in regions else 0
                       for key in ('face', *REGIONS, 'dark_face_foreground')}
        views.append(_FaceView(row['frame'], 'original', source_bin, slot, camera, view, view.mask[0, y, x],
                               face, regions, eligibility))
    for index, item in enumerate(supplemental):
        view = item.guard_view
        x0, y0, _, _ = view.box
        origin = item.valid.new_tensor([[y0], [x0]], dtype=torch.long)
        regions = {'face': item.valid.nonzero().T+origin}
        luma = (item.rgb*item.rgb.new_tensor([.2126, .7152, .0722])[:, None, None]).sum(0)
        dark = item.valid & (luma <= CONFIG['dark_luma'])
        if int(dark.count_nonzero()) >= CONFIG['min_region_pixels']:
            regions['dark_face_foreground'] = dark.nonzero().T+origin
        eligibility = {key: int(value.shape[1]) for key, value in regions.items()}
        views.append(_FaceView(item.row['frame'], 'supplemental', index, item.slot, item.camera, view,
                               item.alpha, item.valid, regions, eligibility))
    if not views or len({view.frame for view in views}) != len(views):
        raise ValueError('Protected face optimization requires a unique original and supplemental roster.')
    return views


def _scene(model, ids, values):
    xyz, scales, rotations = (getattr(model, key).detach() for key in ('get_xyz', 'get_scaling', 'get_rotation'))
    if values is None:
        opacity, features = model.get_opacity.detach(), model.get_features.detach()
    else:
        opacity = model._opacity.detach().index_copy(0, ids, values['opacity']).sigmoid()
        dc = model._features_dc.detach().index_copy(0, ids, values['f_dc'])
        features = torch.cat((dc, model._features_rest.detach()), dim=1)
    return SimpleNamespace(get_xyz=xyz, get_scaling=scales, get_rotation=rotations, get_opacity=opacity,
                           get_features=features, active_sh_degree=0, max_sh_degree=0)


def _draw(model, source, ids, values, render, pipe, background):
    model.select_mesh_by_timestep(source.slot)
    scene = _scene(model, ids, values)
    rgb = render(source.camera, scene, pipe, background)['render']
    alpha = render(source.camera, scene, pipe, background,
                   override_color=torch.ones_like(scene.get_xyz))['render'][0]
    return rgb, alpha


def _source_loss(source, rgb, alpha):
    target = _float_rgb(source.view)
    error = (rgb/alpha.clamp_min(1e-4)-target).abs().mean(0)
    coverage = (alpha-source.target_alpha).abs()
    premultiplied = (rgb-target*source.target_alpha).abs().mean(0)
    face = source.face
    color, opacity, photo = (value[face].sum()/face.count_nonzero().clamp_min(1)
                             for value in (error, coverage, premultiplied))
    score = .30*color+.70*photo+CONFIG['soft_alpha_weight']*opacity
    return score, color, opacity, error, coverage


def _regional_guard(source, alpha, error, coverage, options):
    x0, y0, _, _ = source.view.box
    old = source.baseline_alpha[source.crop]
    old_coverage = (old-source.target_alpha).abs()
    results, failures = {}, []
    tolerance = options['coverage_rgb_mae_increase']+CONFIG['epsilon']
    for name, (y, x) in source.regions.items():
        y, x = y-y0, x-x0
        previous, new = old[y, x], alpha[y, x]
        new_holes = (previous >= CONFIG['near_hole_alpha']) & (new < CONFIG['near_hole_alpha'])
        before_rgb, after_rgb = source.baseline_error[y, x].mean(), error[y, x].mean()
        before_alpha, after_alpha = old_coverage[y, x].mean(), coverage[y, x].mean()
        rgb_change, alpha_change = float(after_rgb-before_rgb), float(after_alpha-before_alpha)
        results[name] = dict(pixels=len(y), rgb_mae_before=float(before_rgb), rgb_mae_after=float(after_rgb),
                             alpha_mae_before=float(before_alpha), alpha_mae_after=float(after_alpha),
                             alpha_below_095_before=int((previous < CONFIG['near_hole_alpha']).count_nonzero()),
                             alpha_below_095_after=int((new < CONFIG['near_hole_alpha']).count_nonzero()),
                             new_alpha_holes_095=int(new_holes.count_nonzero()))
        for key, value, budget in (('region_source_rgb_increase', rgb_change, tolerance),
                                   ('region_source_alpha_increase', alpha_change, tolerance),
                                   ('region_new_alpha_holes_095', results[name]['new_alpha_holes_095'], 0)):
            if value > budget:
                failures.append(dict(item=key, region=name, value=value, budget=budget))
    return results, failures


@torch.no_grad()
def _assess(model, views, ids, values, render, pipe, background, options, target, low_alpha, *, baseline=False):
    frames = []
    by_role = {'original': {}, 'supplemental': {}}
    for source in views:
        rgb, alpha = _draw(model, source, ids, values, render, pipe, background)
        y, x = source.crop
        score, color, opacity, error, coverage = _source_loss(source, rgb[:, y, x], alpha[y, x])
        if not bool(torch.isfinite(score)) or not bool(torch.isfinite(alpha).all()) or not bool(torch.isfinite(rgb).all()):
            return dict(finite=False, all_guards_pass=False, frames=frames, failure='nonfinite_native_render')
        if baseline:
            source.baseline_alpha = alpha.clone()
            source.baseline_error = _rgb_error(rgb, alpha, source.view)
        coverage_guard = _coverage_metrics(source.baseline_alpha, alpha, source.baseline_error,
                                           error, source.view, source.regions, target, low_alpha,
                                           options['coverage_slack_px'], options['coverage_slack_px'], options)
        regions, failures = _regional_guard(source, alpha[y, x], error, coverage, options)
        if bool(source.face.any()):
            by_role[source.role].setdefault(source.source_bin, []).append(float(score))
        frames.append(dict(frame=source.frame, role=source.role, source_bin=source.source_bin,
                           face_pixels=int(source.face.count_nonzero()), region_eligibility=source.eligibility,
                           face_rgb_mae=float(color), face_alpha_mae=float(opacity), face_objective=float(score),
                           regions=regions, coverage_guard=coverage_guard, failures=coverage_guard['failures']+failures))
    role_objective = {key: sum(sum(values)/len(values) for values in group.values())/len(group) if group else None
                      for key, group in by_role.items()}
    original, extra = role_objective['original'], role_objective['supplemental']
    weight = CONFIG['original_weight']
    total = weight*original+(1-weight)*extra if original is not None and extra is not None else (
        original if extra is None else extra)
    failed = [row['frame'] for row in frames if row['failures']]
    return dict(finite=True, all_guards_pass=not failed, failed_frames=failed, frames=frames,
                objective=total, original_objective=original, supplemental_objective=extra)


def _guard_loss(source, alpha, error, coverage, options):
    old = source.baseline_alpha[source.crop]
    rgb_delta = error-source.baseline_error
    target_delta = coverage-(old-source.target_alpha).abs()
    inside = source.view.inside[source.crop]
    tile = options['coverage_tile_px']
    tile_rgb = _tile_sum(rgb_delta.masked_fill(~inside, 0), tile)/_tile_sum(inside.float(), tile).clamp_min(1)
    tolerance = options['coverage_local_rgb_mae_increase']
    penalty = 8*(tile_rgb.amax()-tolerance*.5).relu()
    for y, x in source.regions.values():
        x0, y0, _, _ = source.view.box
        y, x = y-y0, x-x0
        threshold = CONFIG['near_hole_alpha']
        holes = (threshold-alpha[y, x]).relu().masked_fill(old[y, x] < threshold, 0)
        penalty += (3*(rgb_delta[y, x].mean()-tolerance*.5).relu()
                    + 2*(target_delta[y, x].mean()-tolerance*.5).relu()+4*holes.amax())
    return penalty


def _live_groups(model, ids):
    by_name = {group['name']: group for group in model.optimizer.param_groups}
    result = {}
    for name, attribute in GROUPS.items():
        param = getattr(model, attribute)
        group = by_name[name]
        state = model.optimizer.state[param]
        if len(group['params']) != 1 or group['params'][0] is not param or group.get('amsgrad'):
            raise ValueError('Protected face optimization expects ordinary named Adam groups.')
        result[name] = dict(parameter=param, state=state, group=group,
                            values=param[ids].detach().clone(),
                            moments={key: state[key][ids].detach().clone() for key in MOMENTS},
                            step=state['step'].detach().clone())
    return result


def _fingerprints(model, live, ids):
    keep = torch.ones(len(model.binding), dtype=torch.bool, device=ids.device)
    keep[ids] = False
    topology = {key: _digest(getattr(model, key)) for key in FIXED_MODEL
                if torch.is_tensor(getattr(model, key, None))}
    excluded = {}
    mutable = {}
    for group in model.optimizer.param_groups:
        name = group['name']
        for i, param in enumerate(group['params']):
            state = model.optimizer.state.get(param, {})
            if name in live:
                excluded[name] = dict(parameter=_digest(param[keep]), clock=int(state['step']),
                                      **{key: _digest(state[key][keep]) for key in MOMENTS})
                mutable[name] = dict(parameter=_digest(param[ids]), clock=int(state['step']),
                                     **{key: _digest(state[key][ids]) for key in MOMENTS})
            else:
                excluded[f'{name}/{i}'] = dict(parameter=_digest(param),
                                               state={key: _digest(value) for key, value in state.items()
                                                      if torch.is_tensor(value)})
    return dict(fixed_model=topology, fixed_optimizer=excluded, mutable_head=mutable,
                next_densification_id=getattr(model, 'next_densification_id', None), points=len(model.binding))


def _branch(live):
    params = {name: torch.nn.Parameter(item['values'].clone()) for name, item in live.items()}
    settings = {'f_dc': CONFIG['color_lr'], 'opacity': CONFIG['opacity_lr']}
    groups = []
    for name, item in live.items():
        src = item['group']
        groups.append(dict(name=name, params=[params[name]], lr=settings[name], eps=src['eps'], betas=src['betas'],
                           weight_decay=src['weight_decay'], maximize=src['maximize'],
                           capturable=src.get('capturable', False), differentiable=False))
    optimizer = torch.optim.Adam(groups)
    for name, param in params.items():
        item = live[name]
        optimizer.state[param] = dict(step=item['step'].clone(),
                                      **{key: value.clone() for key, value in item['moments'].items()})
    return params, optimizer, settings


@torch.no_grad()
def _limit(params, optimizer, live, bound, *, protected):
    for name, param in params.items():
        start = live[name]['values']
        if name == 'f_dc':
            lower = (start-CONFIG['color_trust']).clamp_min(-bound)
            upper = (start+CONFIG['color_trust']).clamp_max(bound)
        else:
            lower = start if protected else start-CONFIG['opacity_trust']
            upper = start+CONFIG['opacity_trust']
        param.copy_(param.clamp(min=lower, max=upper))
        moment = optimizer.state[param]['exp_avg']
        outward = ((param <= lower) & (moment > 0)) | ((param >= upper) & (moment < 0))
        moment.masked_fill_(outward, 0)


def _ordered(views, cycle):
    kinds = {'original': 0, 'supplemental': 1}
    pairs = [(kinds[view.role], view.source_bin) for view in views]
    return _view_cycle(pairs, cycle)


def _sweep(model, views, ids, params, optimizer, rates, live, bound, render, pipe, background,
           options, cycle, *, protected, rate_factor):
    order = _ordered(views, cycle)
    counts = Counter((view.role, view.source_bin) for view in views)
    bin_counts = Counter(role for role, _ in counts)
    kinds = {'original': CONFIG['original_weight'], 'supplemental': 1-CONFIG['original_weight']}
    for group in optimizer.param_groups:
        group['lr'] = rates[group['name']]*rate_factor*CONFIG['cycle_lr_decay']**cycle
    visited = []
    for index in order:
        source = views[index]
        optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            rgb, alpha = _draw(model, source, ids, params, render, pipe, background)
            y, x = source.crop
            loss, _, _, error, coverage = _source_loss(source, rgb[:, y, x], alpha[y, x])
            if protected:
                loss += _guard_loss(source, alpha[y, x], error, coverage, options)
                weight = kinds[source.role] if len(bin_counts) > 1 else 1.
                loss *= len(views)*weight/(bin_counts[source.role]*counts[(source.role, source.source_bin)])
            else:
                # R18 applied .30*photo+.15*coverage to the main optimizer.
                loss = .30*loss
            loss.backward()
        valid = all(param.grad is not None for param in params.values())
        if not valid or not bool(torch.stack([torch.isfinite(loss), *(
                torch.isfinite(param.grad).all() for param in params.values())]).all()):
            return visited, 'nonfinite_optimizer_update'
        optimizer.step()
        _limit(params, optimizer, live, bound, protected=protected)
        visited.append(source.frame)
    return visited, None


def _snapshot(params, optimizer):
    return dict(values={key: value.detach().clone() for key, value in params.items()},
                states={key: {field: optimizer.state[value][field].detach().clone() for field in ('step', *MOMENTS)}
                        for key, value in params.items()})


def _gains(baseline, measurement):
    if not measurement['finite']:
        return dict(joint=None, supplemental=None)
    initial, final = baseline['supplemental_objective'], measurement['supplemental_objective']
    return dict(joint=baseline['objective']-measurement['objective'],
                supplemental=initial-final if initial is not None and final is not None else None)


def _feasible(gains, measurement):
    return (measurement['finite'] and measurement['all_guards_pass']
            and gains['joint'] >= CONFIG['minimum_gain']
            and (gains['supplemental'] is None or gains['supplemental'] >= CONFIG['minimum_gain']))


def _commit_state(live, checkpoint):
    states = {}
    for name, item in live.items():
        state = checkpoint['states'][name]
        current_step = int(state['step'])
        old_step = int(item['step'])
        moments = {}
        for key, beta in zip(MOMENTS, item['group']['betas'], strict=True):
            # Preserve the main group's scalar clock for every unrelated row.
            ratio = (1-beta**old_step)/(1-beta**current_step)
            moments[key] = state[key]*ratio
        states[name] = dict(step=item['step'].clone(), **moments)
    return dict(values=checkpoint['values'], states=states)


def _apply(live, ids, snapshot):
    for name, item in live.items():
        item['parameter'].index_copy_(0, ids, snapshot['values'][name])
        for key in MOMENTS:
            item['state'][key].index_copy_(0, ids, snapshot['states'][name][key])


def _assessment_summary(measurement):
    summary = {key: value for key, value in measurement.items() if key != 'frames'}
    summary['failure_details'] = [
        dict(frame=row['frame'], role=row['role'], failures=row['failures'])
        for row in measurement['frames'] if row['failures']
    ]
    return summary


def _rng_hash(device):
    value = dict(cpu=_digest(torch.get_rng_state()))
    if device.type == 'cuda':
        value['cuda'] = _digest(torch.cuda.get_rng_state(device))
    return value


@torch.no_grad()
def optimize_protected_face(model, constraints, face_targets, head_faces, render, pipe, background, *,
                            observed, directory=None):
    """Train and verify isolated branches, then atomically commit their protected result."""
    from utils.sh_utils import C0

    started = perf_counter()
    if not face_targets:
        return dict(schema='shuolee-protected-face/1', state='skipped', reason='no_admitted_supplemental_views',
                    optimizer_updates=0, attempted_updates=0, committed_updates=0, main_optimizer_updates=0,
                    seconds=perf_counter()-started, optimizer_state=None)
    if constraints is None or model.active_sh_degree != 0 or model.max_sh_degree != 0:
        raise ValueError('Protected face optimization requires frozen SH0 training constraints.')
    ids = head_faces[model.binding].nonzero(as_tuple=True)[0]
    if not len(ids):
        raise ValueError('No head-owned Gaussian rows are available.')
    views = _views(constraints, face_targets, observed)
    options = constraints.schedule['world_scale']['render_support']
    target, low_alpha = constraints.options['foreground_target'], constraints.options['low_alpha']
    bound = DCColorProjection(model._features_dc, C0).bound
    live = _live_groups(model, ids)
    endpoint = _fingerprints(model, live, ids)
    rng = _rng_hash(ids.device)
    old = dict(values={name: item['values'] for name, item in live.items()},
               states={name: item['moments'] for name, item in live.items()})

    def assess(values=None, *, baseline=False):
        return _assess(model, views, ids, values, render, pipe, background, options, target, low_alpha, baseline=baseline)

    def run_sweep(view_set, branch, cycle, *, protected, rate_factor=1.):
        return _sweep(model, view_set, ids, *branch, live, bound, render, pipe, background, options, cycle,
                      protected=protected, rate_factor=rate_factor)

    with _preserve_pose(model):
        baseline = assess(baseline=True)
        if not baseline['finite'] or not baseline['all_guards_pass'] or baseline['original_objective'] is None:
            raise RuntimeError('Unable to establish a finite protected baseline for original face views.')
        cycles, phases = [], []
        accepted = None
        draws = Counter()
        for phase, rate in (('primary', 1.), ('conservative_restart', CONFIG['conservative_lr_factor'])):
            branch = _branch(live)
            params, optimizer, _ = branch
            previous, stable, completed, phase_updates = None, 0, 0, 0
            stop = 'cycle_budget'
            for cycle in range(CONFIG['max_cycles']):
                order, issue = run_sweep(views, branch, cycle, protected=True, rate_factor=rate)
                draws.update(order)
                phase_updates += len(order)
                measure = assess(params)
                gains = _gains(baseline, measure)
                feasible = issue is None and _feasible(gains, measure)
                record = dict(cycle=len(cycles)+1, phase=phase, phase_cycle=cycle+1, lr_factor=rate,
                              draw_order=order, completed=issue is None and measure['finite'],
                              optimizer_updates=phase_updates, cumulative_attempted_updates=sum(draws.values()),
                              gains=gains, feasible=feasible, **_assessment_summary(measure))
                cycles.append(record)
                if not record['completed']:
                    stop = issue or 'nonfinite_checkpoint'
                    break
                completed += 1
                if feasible and (accepted is None or gains['joint'] > accepted['gains']['joint']):
                    snap = _snapshot(params, optimizer)
                    accepted = dict(cycle=record['cycle'], phase=phase, phase_cycle=cycle+1, lr_factor=rate,
                                    optimizer_updates=phase_updates, local=snap, state=_commit_state(live, snap),
                                    measurement=measure, gains=gains)
                converged = feasible and previous is not None and abs(gains['joint']-previous) <= CONFIG['convergence_delta']
                stable = stable+1 if converged else 0
                previous = gains['joint'] if feasible else None
                if cycle+1 >= CONFIG['min_cycles'] and stable >= CONFIG['stable_cycles']:
                    stop = 'stable_complete_sweeps'
                    break
            phases.append(dict(phase=phase, lr_factor=rate, complete_cycles=completed, optimizer_updates=phase_updates,
                               stop_reason=stop, same_endpoint=True))
            if accepted is not None:
                break

        control_views = [view for view in views if view.role == 'supplemental']
        control = _branch(live)
        control_draws = Counter()
        control_issue = None
        control_cycles = accepted['phase_cycle'] if accepted is not None else phases[0]['complete_cycles']
        control_rate = accepted['lr_factor'] if accepted is not None else 1.
        for cycle in range(control_cycles):
            order, control_issue = run_sweep(control_views, control, cycle, protected=False, rate_factor=control_rate)
            control_draws.update(order)
            if control_issue is not None:
                break
        unprotected = assess(control[0])
        control_result = dict(method='isolated R18-style supplemental RGB/alpha, head DC and opacity only',
                              same_endpoint=True, rng_unchanged=_rng_hash(ids.device) == rng,
                              cycle_budget=control_cycles, lr_factor=control_rate,
                              optimizer_updates=sum(control_draws.values()), view_draws=dict(control_draws),
                              issue=control_issue, gains=_gains(baseline, unprotected), measurement=unprotected)
        if _fingerprints(model, live, ids) != endpoint or _rng_hash(ids.device) != rng:
            raise RuntimeError('Isolated protected-face branches modified the live endpoint or RNG.')

        commit = None
        state = 'rolled_back'
        reason = 'no_feasible_nonzero_gain'
        if accepted is not None:
            try:
                _apply(live, ids, accepted['state'])
                final = assess()
                gains = _gains(baseline, final)
                after = _fingerprints(model, live, ids)
                fixed = {key: value for key, value in after.items() if key != 'mutable_head'}
                endpoint_fixed = {key: value for key, value in endpoint.items() if key != 'mutable_head'}
                if (fixed != endpoint_fixed or after['mutable_head'] == endpoint['mutable_head']
                        or not _feasible(gains, final)
                        or abs(final['objective']-accepted['measurement']['objective']) > 1e-6):
                    raise ValueError('Live applied state did not reproduce the verified branch.')
                if directory is not None:
                    path = Path(directory)/'protected-optimizer-state.pt'
                    temporary = path.with_suffix('.tmp')
                    payload = dict(head_rows=ids.cpu(), endpoint=endpoint, committed=after,
                                   groups={name: dict(step=item['step'].cpu(),
                                       parameter=item['parameter'][ids].detach().cpu(),
                                       **{key: item['state'][key][ids].detach().cpu() for key in MOMENTS})
                                       for name, item in live.items()},
                                   local_groups={name: dict(parameter=accepted['local']['values'][name].cpu(),
                                       **{key: value.cpu() for key, value in states.items()})
                                       for name, states in accepted['local']['states'].items()})
                    try:
                        torch.save(payload, temporary)
                        temporary.replace(path)
                        commit = dict(file=path.name, sha256=rt.sha(path), head_rows=len(ids),
                                      main_adam_clock_preserved=True, groups=list(live),
                                      local_clocks={name: int(state['step'])
                                                    for name, state in accepted['local']['states'].items()},
                                      live_moment_encoding='bias-corrected local moments on unchanged main clock')
                    finally:
                        temporary.unlink(missing_ok=True)
                state, reason = 'committed', 'verified_nonzero_joint_and_supplemental_gain'
            except (ValueError, RuntimeError, OSError) as error:
                _apply(live, ids, old)
                reason = 'live_application_failed: '+str(error)
        if state != 'committed':
            final = assess()
            after = _fingerprints(model, live, ids)
            if after != endpoint or not final['all_guards_pass'] or not math.isclose(
                    baseline['objective'], final['objective'], abs_tol=1e-7, rel_tol=0):
                raise RuntimeError('The protected-face rollback differs from the original endpoint.')

    decision = None if accepted is None else {key: accepted[key] for key in
                                             ('cycle', 'phase', 'phase_cycle', 'lr_factor', 'optimizer_updates', 'gains')}
    attempted_updates = sum(draws.values())
    committed_updates = accepted['optimizer_updates'] if state == 'committed' else 0
    return dict(schema='shuolee-protected-face/1', state=state, reason=reason, config=CONFIG.copy(),
                seconds=perf_counter()-started, head_rows=len(ids), head_row_sha256=_digest(ids),
                endpoint=endpoint, live_final=after, baseline=baseline, unprotected=control_result,
                protected=dict(optimizer_updates=attempted_updates, view_draws=dict(draws), cycles=cycles,
                               phases=phases, stop_reason=phases[-1]['stop_reason']),
                optimizer_updates=attempted_updates, attempted_updates=attempted_updates,
                committed_updates=committed_updates, main_optimizer_updates=0, final=final,
                selected=decision, final_gains=_gains(baseline, final), optimizer_state=commit,
                rng_unchanged=_rng_hash(ids.device) == rng)
