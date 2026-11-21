"""Commit a verified local split without changing unrelated Gaussian state."""

import torch

from .sh0_main import DCColorProjection


PARAMETERS = {
    'xyz': '_xyz', 'f_dc': '_features_dc', 'f_rest': '_features_rest',
    'opacity': '_opacity', 'scaling': '_scaling', 'rotation': '_rotation',
}
MOMENTS = ('exp_avg', 'exp_avg_sq')
LOCAL_GROUPS = ('xyz', 'scaling', 'opacity', 'f_dc')


def _child_order(rows):
    return rows.transpose(0, 1).flatten(0, 1).contiguous()


@torch.no_grad()
def commit_size_children(model, limits, parents, result, chosen, world_cap):
    """Publish only the exact verified subset; restore all live state if publication fails."""
    from utils.sh_utils import C0

    parents = parents[chosen]
    if not len(parents) or len(parents.unique()) != len(parents):
        raise ValueError('Size replacement requires a nonempty, unique parent subset.')
    centers = result.centers[chosen]
    log_sigma = result.log_sigma[chosen]
    logits = result.opacity_logits[chosen]
    dc = result.dc[chosen]
    count, children = centers.shape[:2]
    child_rotation = getattr(result, 'child_rotation', None)
    rotations = (model._rotation[parents, None].expand(-1, children, -1) if child_rotation is None else
                 child_rotation[chosen])
    if rotations.shape != (count, children, 4) or not bool((rotations.norm(dim=-1) > 1e-8).all()):
        raise ValueError('A verified split requires one valid local rotation per child.')
    if children < 2 or not all(torch.isfinite(x).all() for x in (centers, log_sigma, logits, dc, rotations)):
        raise ValueError('Invalid optimized size-replacement parameters.')
    if child_rotation is not None:
        from .render_support import child_world_rotation

        owners = model.binding[parents].repeat(children)
        expected_rotation = child_world_rotation(model, owners, _child_order(rotations))
        expected_scaling = _child_order(log_sigma).exp()*model.face_scaling[owners]
    if bool((log_sigma > torch.minimum(limits.cap[parents], world_cap[parents])[:, None]).any()):
        raise ValueError('An optimized child exceeds its inherited local or world scale limit.')
    rgb_bound = DCColorProjection(dc, C0).bound
    if bool((dc.abs() > rgb_bound).any()):
        raise ValueError('An optimized child exceeds the standard SH0 color range.')

    optimizer = model.optimizer
    groups = {group['name']: group for group in optimizer.param_groups}
    extensions = {}
    for name in LOCAL_GROUPS:
        if name == 'opacity' and result.audit.get('local_opacity_optimized') is False:
            continue  # The compositing seed kept its fresh Adam moments.
        group = groups[name]
        parameter = group['params'][0]
        live = optimizer.state[parameter]
        local = result.moments[name]
        local_step, main_step = int(local['step']), int(live['step'])
        if local_step < 1 or main_step < 1 or local['betas'] != group['betas']:
            raise ValueError('Size replacement requires initialized Adam states with matching beta values.')
        extensions[name] = {}
        for moment, beta in zip(MOMENTS, group['betas'], strict=True):
            # Keep the child's current bias-corrected moments at the main
            # optimizer's shared scalar clock. Existing rows keep that clock.
            factor = (1-beta**main_step)/(1-beta**local_step)
            value = _child_order(local[moment][chosen])*factor
            if value.shape != (count*children, *parameter.shape[1:]) or not torch.isfinite(value).all():
                raise ValueError('An optimized child Adam tensor is invalid.')
            extensions[name][moment] = value

    owners = model.binding[parents].repeat(children)
    appended = len(owners)
    keep = torch.ones(len(model.binding), device=owners.device, dtype=torch.bool)
    keep[parents] = False
    prune = torch.cat((~keep, keep.new_zeros(appended)))
    attributes = (*PARAMETERS.values(), 'binding', 'binding_counter', 'max_radii2D',
                  'xyz_gradient_accum', 'denom', 'densification_ids', 'next_densification_id')
    previous = {name: getattr(model, name) for name in attributes}
    old_limits = vars(limits).copy()
    old_groups = [(group, list(group['params'])) for group in optimizer.param_groups]
    old_states = {parameter: state.copy() for parameter, state in optimizer.state.items()}
    try:
        model.binding_counter = model.binding_counter.clone()
        for name in ('cap', 'lineage', 'counts', 'child_counts'):
            setattr(limits, name, getattr(limits, name).clone())
        limits.set(parents, log_sigma.amax(1))
        model.binding = torch.cat((model.binding, owners))
        model.binding_counter.scatter_add_(0, owners, torch.ones_like(owners, dtype=torch.int32))
        model.densification_postfix(_child_order(centers), _child_order(dc),
            previous['_features_rest'][parents].repeat(children, 1, 1), _child_order(logits),
            _child_order(log_sigma), _child_order(rotations),
            constraint_parents=parents.repeat(children))
        limits.cap[-appended:].copy_(_child_order(log_sigma))
        for name in ('xyz_gradient_accum', 'denom'):
            value = previous[name]
            setattr(model, name, torch.cat((value, value.new_zeros((appended, *value.shape[1:])))))
        for name, moments in extensions.items():
            live = optimizer.state[groups[name]['params'][0]]
            for moment, value in moments.items():
                live[moment][-appended:].copy_(value)
        model.prune_points(prune)

        surviving = len(keep)-count
        if len(model.binding) != surviving+appended or int(model.binding_counter.min()) < 1:
            raise RuntimeError('Optimized size replacement violated the per-face binding guarantee.')
        if not torch.equal(model.densification_ids[:surviving], previous['densification_ids'][keep]):
            raise RuntimeError('Optimized size replacement changed unrelated point identities.')
        for attribute in ('binding', 'max_radii2D', 'xyz_gradient_accum', 'denom'):
            if not torch.equal(getattr(model, attribute)[:surviving], previous[attribute][keep]):
                raise RuntimeError('Optimized size replacement changed unrelated binding or raster statistics.')
        for attribute, value in (('_xyz', centers), ('_scaling', log_sigma), ('_rotation', rotations),
                                 ('_opacity', logits), ('_features_dc', dc)):
            if not torch.equal(getattr(model, attribute)[-appended:], _child_order(value)):
                raise RuntimeError('The committed child parameters differ from the verified proposal.')
        if child_rotation is not None and (not torch.equal(model.binding[-appended:], owners)
                or not torch.allclose(model.get_rotation[-appended:], expected_rotation, atol=1e-7, rtol=0)
                or not torch.equal(model.get_scaling[-appended:], expected_scaling)):
            raise RuntimeError('Child world covariance inputs differ from the native proposal.')
        for name, attribute in PARAMETERS.items():
            old_param = previous[attribute]
            new_param = getattr(model, attribute)
            if not torch.equal(new_param[:surviving], old_param[keep]):
                raise RuntimeError('Optimized size replacement changed an unrelated parameter row.')
            old_state, new_state = old_states[old_param], optimizer.state[new_param]
            if int(old_state['step']) != int(new_state['step']) or any(
                    not torch.equal(new_state[key][:surviving], old_state[key][keep]) for key in MOMENTS):
                raise RuntimeError('Optimized size replacement changed unrelated Adam state.')
        if any(not torch.equal(optimizer.state[groups[name]['params'][0]][key][-appended:], values[key])
               for name, values in extensions.items() for key in MOMENTS):
            raise RuntimeError('Optimized child Adam moments did not survive the commit.')
    except BaseException:
        for name, value in previous.items():
            setattr(model, name, value)
        vars(limits).update(old_limits)
        for group, parameters in old_groups:
            group['params'] = parameters
        optimizer.state.clear()
        optimizer.state.update(old_states)
        raise

    return dict(children=appended, parent_ids=previous['densification_ids'][parents].tolist(),
                child_ids=model.densification_ids[-appended:].tolist(),
                unchanged_adam_rows=surviving, moment_groups=list(extensions),
                world_covariance_inputs_verified=child_rotation is not None)
