"""Save exact post-update endpoints for independent terminal quality branches."""

from collections import defaultdict
from copy import deepcopy
import random

import numpy as np
import torch

from .size_child_commit import PARAMETERS


SCHEMA = 'shuolee-duration-checkpoint/1'
OPTIMIZED = {**PARAMETERS, 'static_offset': 'static_offset'}
MODEL_FIELDS = (
    'binding', 'binding_counter', 'max_radii2D', 'xyz_gradient_accum', 'denom',
    'densification_ids', 'next_densification_id', 'active_sh_degree', 'spatial_lr_scale', 'percent_dense',
    'timestep', 'face_center', 'face_scaling', 'face_orien_mat', 'face_orien_quat',
)
WORLD_FIELDS = (
    'projections', 'reference_scale', 'maximum_scale', 'cap', 'log_cap',
    'checkpoints', 'splits', 'support_history', 'support_prunes',
)
WORLD_SETS = ('size_candidate_ids', 'optimized_candidate_ids')
LIMIT_FIELDS = ('cap', 'lineage', 'overshoot', 'counts', 'peak_overshoot', 'next_lineage', 'child_counts')
COLOR_FIELDS = ('counts', 'optimizer_steps', 'topology_projections')


def _pack(value):
    if isinstance(value, torch.Tensor):
        return dict(tensor=value.detach().to('cpu', copy=True), device=str(value.device))
    return dict(value=deepcopy(value.item() if isinstance(value, np.generic) else value))


def _unpack(entry):
    if 'tensor' in entry:
        return entry['tensor'].to(entry['device'], copy=True)
    return deepcopy(entry['value'])


def _fields(obj, names):
    return {name: _pack(getattr(obj, name)) for name in names}


def _load_fields(fields):
    return {name: _unpack(value) for name, value in fields.items()}


def _groups(model):
    groups = {}
    for group in model.optimizer.param_groups:
        name = group['name']
        if (name in groups or name not in OPTIMIZED or len(group['params']) != 1
                or group['params'][0] is not getattr(model, OPTIMIZED[name])):
            raise ValueError('Duration checkpoint requires a unique Adam group for each live optimized parameter.')
        groups[name] = group
    if not PARAMETERS.keys() <= groups.keys():
        raise ValueError('Duration checkpoint is missing an expected Gaussian Adam group.')
    return groups


def _capture_rng():
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), numpy=(algorithm, torch.from_numpy(keys.astype(np.int64)),
                position, has_gauss, cached), torch_cpu=torch.get_rng_state().clone(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _restore_rng(state):
    algorithm, keys, position, has_gauss, cached = state['numpy']
    random.setstate(state['python'])
    np.random.set_state((algorithm, keys.numpy().astype(np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@torch.no_grad()
def capture_duration_checkpoint(model, world_scale, dc_projection):
    """Copy the mutable training state to CPU, preserving optimizer device placement."""
    if world_scale.model is not model or model.point_limits is not world_scale.limits:
        raise ValueError('Duration checkpoint requires the active model and its live footprint limits.')
    groups = _groups(model)
    parameters = {}
    for attribute in OPTIMIZED.values():
        parameter = getattr(model, attribute)
        parameters[attribute] = dict(value=_pack(parameter), requires_grad=parameter.requires_grad,
                                     grad=_pack(parameter.grad))
    optimizer = model.optimizer
    states = {}
    for name, group in groups.items():
        state = optimizer.state.get(group['params'][0])
        states[name] = None if state is None else {key: _pack(value) for key, value in state.items()}
    return dict(schema=SCHEMA, model=_fields(model, MODEL_FIELDS), parameters=parameters,
                optimizer=dict(groups={name: {key: _pack(value) for key, value in group.items() if key != 'params'}
                                       for name, group in groups.items()}, states=states,
                               defaults={key: _pack(value) for key, value in optimizer.defaults.items()}),
                world=_fields(world_scale, WORLD_FIELDS),
                world_sets={name: sorted(getattr(world_scale, name)) for name in WORLD_SETS},
                point_limits=_fields(world_scale.limits, LIMIT_FIELDS),
                color_projection=_fields(dc_projection, COLOR_FIELDS),
                fixed=dict(num_timesteps=model.num_timesteps, max_sh_degree=model.max_sh_degree,
                           extent=world_scale.extent, reference=world_scale.reference,
                           c0=dc_projection.c0, color_bound=dc_projection.bound), rng=_capture_rng())


@torch.no_grad()
def restore_duration_checkpoint(model, world_scale, dc_projection, checkpoint):
    """Restore an exact CPU endpoint into the same mesh and optimizer objects."""
    if checkpoint['schema'] != SCHEMA:
        raise ValueError('Unexpected duration checkpoint schema.')
    fixed = checkpoint['fixed']
    if (world_scale.model is not model or model.point_limits is not world_scale.limits
            or model.num_timesteps != fixed['num_timesteps'] or model.max_sh_degree != fixed['max_sh_degree']
            or world_scale.extent != fixed['extent'] or world_scale.reference != fixed['reference']
            or dc_projection.c0 != fixed['c0'] or dc_projection.bound != fixed['color_bound']):
        raise ValueError('Duration checkpoint does not match the live fixed mesh and configuration.')
    groups = _groups(model)
    optim = checkpoint['optimizer']
    if list(groups) != list(optim['groups']) or set(checkpoint['parameters']) != set(OPTIMIZED.values()):
        raise ValueError('Duration checkpoint does not match the live Adam group roster.')
    cuda_rng = checkpoint['rng']['torch_cuda']
    if cuda_rng and len(cuda_rng) != torch.cuda.device_count():
        raise ValueError('Duration checkpoint CUDA random state requires the original visible devices.')

    values = _load_fields(checkpoint['model'])
    world = _load_fields(checkpoint['world'])
    limits = _load_fields(checkpoint['point_limits'])
    color = _load_fields(checkpoint['color_projection'])
    parameters, offset = {}, model.static_offset
    parameter_data = {attribute: (_unpack(row['value']), row['requires_grad'], _unpack(row['grad']))
                      for attribute, row in checkpoint['parameters'].items()}
    for attribute, (value, requires_grad, grad) in parameter_data.items():
        live = getattr(model, attribute)
        if (value.device != live.device or value.dtype != live.dtype or value.shape[1:] != live.shape[1:]
                or (grad is not None and (grad.shape != value.shape or grad.device != value.device
                                          or grad.dtype != value.dtype))):
            raise ValueError(f'Duration checkpoint parameter type or geometry differs for {attribute}.')
        if attribute == 'static_offset':
            if value.shape != offset.shape:
                raise ValueError('Duration checkpoint shared offset differs from the fixed vertex count.')
            parameters[attribute] = offset
        else:
            parameter = torch.nn.Parameter(value, requires_grad=requires_grad)
            parameter.grad = grad
            parameters[attribute] = parameter

    count = len(parameters['_xyz'])
    if (any(len(parameters[name]) != count for name in PARAMETERS.values())
            or any(len(values[name]) != count for name in ('binding', 'max_radii2D', 'xyz_gradient_accum',
                                                           'denom', 'densification_ids'))
            or any(len(limits[name]) != count for name in ('cap', 'lineage', 'overshoot'))
            or values['binding_counter'].shape != model.binding_counter.shape):
        raise ValueError('Duration checkpoint Gaussian rows, topology and limits must remain aligned.')
    restored_states = defaultdict(dict)
    group_options = {name: _load_fields(row) for name, row in optim['groups'].items()}
    for name in groups:
        parameter = parameters[OPTIMIZED[name]]
        saved = optim['states'][name]
        if saved is not None:
            state = _load_fields(saved)
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state and (state[key].shape != parameter.shape or state[key].device != parameter.device):
                    raise ValueError(f'Duration checkpoint Adam moments do not match {name}.')
            restored_states[parameter] = state

    offset_value, offset_grad_enabled, offset_grad = parameter_data['static_offset']
    offset.copy_(offset_value)
    offset.requires_grad_(offset_grad_enabled)
    offset.grad = offset_grad
    vars(model).update(values, **parameters)
    for name, group in groups.items():
        group.clear()
        group.update(group_options[name], params=[parameters[OPTIMIZED[name]]])
    optimizer = model.optimizer
    optimizer.defaults.clear()
    optimizer.defaults.update(_load_fields(optim['defaults']))
    optimizer.state.clear()
    optimizer.state.update(restored_states)
    vars(world_scale).update(world)
    for name, members in checkpoint['world_sets'].items():
        setattr(world_scale, name, set(members))
    vars(world_scale.limits).update(limits)
    vars(dc_projection).update(color)
    _restore_rng(checkpoint['rng'])
