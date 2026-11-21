"""CVO-inspired component, independently implemented; not official reproduction.

Uses physically evidenced FIT families only. It does not estimate a camera or
create material correspondences, and cannot supply human training admission.
"""
from collections import Counter, defaultdict
import hashlib
import json
import re
import numpy as np
from native_scene_package122 import validate_camera


def camera_digest(cameras):
    return hashlib.sha256(json.dumps(cameras, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def build_graph(cameras, observations, *, source_sha256, min_shared):
    if type(min_shared) is not int or min_shared < 1 or not re.fullmatch('[0-9a-f]{64}', source_sha256):
        raise ValueError('Explicit minimum physical support and input SHA required')
    by_id = {}
    for c in cameras:
        validate_camera(c)
        if c['id'] in by_id: raise ValueError('Duplicate camera identity')
        by_id[c['id']] = c
    if len(by_id) < 2: raise ValueError('At least two cameras required')
    splits = defaultdict(set); seen = set(); supported = defaultdict(set); rejected = Counter()
    for o in observations:
        required = {'frame_id','family_id','split','group','visible','identity','static','evidence_sha256'}
        if set(o) != required or o['frame_id'] not in by_id or not isinstance(o['family_id'], str) or not o['family_id']:
            raise ValueError('Explicit known frame and physical-family observation required')
        if o['split'] not in ('fit', 'heldout') or o['group'] not in ('foreground', 'background'):
            raise ValueError('Frozen split and material group required')
        if o['identity'] not in ('confirmed', 'unknown', 'contradicted') or o['static'] not in ('confirmed', 'unknown', 'moving') or type(o['visible']) is not bool:
            raise ValueError('Explicit identity, staticity and visibility states required')
        if not isinstance(o['evidence_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', o['evidence_sha256']):
            raise ValueError('Observation evidence hash required')
        key = (o['frame_id'], o['family_id'])
        if key in seen: raise ValueError('Duplicate physical observation; descriptors cannot inflate support')
        seen.add(key); splits[o['family_id']].add(o['split'])
    if any(len(v) != 1 for v in splits.values()):
        raise ValueError('Whole-family FIT/heldout leakage; never move heldout into FIT')
    # A contradicted identity contaminates the complete family, not just one pair.
    bad_families = {o['family_id'] for o in observations if o['identity'] == 'contradicted' or o['static'] == 'moving'}
    for o in observations:
        if o['split'] != 'fit': rejected['heldout'] += 1; continue
        if o['family_id'] in bad_families: rejected['contradicted_family'] += 1; continue
        if not o['visible'] or o['identity'] != 'confirmed' or o['static'] != 'confirmed':
            rejected['unverified_observation'] += 1; continue
        if o['group'] != 'foreground': rejected['background_only'] += 1; continue
        supported[o['frame_id']].add(o['family_id'])
    edges = []; ids = sorted(by_id)
    for i, a in enumerate(ids):
        for b in ids[i+1:]:
            common = sorted(supported[a] & supported[b])
            if len(common) < min_shared: continue
            ra = np.asarray(by_id[a]['w2c'])[:3,:3]; rb = np.asarray(by_id[b]['w2c'])[:3,:3]
            za = ra.T @ np.array([0.,0.,1.]); zb = rb.T @ np.array([0.,0.,1.])
            angle = float(np.degrees(np.arccos(np.clip(za @ zb, -1, 1))))
            edges.append(dict(frames=[a,b], families=common, common_count=len(common), world_axis_angle_deg=angle))
    return dict(schema='native-cvo-fit-graph-v1', source_sha256=source_sha256, cameras_sha256=camera_digest(cameras),
                frames=ids, edges=edges, rejected_observations=dict(rejected), min_shared=min_shared,
                heldout_used_for_pairing=False, physical_training_admission=False)


def select_cycle(graph, *, seed, policy):
    """One explicitly bounded greedy cycle: every frame has exactly two exposures.

    No search/retry/threshold relaxation if the graph cannot support the cycle.
    This handles odd counts, but is not a general sparse-graph cycle solver.
    """
    if policy not in ('covisible', 'random-qualified'): raise ValueError('Explicit pairing policy required')
    ids = list(graph['frames']); rng = np.random.default_rng(seed)
    order = rng.permutation(ids).tolist(); path = [order.pop()]
    edges = {frozenset(e['frames']): e for e in graph['edges']}
    while order:
        choices = [x for x in order if frozenset((path[-1],x)) in edges]
        if not choices: raise ValueError('No supported cycle continuation; no random fallback')
        if policy == 'covisible':
            rank = {x: i for i,x in enumerate(order)}
            choices.sort(key=lambda x: (-edges[frozenset((path[-1],x))]['common_count'],
                                       edges[frozenset((path[-1],x))]['world_axis_angle_deg'],rank[x]))
        chosen = choices[0]; order.remove(chosen); path.append(chosen)
    if frozenset((path[-1],path[0])) not in edges:
        raise ValueError('No supported closing edge; no silent resample')
    batches = [[path[i],path[(i+1)%len(path)]] for i in range(len(path))]
    validate_schedule(graph, batches)
    return batches


def validate_schedule(graph, batches):
    edges = {frozenset(e['frames']) for e in graph['edges']}
    counts = Counter()
    for batch in batches:
        if len(batch) != 2 or len(set(batch)) != 2 or frozenset(batch) not in edges:
            raise ValueError('Exactly two distinct supported cameras required per update')
        counts.update(batch)
    if set(counts) != set(graph['frames']) or len(set(counts.values())) != 1:
        raise ValueError('Every registered camera must have equal exposure')
    return dict(sorted(counts.items()))


def accumulate_update(model, views, contract, optimizer, *, objective=None):
    """Mean of view-normalized losses, one backward/view and one Adam step.

    Ray3D eval bypasses differentiable projected means. Statistics therefore use
    per-view WORLD position gradients, explicitly not 3DGS screen-grad scores.
    Counts, max radii and sums include both views. This function does not split,
    clone, prune, or bind any Gaussian to SMPL.
    """
    import torch
    from native_training122 import foreground_objective
    from native_portrait_renderer122_v2 import NativeCameraContractV2, render_native
    if not isinstance(contract, NativeCameraContractV2) or contract.projection != 'pinhole':
        raise ValueError('This processed-video component requires explicit full-K pinhole contract')
    if len(views) != 2 or len({v.camera['id'] for v in views}) != 2:
        raise ValueError('Exactly two distinct views required')
    objective = foreground_objective if objective is None else objective
    before = [(v.K.detach().clone(), v.w2c.detach().clone()) for v in views]
    for v in views:
        validate_camera(v.camera)
        if v.K.requires_grad or v.w2c.requires_grad: raise ValueError('Fixed cameras required')
        expected_k = torch.as_tensor(v.camera['K'], device=v.K.device, dtype=v.K.dtype)[None]
        expected_e = torch.as_tensor(v.camera['w2c'], device=v.w2c.device, dtype=v.w2c.dtype)[None]
        if not torch.equal(v.K, expected_k) or not torch.equal(v.w2c, expected_e):
            raise ValueError('Full recorded K/E and tensor differ')
    optimizer.zero_grad(set_to_none=True)
    n = model.initial_count; device = model.means.device
    counts = torch.zeros(n, dtype=torch.int32, device=device)
    radii = torch.zeros(n, device=device); gradnorm_sum = torch.zeros(n, device=device)
    per_view = []; losses = []
    for v in views:
        p = model.parameters_linear()
        rgb, alpha, meta = render_native(p['means'],p['quats'],p['scales'],p['opacities'],p['sh'],v.w2c,v.K,
            v.camera['width'],v.camera['height'],contract=contract,sh_degree=3)
        loss, terms = objective(rgb[0], alpha[0], v)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite per-view objective')
        # The hook sees this backward contribution only, before .grad accumulation.
        contribution = []
        hook = model.means.register_hook(lambda g: contribution.append(g.detach().clone()))
        try: (loss / len(views)).backward()
        finally: hook.remove()
        if len(contribution) != 1: raise RuntimeError('Missing/ambiguous per-view world gradient')
        visible = (meta['radii'][0] > 0).all(-1)
        radius = meta['radii'][0].amax(-1).float()
        gradnorm = torch.linalg.vector_norm(contribution[0] * len(views), dim=-1)
        counts += visible.to(torch.int32)
        radii = torch.maximum(radii, torch.where(visible, radius, 0))
        gradnorm_sum += torch.where(visible, gradnorm, 0)
        per_view.append(dict(frame_id=v.camera['id'],visible=visible.detach(),radius=radius.detach(),
                             world_grad_norm=gradnorm.detach(),visible_count=int(visible.sum())))
        losses.append(dict(frame_id=v.camera['id'],loss=float(loss.detach()),
                           **{k:float(x.detach()) for k,x in terms.items()}))
        del rgb, alpha, meta, loss, contribution, p
    gradients = {}
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError('Missing/nonfinite accumulated gradient: ' + name)
        gradients[name] = p.grad.detach().clone()
    optimizer.step()
    if any(not torch.equal(v.K,k) or not torch.equal(v.w2c,e) for v,(k,e) in zip(views,before)):
        raise RuntimeError('Camera tensor mutated')
    return dict(losses=losses,gradients=gradients,per_view=per_view,visible_count=counts,
                max_projected_radius=radii,world_gradient_norm_sum=gradnorm_sum,
                world_gradient_norm_mean=gradnorm_sum/counts.clamp_min(1),
                statistic='unscaled per-view world-XYZ gradient norms; not screen-space gradients',
                optimizer_steps=1,adaptive_density_implemented=False)
