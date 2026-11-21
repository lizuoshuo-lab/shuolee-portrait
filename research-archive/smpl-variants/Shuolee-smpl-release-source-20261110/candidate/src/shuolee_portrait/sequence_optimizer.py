"""Bounded joint optimizer; schedule completion is not numerical convergence."""
import time
import torch
from .sequence_fitting import sequence_residuals, sequence_objective_streamed


_TORSO = (9, 12, 2, 5)
_GRAD_TOL = 1e-9


def _root_data(model, observations, image_heights):
    """Freeze eligibility before conditioning; never promote an ignored anchor."""
    confidence = observations.fitting_confidence().detach().clone()[:, _TORSO]
    targets = observations.targets.detach().clone()[:, _TORSO]
    heights = targets.new_tensor(image_heights)
    if heights.shape != (len(model.frame_ids),) or not (heights > 0).all():
        raise ValueError('Native image heights required.')
    counts = (confidence > 0).sum(1).cpu().tolist()
    support = dict(torso_indices=list(_TORSO), eligible_counts=counts,
                   minimum_required=3, sufficient_observability_established=False)
    # Two shoulders provide only four image constraints for six root variables.
    # Three anchors are a necessary count check, not a rank/conditioning proof.
    support['minimum_count_passed'] = all(count >= 3 for count in counts)
    return dict(confidence=confidence, targets=targets, heights=heights), support


def _root_objective(model, vector, initial_t, data):
    physical = model.unpack(vector)
    out = model.forward_state(physical)
    if not torch.isfinite(out.vertices).all() or float(out.joints[..., 2].min()) <= .05:
        raise RuntimeError('Invalid root trial; not accepted.')
    difference = model.project(out)[:, _TORSO] - data['targets']
    difference = torch.where(data['confidence'][..., None] > 0,
                             difference, torch.zeros_like(difference))
    difference = difference * data['confidence'][..., None]
    difference = difference * (1000. / data['heights'][:, None, None])
    return (difference.square().sum() / len(model.frame_ids) +
            10000. * (physical['transl'][:, 2] - initial_t[:, 2]).square().mean())


def optimize_sequence(model, observations, image_heights, *, seconds=600,
                      max_evaluations=6000, progress=None,
                      root_policy='eligible_torso', initialization_source=None,
                      evaluation_mode='dense'):
    """Fit with an explicit root-initialization policy and unchanged joint loss.

    ``eligible_torso`` retains the original separate root stage and its support
    refusal. ``provided_initialization`` uses the model's supplied root as the
    start of the same five joint stages, without a torso-only pre-alignment.
    This choice establishes neither root observability nor physical accuracy.

    ``streamed`` evaluates the identical frame-separable objective and column
    scaling one frame at a time. Shared identity derivatives are accumulated;
    LBFGS still optimizes the one complete parameter vector and all five stages.
    """
    if evaluation_mode not in ('dense', 'streamed'):
        raise ValueError('evaluation_mode must be dense or streamed.')
    if root_policy not in ('eligible_torso', 'provided_initialization'):
        raise ValueError('Unknown root policy; no implicit fallback is allowed.')
    if initialization_source is not None and (
            not isinstance(initialization_source, str) or not initialization_source.strip()):
        raise ValueError('initialization_source must be a nonempty provenance string.')
    if root_policy == 'provided_initialization' and initialization_source is None:
        raise ValueError('provided_initialization requires an explicit initialization_source.')
    started = time.monotonic()
    evaluations = 0
    stream_work = dict(sweeps_started=0, sweeps_completed=0, frame_forwards_completed=0,
                       objective_backward_calls_completed=0, jacobian_vjp_calls_completed=0,
                       jacobian_vjp_seed_rows_completed=0)
    theta = model.flatten()
    history = []
    state, stop = 'schedule_complete', None
    schedule_complete = False
    active_stage, active_scale = ('root' if root_policy == 'eligible_torso' else 0), None
    best, best_loss = theta.clone(), float('inf')
    root_data, root_support = _root_data(model, observations, image_heights)
    root_initialization = dict(policy=root_policy, source=initialization_source,
                               prealignment_performed=False,
                               provided_root_preserved_before_joint=(root_policy == 'provided_initialization'),
                               sufficient_observability_established=False,
                               physical_accuracy_passed=False)
    if root_policy == 'eligible_torso' and not root_support['minimum_count_passed']:
        return dict(state='root_support_insufficient', stop='Fewer than three eligible torso anchors in a frame; initialization preserved.',
                    schedule_complete=False, numerical_converged=False,
                    final_stationary=None, final_diagnostic=dict(status='unavailable', reason='root_support_insufficient'),
                    root_support=root_support, root_initialization=root_initialization,
                    seconds=time.monotonic()-started,
                    evaluations=0, stages=[], evaluation_mode=evaluation_mode,
                    streamed_work=stream_work,
                    physical_accuracy_passed=False), theta

    def bound():
        if time.monotonic() - started >= seconds or evaluations >= max_evaluations:
            raise TimeoutError('Registered joint fitting budget exhausted.')

    def evaluate_loss(vector, stage):
        nonlocal evaluations
        bound()
        evaluations += 1
        if stage == 'root':
            loss = _root_objective(model, vector, initial_t, root_data)
        else:
            residual, out = sequence_residuals(model, vector, observations, stage, image_heights)
            if not torch.isfinite(out.vertices).all() or float(out.joints[..., 2].min()) <= .05:
                raise RuntimeError('Nonfinite or behind-camera state; not accepted.')
            loss = residual.square().sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite objective; not accepted.')
        return loss

    def stream_boundary():
        # The evaluation slot was already reserved at sweep entry. A partial
        # sweep must not become a candidate, including a mid-Jacobian timeout.
        if time.monotonic() - started >= seconds:
            raise TimeoutError('Registered joint fitting budget exhausted during frame sweep.')

    def evaluate_streamed(vector, stage, with_scaling=False):
        nonlocal evaluations
        bound()
        evaluations += 1
        stream_work['sweeps_started'] += 1
        def count_work(event, count):
            stream_work[event] += count
        if stage != 'root':
            result = sequence_objective_streamed(model, vector, observations, stage, image_heights,
                         with_scaling=with_scaling, boundary=stream_boundary, work_callback=count_work)
            stream_work['sweeps_completed'] += 1
            return result
        # Same separate root objective, only for explicitly eligible support.
        # The provided_initialization policy never takes this branch.
        frozen = vector.detach()
        loss, gradient = vector.new_zeros(()), torch.zeros_like(frozen)
        for i in range(len(model.frame_ids)):
            stream_boundary()
            local_indices = model.frame_parameter_indices(i)
            local = frozen[local_indices].clone().requires_grad_(True)
            physical = model.unpack(frozen.scatter(0, local_indices, local))
            out = model.forward_state(physical, frame_indices=[i])
            count_work('frame_forwards_completed', 1)
            if (not torch.isfinite(out.vertices).all() or not torch.isfinite(out.joints).all()
                    or float(out.joints[..., 2].min()) <= .05):
                raise RuntimeError('Invalid streamed root trial; incomplete sweep not accepted.')
            difference = model.project(out, frame_indices=[i])[:, _TORSO] - root_data['targets'][i:i+1]
            confidence = root_data['confidence'][i:i+1, :, None]
            difference = torch.where(confidence > 0, difference, torch.zeros_like(difference))
            difference = difference * confidence * (1000. / root_data['heights'][i])
            value = (difference.square().sum() +
                     10000. * (physical['transl'][i, 2] - initial_t[i, 2]).square()) / len(model.frame_ids)
            grad = torch.autograd.grad(value, local)[0]
            count_work('objective_backward_calls_completed', 1)
            if not torch.isfinite(value) or not torch.isfinite(grad).all():
                raise RuntimeError('Nonfinite streamed root objective.')
            loss += value.detach()
            gradient.index_add_(0, local_indices, grad.detach())
            del local, physical, out, difference, value, grad
        stream_boundary()
        stream_work['sweeps_completed'] += 1
        return dict(loss=loss, gradient=gradient, complete=True)

    def consider(vector, loss):
        nonlocal best, best_loss
        if loss < best_loss:
            best_loss, best = loss, vector.detach().clone()

    def diagnose(vector, stage, scale=None):
        """Evaluate the supplied endpoint itself, independent of closure state."""
        probe = vector.detach().clone().requires_grad_(True)
        if evaluation_mode == 'streamed':
            evaluated = evaluate_streamed(probe, stage)
            loss, gradient = evaluated['loss'], evaluated['gradient']
        else:
            loss = evaluate_loss(probe, stage)
            gradient = torch.autograd.grad(loss, probe)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError('Nonfinite endpoint gradient.')
        if stage == 'root':
            gradient = gradient[indices]
        optimizer_gradient = gradient if scale is None else gradient * scale
        gradient_max = float(optimizer_gradient.abs().max())
        return dict(status='evaluated', stage=stage, loss=float(loss.detach()),
                    raw_gradient_max=float(gradient.abs().max()),
                    optimizer_gradient_max=gradient_max,
                    stationarity_coordinates=('root_parameters' if stage == 'root' else
                                              'unscaled_parameters' if scale is None else 'frozen_stage_scaled_parameters'),
                    gradient_tolerance=_GRAD_TOL, stationary=gradient_max <= _GRAD_TOL)

    try:
        if root_policy == 'eligible_torso':
            origin = theta.clone()
            root_indices = []
            for name in ['global_orient', 'transl']:
                a, b, _ = model.layout()[name]
                root_indices.extend(range(a, b))
            indices = torch.tensor(root_indices, device=theta.device)
            root = torch.nn.Parameter(theta[indices].clone())
            initial_t = model.state['transl'].detach().clone()
            root_initial = None
            opt = torch.optim.LBFGS([root], lr=1, max_iter=60, max_eval=90,
                                   tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn='strong_wolfe')
            def closure():
                nonlocal root_initial
                opt.zero_grad()
                vector = origin.scatter(0, indices, root)
                if evaluation_mode == 'streamed':
                    evaluated = evaluate_streamed(vector, 'root')
                    loss = evaluated['loss']
                    root.grad = evaluated['gradient'][indices].clone()
                else:
                    loss = evaluate_loss(vector, 'root')
                value = float(loss.detach())
                if root_initial is None:
                    root_initial = value
                consider(vector, value)
                if evaluation_mode == 'dense':
                    loss.backward()
                if root.grad is None or not torch.isfinite(root.grad).all():
                    raise RuntimeError('Nonfinite root gradient.')
                return loss
            opt.step(closure)  # LBFGS returns the entry loss, not the endpoint loss.
            root_initialization['prealignment_performed'] = True
            endpoint_vector = origin.scatter(0, indices, root.detach())
            endpoint = diagnose(endpoint_vector, 'root')
            consider(endpoint_vector, endpoint['loss'])
            theta = best
            restored = diagnose(theta, 'root')
            history.append(dict(stage='root', initial_loss=root_initial, best_loss=restored['loss'],
                                evaluations=evaluations, endpoint=endpoint, restored_best=restored))
            if progress:
                progress(history[-1], theta)
        for stage in range(5):
            active_stage, active_scale = stage, None
            count0 = evaluations
            best, best_loss = theta.clone(), float('inf')
            # Keep the original objective-derived, per-stage column scaling.
            # Accumulate blocks immediately even in the dense reference path.
            if evaluation_mode == 'streamed':
                evaluated = evaluate_streamed(theta, stage, with_scaling=True)
                gradient = evaluated['gradient']
                columns, jtr = evaluated['column_square'], evaluated['jtr']
                value = evaluated['loss']
            else:
                probe = theta.detach().requires_grad_(True)
                bound()
                evaluations += 1
                residual, out = sequence_residuals(model, probe, observations, stage, image_heights)
                if (not torch.isfinite(residual).all() or not torch.isfinite(out.vertices).all()
                        or float(out.joints[..., 2].min()) <= .05):
                    raise RuntimeError('Nonfinite or behind-camera state; not accepted.')
                value = residual.square().sum()
                gradient = torch.autograd.grad(value, probe, retain_graph=True)[0]
                columns, jtr = torch.zeros_like(theta), torch.zeros_like(theta)
                for start in range(0, residual.numel(), 8):
                    bound()
                    n = min(8, residual.numel() - start)
                    basis = torch.zeros((n, residual.numel()), device=theta.device, dtype=theta.dtype)
                    basis[torch.arange(n, device=theta.device), start + torch.arange(n, device=theta.device)] = 1
                    block = torch.autograd.grad(residual, probe, grad_outputs=basis,
                                    is_grads_batched=True, retain_graph=True)[0].detach()
                    columns += block.square().sum(0)
                    jtr += block.T @ residual.detach()[start:start + n]
                del probe, residual, out, block, basis
            scale = torch.rsqrt((2 * columns).clamp_min(1))
            active_scale = scale
            jacobian_error = float((2 * jtr - gradient).abs().max()) / max(1., float(gradient.abs().max()))
            if (not torch.isfinite(columns).all() or not torch.isfinite(jtr).all()
                    or not torch.isfinite(gradient).all() or jacobian_error >= 1e-8):
                raise RuntimeError('Residual/gradient scaling check failed.')
            consider(theta, float(value.detach()))
            del value, gradient, columns, jtr
            origin = theta.clone()
            q = torch.nn.Parameter(torch.zeros_like(theta))
            opt = torch.optim.LBFGS([q], lr=1, max_iter=30, max_eval=45,
                                   tolerance_grad=1e-9, tolerance_change=1e-9, line_search_fn='strong_wolfe')
            first_loss, previous = None, None
            reason, outer = 'outer_budget', 0
            def closure():
                nonlocal first_loss
                opt.zero_grad()
                vector = origin + scale * q
                if evaluation_mode == 'streamed':
                    evaluated = evaluate_streamed(vector, stage)
                    loss = evaluated['loss']
                    q.grad = scale * evaluated['gradient']
                else:
                    loss = evaluate_loss(vector, stage)
                value = float(loss.detach())
                if first_loss is None:
                    first_loss = value
                consider(vector, value)
                if evaluation_mode == 'dense':
                    loss.backward()
                if q.grad is None or not torch.isfinite(q.grad).all():
                    raise RuntimeError('Nonfinite optimizer gradient.')
                return loss
            for outer in range(30):
                opt.step(closure)
                endpoint_vector = origin + scale * q.detach()
                endpoint = diagnose(endpoint_vector, stage, scale)
                current = endpoint['loss']
                consider(endpoint_vector, current)
                if progress:
                    progress(dict(stage=stage, outer=outer, evaluations=evaluations,
                                  seconds=time.monotonic()-started, best_loss=best_loss,
                                  endpoint=endpoint, streamed_work=dict(stream_work)), best)
                if endpoint['stationary']:
                    reason = 'stationary_gradient'
                    break
                if previous is not None and abs(previous-current) / max(abs(previous), abs(current), 1.) <= 1e-9:
                    reason = 'objective_stagnation'
                    break
                previous = current
            theta = best
            restored = diagnose(theta, stage, scale)
            history.append(dict(stage=stage, initial_loss=first_loss, best_loss=restored['loss'],
                                evaluations=evaluations-count0, outer_calls=outer+1, stop=reason,
                                endpoint=endpoint, restored_best=restored,
                                residual_gradient_relative_error=jacobian_error,
                                scale_min=float(scale.min()), scale_max=float(scale.max())))
        schedule_complete = True
    except (TimeoutError, RuntimeError) as error:
        theta = best
        state = 'budget_ended' if isinstance(error, TimeoutError) else 'failed'
        stop = str(error)
    model.restore(theta)
    try:
        final = diagnose(model.flatten(), active_stage, active_scale)
    except (TimeoutError, RuntimeError) as error:
        final = dict(status='unavailable', stage=active_stage, reason=str(error))
        if state == 'schedule_complete':
            state = 'budget_ended' if isinstance(error, TimeoutError) else 'failed'
            stop = str(error)
    stationary = final.get('stationary')
    converged = bool(schedule_complete and state == 'schedule_complete' and stationary)
    return dict(state=state, stop=stop, schedule_complete=schedule_complete,
                numerical_converged=converged, final_stationary=stationary,
                final_diagnostic=final, root_support=root_support,
                seconds=time.monotonic()-started, evaluations=evaluations,
                stages=history, root_initialization=root_initialization,
                evaluation_mode=evaluation_mode,
                streamed_work=stream_work,
                physical_accuracy_passed=False), theta
