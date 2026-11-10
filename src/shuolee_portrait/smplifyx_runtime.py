"""Runtime-only adaptation for the official SMPLify-X fitting entry."""


def cache_static_topology(body):
    """Keep the 55 fixed parent indices on CPU after the body's CUDA transfer.

    Call before fitting, not inside the loss closure. No floating tensor or
    fitting option is changed. A subsequent body.to()/cuda() moves buffers
    again; call this function after that transfer. To restore the old runtime,
    move only body.parents back to its original device before the next fit.
    """
    import torch

    original = body.parents
    if original.dtype != torch.int64 or tuple(original.shape) != (55,):
        raise ValueError('Expected the 55-node SMPL-X int64 parent hierarchy.')
    parents = original.detach().cpu()
    ids = parents.tolist()
    if ids[0] != -1 or any(not 0 <= parent < child
                           for child, parent in enumerate(ids[1:], 1)):
        raise ValueError('Invalid parent topology; no reordering or repair is allowed.')
    tensors = list(body.parameters()) + list(body.buffers())
    if any(t.is_floating_point() and t.device.type != 'cuda' for t in tensors):
        raise ValueError('Floating-point model parameters and buffers must remain on CUDA.')
    body.parents = parents
    return dict(optimization='invariant55_node_parent_indices_on_CPU',
                all_float_model_parameters_on_CUDA=True, topology_values=ids,
                objective_optimizer_precision_unchanged=True)


SHARED_BODY_WEIGHT = 30.


def install_shared_body_loss(fitting, target, trace):
    """Add the validated body-only rotation term to a freshly loaded loss factory."""
    from smplx.lbs import batch_rodrigues

    original_factory = fitting.create_loss
    original_loss = fitting.SMPLifyLoss
    target_rotations = batch_rodrigues(target.reshape(-1, 3)).reshape(1, 21, 3, 3).detach()

    class SharedBodyLoss(original_loss):
        def forward(self, body_model_output, *args, **kwargs):
            loss = super().forward(body_model_output, *args, **kwargs)
            pose = body_model_output.full_pose
            rotations = batch_rodrigues(pose[:, :66].reshape(-1, 3)).reshape(-1, 22, 3, 3)
            difference = rotations[:, 1:] - target_rotations
            penalty = SHARED_BODY_WEIGHT * (.5 * difference.square().sum(dim=(-2, -1))).sum()
            trace['calls'] += 1
            trace['last'] = penalty.detach()
            return loss + penalty

    def create_loss(loss_type='smplify', **kwargs):
        if loss_type == 'smplify':
            return SharedBodyLoss(**kwargs)
        return original_factory(loss_type=loss_type, **kwargs)

    # Upstream SMPLifyLoss uses explicit super(SMPLifyLoss, self); keep its global name.
    fitting.create_loss = create_loss
