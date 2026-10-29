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
