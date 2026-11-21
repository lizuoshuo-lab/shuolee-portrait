"""Original official sequence entry with static integer topology on the host.

All floating-point model parameters, fitting and gradients remain on CUDA.
The pinned official entry/monitor/loss/optimizer files remain unchanged.
Existing arguments and old run receipts remain usable without modification.
"""
import argparse
from pathlib import Path

import official_smplifyx_entry123 as entry
import run_official_sequence123 as sequence


def cache_static_topology(body):
    parents = body.parents.detach().cpu()
    if parents.dtype.__str__() != 'torch.int64' or tuple(parents.shape) != (55,):
        raise ValueError('This measured adapter is for the55-node SMPL-X integer hierarchy.')
    ids = parents.tolist()
    if ids[0] != -1 or any(not 0 <= parent < child for child, parent in enumerate(ids[1:], 1)):
        raise ValueError('Invalid parent topology; no reordering or repair is allowed.')
    if any(p.device.type != 'cuda' for p in body.parameters()):
        raise ValueError('Floating-point model parameters must remain on CUDA.')
    body.parents = parents
    return dict(optimization='invariant55_node_parent_indices_on_CPU',
                all_float_model_parameters_on_CUDA=True, topology_values=ids,
                objective_optimizer_precision_unchanged=True)


def main():
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--output-dir')
    options.add_argument('--inspect-only', action='store_true')
    args, _ = options.parse_known_args()
    source = Path(__file__).resolve()
    digest = entry.sha(source)
    original = entry.fit_one
    completed = []
    def fit_one(**kwargs):
        if entry.sha(source) != digest:
            raise ValueError('Runtime adapter changed during the fitting process.')
        metadata = cache_static_topology(kwargs['body_model'])
        arrays, report = original(**kwargs)
        report['runtime_optimization'] = dict(metadata, source_path=str(source), source_sha256=digest)
        completed.append(str(kwargs['result_path']))
        return arrays, report
    entry.fit_one = fit_one
    try:
        sequence.main()
    finally:
        entry.fit_one = original
    if args.inspect_only:
        return
    root = Path(args.output_dir).resolve()
    path = root/'fit/official-fit-result.json'
    receipt = sequence.read(path)
    if receipt['state'] != 'complete' or len(completed) != receipt['frames'] or entry.sha(source) != digest:
        raise ValueError('Runtime receipt does not match the completed frame sequence.')
    receipt['sources'].append(dict(path=str(source), sha256=digest))
    receipt['runtime_optimization'] = dict(name='static55_integer_topology_host_residency',
        float_fitting_on_CUDA=True, official_solver_unchanged=True, no_previous_fit_resumed=True)
    sequence.save(path, receipt)


if __name__ == '__main__':
    main()
