"""CPU-only forensic summary of the retained native lifecycle control."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/mnt/c/Users/Administrator/Desktop/GS-SLAM-Lab/outputs/mini-splatting2-mip-depth-001/shuolee/current/robust-portrait-122/checkpoint-lifecycle128')
PARAMS = {1: 'xyz', 2: 'f_dc', 3: 'f_rest', 4: 'scaling', 5: 'rotation', 6: 'opacity'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def same(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def tensor_stats(a, b):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return {'same': same(a, b), 'not_tensor': True}
    result = {'same': same(a, b), 'shape_a': list(a.shape), 'shape_b': list(b.shape),
              'dtype_a': str(a.dtype), 'dtype_b': str(b.dtype), 'elements': a.numel()}
    if a.shape != b.shape:
        return result
    if not a.numel():
        result.update(max_abs=0., mean_abs=0., differing_elements=0)
        return result
    x, y = a.double(), b.double()
    delta = (x-y).abs()
    finite = torch.isfinite(x) & torch.isfinite(y)
    result.update(differing_elements=int((a != b).sum()), finite_both=bool(finite.all()),
                  max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                  rms_abs=float(delta.square().mean().sqrt()),
                  reference_max_abs=float(x.abs().max()))
    return result


def capture_stats(a, b):
    result = {'same_entire_capture': same(a, b), 'parameter_tensors': {},
              'density_and_binding': {}, 'adam': []}
    for i, name in PARAMS.items():
        result['parameter_tensors'][name] = tensor_stats(a[i], b[i])
    for i, name in [(7, 'binding'), (8, 'binding_counter'), (9, 'max_radii2D'),
                    (10, 'xyz_gradient_accum'), (11, 'denom')]:
        result['density_and_binding'][name] = tensor_stats(a[i], b[i])
    result['active_sh_degree_equal'] = same(a[0], b[0])
    result['spatial_lr_scale_equal'] = same(a[13], b[13])
    oa, ob = a[12], b[12]
    result['adam_param_group_metadata_exact'] = same(oa['param_groups'], ob['param_groups'])
    for ga, gb in zip(oa['param_groups'], ob['param_groups']):
        item = {'name': ga['name'], 'param_ids_equal': ga['params'] == gb['params'], 'states': []}
        for ida, idb in zip(ga['params'], gb['params']):
            sa, sb = oa['state'].get(ida, {}), ob['state'].get(idb, {})
            item['states'].append({'keys_equal': sa.keys() == sb.keys(), 'same': same(sa, sb),
                'values': {str(k): tensor_stats(sa[k], sb[k]) if torch.is_tensor(sa[k]) else {'same': same(sa[k], sb[k])}
                           for k in sa.keys() & sb.keys()}})
        result['adam'].append(item)
    return result


def trace_stats(a, b):
    assert len(a) == len(b)
    discrete = ('step', 'frame', 'python_random', 'numpy_random', 'cpu_random',
                'cuda_random', 'background_state_sha', 'points')
    rows = []
    for x, y in zip(a, b):
        rows.append({'step': x['step'], 'discrete_equal': all(same(x[k], y[k]) for k in discrete),
                     'discrete_differing_keys': [k for k in discrete if not same(x[k], y[k])],
                     'loss_exact': x['loss'] == y['loss'],
                     'loss_a': x['loss'], 'loss_b': y['loss'],
                     'loss_abs_diff': abs(x['loss']-y['loss'])})
    return {'length': len(rows), 'all_discrete_exact': all(x['discrete_equal'] for x in rows),
            'first_discrete_difference': next((x['step'] for x in rows if not x['discrete_equal']), None),
            'first_loss_difference': next((x['step'] for x in rows if not x['loss_exact']), None),
            'max_loss_abs_diff': max(x['loss_abs_diff'] for x in rows), 'rows': rows}


def main():
    assert not torch.cuda.is_initialized(), 'CPU analysis must not initialize CUDA'
    files = {key: ROOT/path for key, path in {
        'continuous_native8': 'continuous/chkpnt8.pth', 'staged_native8': 'staged/chkpnt8.pth',
        'continuous24': 'continuous/actual-state-24.pt', 'staged8': 'staged/actual-state-08.pt',
        'staged24': 'staged/actual-state-24.pt',
        'continuous_render24': 'continuous/actual-render-24.pt',
        'staged_render24': 'staged/actual-render-24.pt'}.items()}
    before_hashes = {str(p): sha(p) for p in files.values()}
    loaded = {k: load(p) for k, p in files.items()}
    c8, s8 = loaded['continuous_native8'], loaded['staged_native8']
    c, s, early = loaded['continuous24'], loaded['staged24'], loaded['staged8']
    assert c8[1] == s8[1] == 8
    prefix = trace_stats(c['counters']['traces'][:8], early['counters']['traces'])
    alltrace = trace_stats(c['counters']['traces'], s['counters']['traces'])
    final_discrete = {k: same(c[k], s[k]) for k in
                      ('python_rng', 'numpy_rng', 'cpu_rng', 'cuda_rng', 'paired_rng', 'paired_calls')}
    final_discrete['camera_counts_exact'] = c['counters']['counts'] == s['counters']['counts']
    result = {
        'schema': 'synthetic-lifecycle128-cpu-numerical-diagnosis-v1',
        'cpu_only': True, 'strict_bitwise_passed': False, 'strict_failure_preserved': True,
        'new_threshold_or_pass_criterion_added': False, 'quality_accepted': False,
        'person_data_used': False, 'source_hashes': before_hashes,
        'pre_resume_step8_capture': capture_stats(c8[0], s8[0]),
        'pre_resume_prefix8_trace': prefix, 'complete24_trace': alltrace,
        'final24_capture': capture_stats(c['capture'], s['capture']),
        'final24_static_offset': tensor_stats(c['static_offset'], s['static_offset']),
        'final24_discrete_states_exact': final_discrete,
        'final24_native_render': tensor_stats(loaded['continuous_render24'], loaded['staged_render24']),
        'counts': {'continuous': c['counters']['counts'], 'staged': s['counters']['counts']},
        'pre_resume_float_divergence_already_present': not same(c8[0], s8[0]),
        'interpretation': [
            'Any step8 capture or prefix loss difference precedes the resume operation and cannot be caused by resume.',
            'Exact discrete states and sample traces are assessed separately from floating model/render differences.',
            'A continuous-repeat control is still needed to characterize native CUDA run-to-run variation; this analysis does not prove atomic operations are the unique cause.',
            'No posterior difference magnitude is an acceptance tolerance; the original strict test remains failed.'],
    }
    assert before_hashes == {str(p): sha(p) for p in files.values()}, 'Frozen input changed during CPU analysis'
    assert not torch.cuda.is_initialized()
    result['input_hashes_unchanged'] = True
    path = ROOT/'numerical-diagnosis128.json'
    assert not path.exists(), 'Preserve any earlier diagnosis'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(path), 'sha256': sha(path),
        'prefix_first_loss_difference': prefix['first_loss_difference'],
        'pre_resume_capture_exact': result['pre_resume_step8_capture']['same_entire_capture'],
        'all24_discrete_exact': alltrace['all_discrete_exact'],
        'final_discrete': final_discrete,
        'render': result['final24_native_render']}, allow_nan=False))


if __name__ == '__main__':
    main()
