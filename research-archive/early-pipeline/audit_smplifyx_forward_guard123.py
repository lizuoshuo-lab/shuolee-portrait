"""Reuse fixed selection/regional/SO3 checks, plus exact chronological seeds."""
import inspect
import json
from pathlib import Path
import numpy as np
import audit_gaussianavatars_params123 as a
import audit_smplifyx_guarded_selection123 as selection
import audit_smplifyx_guarded_result123 as criteria


def local(path):
    return Path(str(path).replace('/mnt/c/', 'C:/').replace('/mnt/d/', 'D:/'))


def invoke(module):
    source = inspect.getsource(module.main)
    assert 'guarded-root-sequence' in source
    source = source.replace('guarded-root-sequence', 'forward-guarded-sequence')
    scope = dict(module.main.__globals__)
    exec(compile(source, '<same-criteria-new-initialization-control>', 'exec'), scope)
    scope['main']()


def main():
    invoke(selection)
    invoke(criteria)
    body = a.REC/'demo-vhap-initialization/fullbody-smplifyx'
    run = body/'forward-guarded-sequence'
    plan = a.read(body/'forward-guarded-sequence-plan.json')
    rows = []
    for item in plan['schedule']:
        frame = item['frame']
        path = run/'frames'/frame/'result.json'
        if not a.lp(path).exists(): break
        result = a.read(path)
        assert result['temporal_seed'] == item
        if item['seed']:
            folder = body/'full-sequence' if item['seed_source']=='frozen_independent_boundary' else run
            seed = folder/'frames'/item['seed']/'winner-camera-space.npz'
            assert local(result['seed_path']) == local(seed)
            assert result['seed_sha256'] == a.sha(seed)
            with np.load(a.lp(seed),allow_pickle=False) as s:
                assert np.array_equal(s['global_orient'],result['initial_state']['body_fit_initial_root'])
        else:
            assert frame == plan['anchor'] == 'frame_000033' and result['seed_path'] is None
            assert all(v['exact'] for v in result['anchor_parity'].values())
        rows.append(dict(frame=frame,seed=item['seed'],seed_source=item['seed_source'],actual_seed_verified=True))
    report=dict(state='all59_seed_paths_verified' if len(rows)==59 else 'partial_seed_paths_verified',
        frames=len(rows),rows=rows,quality_accepted=False,goal_complete=False,
        first_frame_is_not_shape_reference=True,reference_frame='frame_000033')
    a.save(run/'seed-path-audit.json',report)
    print(json.dumps(dict(seed_paths_verified=len(rows))))


if __name__ == '__main__': main()
