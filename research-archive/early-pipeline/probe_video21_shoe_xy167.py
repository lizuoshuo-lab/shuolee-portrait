"""Test one added native shoe reprojection term in the 021 depth control."""
from datetime import datetime, timezone
from importlib.util import find_spec
import inspect
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

from extract_video21_shoe_depth162 import replace
import probe_video21_shoe_fit165 as baseline
from update_leafwoman129_state import atomic, read, sha


OUT = baseline.BASE.OUT / 'shoe-depth-xy167'


def evaluation_anchors(plan, tracks):
    """Reuse only reviewed material tracks excluded from both residuals."""
    with np.load(baseline.BASE.original.ASSET, allow_pickle=False) as model:
        weights = model['weights']
    checks = {}
    for frame in plan['frames']:
        pair = plan['pairs'][frame]
        index = pair['native_index']
        used = {(a['window'], a['material']) for a in pair['anchors']}
        seed = pair['seed']
        assert sha(seed['output']) == seed['sha256']
        with np.load(seed['output'], allow_pickle=False) as mesh:
            vertices = mesh['vertices_camera'][0].astype(float)
            faces = mesh['faces']
            K = mesh['K_integer_centers']
        tags = np.full(len(faces), 'other', dtype='<U5')
        for side, bones in baseline.FOOT_BONES.items():
            on_foot = weights[:, bones].sum(1) > .5
            tags[on_foot[faces].all(1)] = side
        anchors = []
        for track in tracks:
            window, material = track['reference'], track['material']
            if ((window, material) in used or material not in baseline.REVIEWED[window]
                    or index not in track['fit_native_indices']):
                continue
            assert track['state'] == 'local_prediction' and track['held_max_px'] < 1.
            side = material.split('_')[0]
            uv = track['observed_uv'][str(index)]
            hit = baseline.bind_one(vertices, faces, tags, K, uv, side)
            if hit['bound'] and hit['pixel_distance'] == 0.:
                anchors.append(dict(window=window, material=material, side=side, uv=uv,
                                    tracker_held_max_px=track['held_max_px'], **hit))
        checks[frame] = anchors
    return checks


def fit_shard(folder, shard):
    source = inspect.getsource(baseline.fit_shard)
    source = replace(source, '        def anchored_loader(root,anchor,receipt):',
                  '        observed = torch.as_tensor([a["uv"] for a in pair["anchors"]], dtype=bary.dtype)\n\n'
                  '        def anchored_loader(root,anchor,receipt):')
    source = replace(source, '                    return original+loss.robustifier(mismatch).sum()*loss.data_weight.square()',
                  '                    surface = (obj.vertices[:, ids, :] * bary[None, :, :, None]).sum(2)\n'
                  '                    pixel_error = cam(surface) - observed[None]\n'
                  '                    depth_cost = loss.robustifier(mismatch).sum()\n'
                  '                    pixel_cost = loss.robustifier(pixel_error).sum()\n'
                  '                    return original + (depth_cost + pixel_cost) * loss.data_weight.square()')
    source = replace(source, '        return arrays,receipt',
                  '        receipt.update(shoe_xy_term_enabled=True, shoe_xy_calls=calls)\n'
                  '        return arrays,receipt')
    scope = dict(baseline.__dict__, __file__=__file__)
    exec(compile(source, '<matched-shoe-depth-plus-native-xy>', 'exec'), scope)
    scope['fit_shard'](folder, shard)


def main():
    assert os.name == 'nt' and not OUT.exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run in the existing VPoser environment.')
    source_plan = baseline.OUT / 'plan.json'
    source_result = baseline.OUT / 'result.json'
    plan, result = read(source_plan), read(source_result)
    assert result['completed'] and result['plan_sha256'] == sha(source_plan)
    assert result['frames'] == plan['frames'] and not set(plan['frames']) & set(plan['retained_holdouts'])
    assert all(sha(path) == digest for path, digest in plan['source_hashes'].items())
    for branch, digest in result['branch_result_sha256'].items():
        assert sha(baseline.OUT / branch / 'result.json') == digest
    source_tracks = baseline.MATERIAL / 'result.json'
    checks = evaluation_anchors(plan, read(source_tracks)['rows'])
    assert checks['frame_000029'] and checks['frame_000053'], 'Require checks on both verified shoe windows.'
    paths = [Path(__file__), Path(baseline.__file__), Path(inspect.getsourcefile(replace)),
             source_plan, source_result, source_tracks,
             baseline.OUT / 'depth/result.json', baseline.OUT / 'control/result.json',
             baseline.BASE.original.ASSET]
    registered = dict(plan, updated=datetime.now(timezone.utc).isoformat(),
        runner_sha256=sha(__file__), branches=['depth_xy'],
        source_hashes=plan['source_hashes'] | {str(path): sha(path) for path in paths},
        hypothesis='Adding native XY for the same two bound shoe material proxies preserves their position while retaining the relative depth constraint.',
        single_change='Two full-K native shoe image residuals in the existing GMoF and stage data weight. No tuning.',
        fixed='Same five input frames, exact original starting archives, depth pair, relative depth weight, full K, shared betas, 2D detector input, final stages, LBFGS and rotation anchor as 165 depth.',
        evaluation_anchors=checks,
        criteria='Record used-anchor and excluded-anchor RMS separately. Require better excluded-anchor projection in both material windows, no worsening of the matched depth error median, and the existing pose gate and source/side review before admitting any frame.',
        limitation='Barycentric points are visible shoe-surface proxies associated on the seed ray, not identity of bare skin and shoe. Heldout material tracks here are excluded from the fit, but share video and the same mesh association method.',
        gaussian_started=False, quality_accepted=False)
    atomic(OUT / 'plan.json', registered)
    atomic(OUT / 'launch.json', dict(pid=os.getpid(), command=[sys.executable, *sys.argv],
        frames=plan['frames'], excluded_anchor_counts={f: len(rows) for f, rows in checks.items()}))
    print('VIDEO21_SHOE_XY_PREFLIGHT', {f: len(rows) for f, rows in checks.items()}, flush=True)
    scope = dict(baseline.BASE.__dict__, fit_shard=fit_shard)
    stage = types.FunctionType(baseline.BASE.stage.__code__, scope)
    indices = [plan['source_frames'].index(frame) for frame in plan['frames']]
    started = time.monotonic()
    try:
        # The reused 165 worker recognizes this folder name as its depth branch.
        rows = stage(OUT.name + '/depth', registered, indices, plan['shared_shape'])
        assert all(r['shoe_xy_term_enabled'] and r['shoe_xy_calls'] > 0 for r in rows)
        assert all(sha(path) == digest for path, digest in registered['source_hashes'].items())
        atomic(OUT / 'result.json', dict(completed=True, frames=plan['frames'],
            plan_sha256=sha(OUT / 'plan.json'), fit_result_sha256=sha(OUT / 'depth/result.json'),
            seconds=time.monotonic() - started, gaussian_started=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
