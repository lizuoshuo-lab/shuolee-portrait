"""One bounded artifact replay: are frozen xy and descriptors from the same pixels?"""
from pathlib import Path
import ast
import json
import os
import time
import traceback
import cv2
import numpy as np
import audit_smplify_observation_contract123 as a

OUT = a.REF / 'solver-track-membership/feature-binding-replay'
SLOTS = [31, 35]
TID = '82e918800e2525059cba07a2f837fbae3023c1fa09371cd75f34a9b12d1901b7'

def main():
    assert not OUT.exists(), 'Once-only artifact replay already exists; inspect it.'
    OUT.mkdir()
    src = a.H / 'local_surface123_camera.py'
    assert a.sha(src) == a.read(a.ROOT / 'camera/plan.json')['source_sha256']
    tree = ast.parse(src.read_text())
    camera = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'camera')
    # Execute the actual frozen feature block, not a rewritten implementation.
    start = next(i for i, n in enumerate(camera.body)
                 if ast.unparse(n).startswith('cv2.setNumThreads(6)'))
    end = next(i for i, n in enumerate(camera.body)
               if isinstance(n, ast.For) and ast.unparse(n.target) == '(i, r)')
    feature_loop = camera.body[end]
    save_index = next(i for i, n in enumerate(feature_loop.body)
                      if ast.unparse(n).startswith('np.savez_compressed'))
    feature_nodes = camera.body[start:end] + feature_loop.body[:save_index]
    code = compile(ast.Module(body=feature_nodes, type_ignores=[]), str(src), 'exec')
    decoded = a.read(a.ROOT / 'decoded.json')['records']
    rows = a.read(a.ROOT / 'data/training-inputs.json')['records']
    tr = next(t for t in a.read(a.ROOT / 'camera/appearance-tracks.json')['records'] if t['id'] == TID)
    nodes = dict(tr['nodes'])
    inputs = [src, a.ROOT / 'decoded.json', a.ROOT / 'data/training-inputs.json',
              a.ROOT / 'camera/appearance-tracks.json', a.ROOT / 'camera/solver-output.npz']
    inputs += [a.ROOT / f'camera/features-{s:03d}.npz' for s in SLOTS]
    inputs += [a.ROOT / 'raw' / decoded[s]['name'] for s in SLOTS]
    hashes = {str(p): a.sha(p) for p in inputs}
    a.save(OUT / 'plan.json', dict(registered=a.now(), hypothesis=
        'The verified mast-to-hand false family could be caused by a saved xy/descriptor indexing mismatch. '
        'Replay exactly two original feature blocks to distinguish that implementation defect from genuine descriptor ambiguity.',
        source_frames=[rows[s]['source_frame_index'] for s in SLOTS], slots=SLOTS,
        budget=dict(cpu_seconds=120, raw_image_detector_replays=2, exported_new_features=0,
                    pair_graph_rebuilds=0, camera_updates=0, fit_steps=0),
        selection='Two ends of the already visually verified mast-to-person accepted edge; no residual or threshold search.',
        execution='AST-selected original detector/ordering/spatial-dedup block. New arrays stay in memory and only compare against frozen arrays.',
        stops='Any mismatch is reported; do not replace frozen features or continue with alternate detector parameters.',
        source_hashes=hashes, opencv=cv2.__version__, held_remains_held=True))
    a.save(OUT / 'progress.json', dict(state='running', pid=os.getpid(), started=a.now()))
    began = time.monotonic()
    result = []
    for s in SLOTS:
        assert time.monotonic() - began < 120
        r = decoded[s]
        assert (a.ROOT / 'raw' / r['name']).resolve() == Path(rows[s]['image']).resolve()
        scope = dict(cv2=cv2, np=np, root=a.ROOT, r=r)
        exec(code, scope)
        frozen = dict(np.load(a.ROOT / f'camera/features-{s:03d}.npz'))
        xy, ds = scope['xy'][0], scope['ds'][0]
        same_shape = xy.shape == frozen['xy'].shape and ds.shape == frozen['descriptors'].shape
        exact_xy = same_shape and np.array_equal(xy, frozen['xy'])
        exact_ds = same_shape and np.array_equal(ds, frozen['descriptors'])
        index = nodes[s]
        record = dict(slot=s, source_frame=rows[s]['source_frame_index'],
                      stored_features=len(frozen['xy']), replayed_features=len(xy),
                      all_xy_bit_identical=bool(exact_xy), all_descriptors_bit_identical=bool(exact_ds),
                      inspected_track_index=index, inspected_track_uv=frozen['xy'][index].tolist(),
                      inspected_track_same_xy=bool(np.array_equal(xy[index], frozen['xy'][index])) if same_shape else False,
                      inspected_track_same_descriptor=bool(np.array_equal(ds[index], frozen['descriptors'][index])) if same_shape else False)
        result.append(record)
    for p, digest in hashes.items():
        assert a.sha(Path(p)) == digest
    assert time.monotonic() - began < 120
    passed = all(r['all_xy_bit_identical'] and r['all_descriptors_bit_identical'] for r in result)
    a.save(OUT / 'result.json', dict(updated=a.now(), records=result,
        sampled_saved_binding_reproduced_exactly=passed, source_hashes_unchanged=True,
        elapsed_seconds=time.monotonic()-began, no_new_feature_arrays_saved=True,
        no_camera_or_model_update=True, physical_correspondence_passed=False,
        interpretation='Exact replay rules out saved xy/descriptor permutation for these two frames only; it does not certify semantic identity or all frames.'))
    a.save(OUT / 'progress.json', dict(state='complete', pid=None, finished=a.now()))
    print(json.dumps(dict(records=result, sampled_saved_binding_reproduced_exactly=passed,
                         seconds=time.monotonic()-began)))

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            a.save(OUT / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), retry=False))
            a.save(OUT / 'progress.json', dict(state='failed', pid=None, finished=a.now()))
        raise
