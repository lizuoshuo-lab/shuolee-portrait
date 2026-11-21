"""One chronological initialization pass using the unchanged original fitter.

The shape reference remains frame33. Frame1 uses the frozen independent frame2
only as an edge seed; it never defines shape, calibration, or a new zero-pose
sequence anchor. Original-objective selection and cache disclosure are retained.
"""
import argparse
import difflib
import inspect
from pathlib import Path
import numpy as np
import run_smplifyx_guarded_seed123 as guarded

temporal = guarded.temporal
BODY, BASE = guarded.BODY, guarded.BASE
OUT = BODY/'forward-guarded-sequence'
PLAN = BODY/'forward-guarded-sequence-plan.json'
HANDOFF = BODY/'forward-guarded-sequence-gpu-handoff.json'
PREVIOUS = BODY/'guarded-root-sequence'
read, save, sha, now = guarded.read, guarded.save, guarded.sha, guarded.now


def bind():
    guarded.OUT, guarded.PLAN, guarded.HANDOFF = OUT, PLAN, HANDOFF
    guarded.bind()


def worker_source():
    original = inspect.getsource(temporal.worker)
    before = "            assert item['seed'] in completed\n            seed_path=OUT/'frames'/item['seed']/'winner-camera-space.npz'"
    after = ("            is_boundary = item['seed_source'] == 'frozen_independent_boundary'\n"
             "            assert is_boundary or item['seed'] in completed\n"
             "            assert not is_boundary or (frame == p['temporal_roster'][0] and not completed)\n"
             "            seed_path=(BASE if is_boundary else OUT)/'frames'/item['seed']/'winner-camera-space.npz'")
    assert original.count(before) == 1
    changed = original.replace(before, after)
    old = 'Original anchor zero/mean, then within-run adjacent body/root start after unchanged camera initialization'
    new = 'Fixed shape reference33; chronological body/root start, frozen frame2 edge seed; unchanged camera initialization'
    assert changed.count(old) == 1
    changed = changed.replace(old, new)
    compile(changed, '<original-fitter-chronological-seed-path>', 'exec')
    delta = ''.join(difflib.unified_diff(original.splitlines(True), changed.splitlines(True),
        fromfile='verified-guarded-sequence-adapter', tofile='chronological-seed-boundary-adapter'))
    return changed, delta


def prepare():
    assert not OUT.exists() and not PLAN.exists()
    old = read(PREVIOUS/'review/closeout.json')
    assert old['actual_reviewed_frames'] == 59 and not old['quality_accepted']
    path_evidence = read(PREVIOUS/'review/initialization-path-evidence.json')
    assert path_evidence['camera_initialization_exact']
    assert path_evidence['original_matched_probe']['original_objective'] < path_evidence['current_guarded']['original_objective']
    bind()
    guarded.prepare()
    p = read(PLAN)
    roster, anchor = p['temporal_roster'], p['anchor']
    assert anchor == 'frame_000033' and roster[0] == 'frame_000001'
    schedule = []
    for i, frame in enumerate(roster):
        if frame == anchor:
            schedule.append(dict(frame=frame, seed=None, direction='fixed_shape_reference', seed_source='original_zero_mean_reference'))
        elif i == 0:
            schedule.append(dict(frame=frame, seed=roster[1], direction='boundary', seed_source='frozen_independent_boundary'))
        else:
            schedule.append(dict(frame=frame, seed=roster[i-1], direction='forward', seed_source='within_run_selected_predecessor'))
    boundary = BASE/'frames'/roster[1]/'winner-camera-space.npz'
    with np.load(boundary, allow_pickle=False) as s, np.load(BASE/'frames'/roster[0]/'winner-camera-space.npz', allow_pickle=False) as t:
        assert str(s['frame_id'].item()) == roster[1]
        assert s['pose_embedding'].shape == (1,32) and s['global_orient'].shape == (1,3)
        for key in ['K_integer_centers','betas','faces','w2c']:
            assert np.array_equal(s[key],t[key]), key
        assert np.isfinite(s['pose_embedding']).all() and np.isfinite(s['global_orient']).all()
    source, delta = worker_source()
    p.update(updated=now(), processing_order=roster, schedule=schedule,
        hypothesis='A fixed chronological predecessor path may carry the existing good frame17 arm state into18..23, which the previously tested33-to1 path did not reach. Every frame still uses the original per-frame solve and original-objective guard.',
        single_change_from_failed_full59='Initialization traversal policy: chronological pass with a fixed frozen-neighbor boundary seed; shape reference33 unchanged.',
        causal_scope='Traversal direction and the required boundary seed form one initialization-policy intervention. Do not claim a direction-only causal effect; no loss, camera, shape, optimizer or iteration change.',
        difference_from_probe='All59 without selecting a successful frame by inspection; every non-reference interior frame uses its actual within-run selected predecessor.',
        anchor_policy='Shape reference33 and its original zero/mean fit remain fixed. Frame1 is initialized from predeclared independent frame2; no first-frame shape/calibration initialization.',
        boundary_seed=dict(target=roster[0],seed=roster[1],path=str(boundary),sha256=sha(boundary),selection='Fixed temporal boundary, no quality ranking.'),
        previous_closeout_sha256=sha(PREVIOUS/'review/closeout.json'),
        source_adapter_delta=delta,
        unchanged_from_previous=['all59 observations/confidence and fixed split','complete K','independent camera initialization','shared betas from33','original five stages/weights/LBFGS/maxiters30','fresh optimizers','zero hands/expression initialization','original-objective candidate selection and independent cache'],
        risks=['Chronological seeds may enter another incorrect local solution.',
            'Original-objective selection can still trade off regional residuals and fail anatomy.',
            'The boundary source is an old independent fit; cache construction is mandatory in a fresh-video runtime accounting.'],
        no_automatic_followup=True)
    p['selection_policy']['candidate0'] = 'new chronological predecessor body/root fit; fixed frame2 boundary seed'
    p['cache_policy']['purpose'] = 'Uniform same-frame fallback plus predeclared frame2 edge initialization; not an end-to-end speed result.'
    p['criteria'][0] = 'Reference33 must replay the original fit; all59 frames preserve K, independent camera initialization, beta, observations and topology; every actual seed must match the registered chronological policy.'
    p['source_hashes'][str(Path(__file__).resolve())] = sha(__file__)
    p['source_hashes'][str(Path(guarded.__file__).resolve())] = sha(guarded.__file__)
    p['input_hashes'][str(PREVIOUS/'review/closeout.json')] = sha(PREVIOUS/'review/closeout.json')
    p['input_hashes'][str(PREVIOUS/'review/initialization-path-evidence.json')] = sha(PREVIOUS/'review/initialization-path-evidence.json')
    p['input_hashes'][str(boundary)] = sha(boundary)
    save(PLAN, p)
    temporal.verify_plan()
    print('PREPARED_CHRONOLOGICAL_FULL59_REFERENCE33', flush=True)


def worker():
    bind()
    p = temporal.verify_plan()
    source, delta = worker_source()
    assert delta == p['source_adapter_delta']
    previous = temporal.worker
    scope = dict(previous.__globals__)
    exec(compile(source, '<original-fitter-chronological-seed-path>', 'exec'), scope)
    temporal.worker = scope['worker']
    try:
        guarded.worker()
    finally:
        temporal.worker = previous


def run(pid):
    bind()
    # Reuse the already verified exclusive GPU lease and finally-resume logic.
    source = inspect.getsource(guarded.run)
    scope = dict(guarded.run.__globals__, __file__=__file__)
    exec(compile(source, '<existing-supervisor-new-entrypoint>', 'exec'), scope)
    scope['run'](pid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','run','worker'])
    parser.add_argument('--pid', type=int)
    args = parser.parse_args()
    if args.mode == 'prepare': prepare()
    elif args.mode == 'run': run(args.pid)
    else: worker()
