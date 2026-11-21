"""One matched SMPLify-X observation-visibility control on verified source views."""
from datetime import datetime, timezone
from importlib.util import find_spec
import inspect
import os
from pathlib import Path
import sys
import time
import types

import probe_video21_shoe_fit165 as baseline
from extract_video21_shoe_depth162 import replace
from update_leafwoman129_state import atomic, read, sha


OUT = baseline.BASE.OUT / 'right-elbow-visibility168'
VISIBILITY = baseline.BASE.OUT / 'shoe-depth-xy167/review/upper-limb-source'


def fit_shard(folder, shard):
    code = inspect.getsource(baseline.fit_shard)
    code = replace(code, '        arrays,receipt=solve(saved_seed=seed,replay_only=False,**inputs)',
        '''        # Replay kept every original observation. Exclude only the source-reviewed elbow in this control.
        assert not use_depth and frame in plan['occluded_fit_frames']
        observations = inputs['keypoints'].copy()
        elbow_confidence = float(observations[0, 3, 2])
        assert elbow_confidence > 0
        observations[0, 3, 2] = 0
        candidate = dict(inputs, keypoints=observations)
        arrays,receipt=solve(saved_seed=seed,replay_only=False,**candidate)
        receipt.update(excluded_coco25_index=3, original_elbow_confidence=elbow_confidence,
                       effective_elbow_confidence=0., excluded_joint='right_elbow')''')
    scope = dict(baseline.__dict__, __file__=__file__)
    exec(compile(code, '<native-occluded-elbow-observation>', 'exec'), scope)
    scope['fit_shard'](folder, shard)


def main():
    assert os.name == 'nt' and not OUT.exists()
    if find_spec('omegaconf') is None:
        raise RuntimeError('Run in the existing VPoser environment.')
    source_plan, source_result = (baseline.OUT / name for name in ('plan.json', 'result.json'))
    plan, result = read(source_plan), read(source_result)
    control = baseline.OUT / 'control/result.json'
    assert result['completed'] and result['plan_sha256'] == sha(source_plan)
    assert result['branch_result_sha256']['control'] == sha(control)
    assert all(sha(path) == digest for path, digest in plan['source_hashes'].items())
    assert baseline.entry.load_official_config(baseline.BASE.original.OFFICIAL)['use_joints_conf'] is True
    review = read(VISIBILITY / 'visual-decision.json')
    source = read(VISIBILITY / 'result.json')
    assert review['actual_visual_review'] and review['source_result_sha256'] == sha(VISIBILITY / 'result.json')
    assert source['source_only'] and not source['landmarks_or_fit_rendered']
    assert all(sha(VISIBILITY / page['name']) == page['sha256'] for page in source['images'])
    frames = [f for f in result['frames'] if f in review['right_elbow_not_independently_visible']]
    assert len(frames) == 3 and not set(frames) & set(plan['retained_holdouts'])
    inputs = [Path(__file__), Path(baseline.__file__), Path(inspect.getsourcefile(replace)),
              source_plan, source_result, control, VISIBILITY / 'visual-decision.json',
              VISIBILITY / 'result.json', *(VISIBILITY / row['name'] for row in source['images'])]
    registered = dict(plan, updated=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(__file__),
        frames=frames, occluded_fit_frames=frames, branches=['elbow_occluded'],
        source_hashes=plan['source_hashes'] | {str(p): sha(p) for p in inputs},
        hypothesis='The detector gives positive weight to a right elbow hidden by the armrest and can pull the arm away from visible limb evidence.',
        single_change='Zero the COCO25 right-elbow confidence in the existing objective for the matched source-reviewed frames only. Keep raw observations in the archive.',
        fixed='165 control with no shoe terms. Same full K, shared shape, camera/root initial state, raw source landmarks, final stages, anchor weight, optimizer and all other observation weights.',
        criteria='Compare visible right shoulder and wrist as a pair, source/side arm fit, face and feet against the frozen control. Record the original and visibility-aware gate diagnostics separately. This three-frame result grants no Gaussian admission.',
        gaussian_started=False, quality_accepted=False)
    atomic(OUT / 'plan.json', registered)
    atomic(OUT / 'launch.json', dict(pid=os.getpid(), command=[sys.executable, *sys.argv], frames=frames))
    scope = dict(baseline.BASE.__dict__, fit_shard=fit_shard)
    stage = types.FunctionType(baseline.BASE.stage.__code__, scope)
    indices = [plan['source_frames'].index(frame) for frame in frames]
    started = time.monotonic()
    try:
        rows = stage(OUT.name + '/control', registered, indices, plan['shared_shape'])
        assert all(r['excluded_coco25_index'] == 3 and r['effective_elbow_confidence'] == 0.
                   and not r['depth_term_enabled'] for r in rows)
        assert all(sha(path) == digest for path, digest in registered['source_hashes'].items())
        atomic(OUT / 'result.json', dict(completed=True, frames=frames, plan_sha256=sha(OUT / 'plan.json'),
            fit_result_sha256=sha(OUT / 'control/result.json'), seconds=time.monotonic()-started,
            gaussian_started=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
