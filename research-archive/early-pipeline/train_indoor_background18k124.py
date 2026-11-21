"""One registered long paired-background control; no checkpoint continuation."""
import ast
import copy
import difflib
from pathlib import Path
import shutil
import signal
import sys
import traceback
import cross_subject_sh0124 as gs

SUBJECT = 'indoor-paired-background18k124'
BASE = gs.ROOT/'indoor-paired-background124'
BLACK = gs.ROOT/'indoor-zoom-clock18k124'
OUT = gs.ROOT/SUBJECT


def prepare():
    assert not OUT.exists(), 'One candidate only'
    baseline = gs.read(BASE/'plan.json')
    assert gs.read(BASE/'process-exit.json')['exit_code'] == 0
    assert gs.read(BASE/'gaussian-run/step-001000/visual-review.json')['actual_visual_review']
    assert gs.read(BASE/'matched-orbit/visual-review.json')['actual_visual_review']
    assert gs.read(BASE/'background-preflight.json')['passed']
    for path, digest in baseline['source_hashes'].items():
        assert gs.sha(path) == digest, path
    original = (BASE/'adapter.py').read_text()
    changes = {
        '    LIMIT = 1000\n    SNAPSHOTS = [1000]\n':
        '    LIMIT = 18000\n    SNAPSHOTS = [1000,10000,18000]\n',
        'Same indoor ROI2x and18k clock; only matched stochastic background; hard stop1000':
        'Same indoor ROI2x and18k clock; paired stochastic background; reviewed1000 and hard stop18000',
    }
    source = original
    for old, new in changes.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    ast.parse(source)
    names = ['fit/fitted.npz', 'fit/surface-cache.npz', 'data/training-inputs.json',
        'smplify-gaussian-interface-receipt.json', 'selection.json', 'selection-rule.json',
        'source-native-inputs.json', 'pixel-mapping.json', 'loss_utils_validity.py',
        'validity-review.json', 'validity-cuda-preflight.json']
    for folder in ('fit', 'data'):
        (OUT/folder).mkdir(parents=True)
    for name in names:
        shutil.copyfile(BASE/name, OUT/name)
        assert gs.sha(BASE/name) == gs.sha(OUT/name)
    (OUT/'adapter.py').write_text(source)
    (OUT/'budget-only.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True), fromfile='completed1000-control',
        tofile='separate-from-scratch18k-control')))
    plan = copy.deepcopy(baseline)
    plan.update(updated=gs.now(), state='registered_ready', subject=SUBJECT,
        label='室内人物 · 放大2倍＋配对随机背景18k',
        baseline_run=str(BLACK), early_control_run=str(BASE),
        comparison_budget=18000, configured_schedule_steps=18000, snapshots=[1000,10000,18000],
        baseline_model_sh_degree=0,
        hypothesis='Paired random backgrounds reduce late opaque-black camouflage seen in the fixed-black18k control; may still fail real surface continuity.',
        causal_scope='One background variable versus the same ROI2x fixed-black18k candidate. Reuse identical selected fit and masks only to isolate this Gaussian-stage control.',
        only_parameter_change='Fixed black training background -> seeded uniform RGB, applied identically to renderer and target composite.',
        renderer_changed=False, renderer_wrapper_changed=True,
        fixed='Same6frames/reference105/ROI2x/fullK/SH0/18kclock/learning rates/losses;20908 fresh Gaussians and empty optimizer; no pose or frame reselection.',
        criteria='At1000 inspect all6 in native source pixel scale. Stop for new gross visible body/face collapse, folds, face motion, nonfinite gradients, leakage or export mismatch. Continue this live optimizer only after actual review. At18000 compare identical6source views and4fixed+36novel views, RGB and alpha separately. No automatic retry, seed scan or further training.',
        primary_evidence='Opaque dark area at the four frozen novel cameras, with unchanged alpha compositing checks; continuous-view actual visual review. Colour threshold only descriptive, not pruning or truth.',
        protection='Face/head offsets frozen; skeleton frozen; existing relative-normal/area/20mm guards; occluded lower regions remain unknown. Source-face/body colour error reported even if it worsens.',
        known_limitation='Early1000 random-background control worsened known foreground colour MAE17.4%; cannot promote without late same-camera and novel-view review. This uses cached pose/inputs, not a15minute cold pipeline.',
        budget_note='One separate18000-step maximum candidate, actual1000-step review, wall budget2400seconds including review; no resumption of prior1000 optimizer.',
        review_gate='Actual1000 review; same live optimizer may continue to18000 only without new severe deterioration.',
        preregistered_at=gs.now(), from_scratch=True, old_gaussian_checkpoint_loaded=False,
        optimizer_state_reused=False, gaussian_parameters_reused=False,
        no_automatic_new_experiment=True, quality_accepted=False, model_promoted=False,
        end_to_end_15min_accepted=False)
    paths = [Path(__file__), OUT/'adapter.py', BASE/'plan.json',
        BASE/'background-preflight.json', BASE/'gaussian-run/step-001000/visual-review.json',
        BASE/'matched-orbit/visual-review.json', BLACK/'gaussian-run/step-018000/result.json']
    paths += [OUT/name for name in names]
    plan['source_hashes'].update({str(p):gs.sha(p) for p in paths})
    plan['preflight_reuse'] = dict(source=str(BASE/'background-preflight.json'),
        sha256=gs.sha(BASE/'background-preflight.json'), identical_background_code=True,
        repeated_gpu_preflight=False, reason='Only bounded runner save/stop times differ from tested1000 process.')
    gs.save(OUT/'plan.json', plan)
    gs.save(OUT/'progress.json', dict(updated=gs.now(), state='registered_ready', step=0,
        pid=None, budget=18000, quality_accepted=False))
    print('BACKGROUND18K_REGISTERED', flush=True)


if __name__ == '__main__':
    try:
        if sys.argv[1] == 'prepare':
            prepare()
        elif sys.argv[1] == 'train':
            assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
            def timeout(*unused):
                raise TimeoutError('Registered2400second budget exhausted; no auto restart')
            signal.signal(signal.SIGALRM, timeout)
            signal.alarm(2400)
            try:
                gs.train(SUBJECT)
            finally:
                signal.alarm(0)
        else:
            raise ValueError('prepare or train')
    except BaseException as error:
        if OUT.exists():
            gs.save(OUT/(sys.argv[1]+'-failure.json'), dict(updated=gs.now(),
                error=repr(error), traceback=traceback.format_exc(), auto_retry=False))
        raise
