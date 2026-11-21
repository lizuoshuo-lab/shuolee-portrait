"""Sixteen-view input adaptation; reuse the completed indoor SH0 training code."""
import ast
import copy
import difflib
import inspect
import os
from pathlib import Path
import signal
import sys
import traceback
import train_indoor_selected_sh0124 as native
import train_indoor_person_zoom124 as zoom
import cross_subject_sh0124 as gs

SUBJECT = 'indoor-circle16-paired18k124'
OUT = gs.ROOT/SUBJECT
FIT = gs.ROOT/'indoor-circle16-pose124'
HANDOFF = FIT/'gaussian-native-handoff'
BASELINE = gs.ROOT/'indoor-paired-background18k124'


def replace(source, old, new):
    assert source.count(old) == 1, (old, source.count(old))
    return source.replace(old, new)


def reused(function, edits, scope, label):
    before = inspect.getsource(function)
    after = before
    for old, new in edits:
        after = replace(after, old, new)
    ast.parse(after)
    ns = dict(function.__globals__, **scope)
    exec(compile(after, str(Path(__file__))+':'+label, 'exec'), ns)
    return ns[function.__name__], ''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='existing-interface', tofile=label))


def prepare():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists() and not HANDOFF.exists()
    review = gs.read(FIT/'conclusion.json')
    assert review['actual_visual_review'] and review['exploratory_gaussian_admitted']
    edits = [
        ("'assert n == 6'", "'assert n == 16'"),
        ("室内人物6帧 SH0 可见区域试跑", "室内人物16帧 SH0 可见区域试跑"),
        ("'assert len(training_slots)==6'", "'assert len(training_slots)==16'"),
        ("'training_frames=6'", "'training_frames=16'"),
        ('six-frame-interface.diff', 'sixteen-frame-interface.diff'),
        ('Six preregistered indoor inputs', 'Sixteen preregistered indoor inputs'),
        ('All six preregistered', 'All sixteen preregistered'),
        ('分析_室内可见区域124_', '分析_室内16视角124_'),
        ('for start in (0, 3):', 'for start in range(0, len(tiles), 3):'),
    ]
    fn, diff = reused(native.prepare, edits, dict(OUT=HANDOFF, SOURCE=FIT,
        SUBJECT=SUBJECT, REFERENCE='frame_000105', __file__=__file__), 'sixteen-native-handoff')
    fn()
    (HANDOFF/'sixteen-input.diff').write_text(diff)
    edits = [
        ("assert gs.read(BASE/'gaussian-run/result.json')['state']=='bounded_control_completed'",
         "assert gs.read(BASE/'smplify-gaussian-interface-receipt.json')['frames']==16"),
        ('assert len(source_rows)==6', 'assert len(source_rows)==16'),
        ("sheet=Image.new('RGB',(1800,1380),(35,35,35))", "sheet=Image.new('RGB',(1800,460*((len(panels)+1)//2)),(35,35,35))"),
        ('all6_projection_checks=True', 'all16_projection_checks=True'),
    ]
    fn, diff = reused(zoom.prepare, edits, dict(BASE=HANDOFF, OUT=OUT,
        SUBJECT=SUBJECT, LIMIT=18000, __file__=__file__), 'sixteen-existing-roi')
    fn()
    (OUT/'sixteen-roi-interface.diff').write_text(diff)
    original = (BASELINE/'adapter.py').read_text()
    code = original
    edits = [
        ("receipt['frames'] == 6", "receipt['frames'] == 16"),
        ('self.num_timesteps = 6', 'self.num_timesteps = 16'),
        ('assert len(training_slots)==6', 'assert len(training_slots)==16'),
        ('counts = np.zeros(6, dtype=int)', 'counts = np.zeros(16, dtype=int)'),
        ('frames_rendered=6', 'frames_rendered=16'),
        ('training_frames=6, review_frames=6', 'training_frames=16, review_frames=16'),
        ('Same indoor ROI2x and18k clock; paired stochastic background; reviewed1000 and hard stop18000',
         'Sixteen-view input; same ROI2x/SH0/18k clock/paired background; reviewed1000 and hard stop18000'),
        ('Only paired stochastic training background relative to indoor zoom clock; original RGB loss and all options unchanged',
         'Sixteen-view newly fitted input; all Gaussian options equal completed six-view paired-background18k'),
    ]
    for old, new in edits:
        code = replace(code, old, new)
    ast.parse(code)
    (OUT/'adapter.py').write_text(code)
    (OUT/'roster-only-training.diff').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), code.splitlines(True), fromfile='six-view-completed', tofile='sixteen-view-fresh')))
    old = gs.read(BASELINE/'plan.json')
    p = gs.read(OUT/'plan.json')
    for key in ('options', 'upstream_options', 'schedule_overrides', 'shared_laplacian_weight',
                'goal_points_approx', 'goal_point_band', 'max_points_safety_stop'):
        p[key] = copy.deepcopy(old[key])
    p.update(updated=gs.now(), state='prepared_input_review_pending', subject=SUBJECT,
        label='室内人物 · 一圈16帧 SH0', frames=16, training_frames=16, evaluation_frames=16,
        baseline_run=str(BASELINE), source_fit=str(FIT/'selected-fit.npz'), reference_frame='frame_000105',
        user_authorization='半圈8，一圈16；先训练高斯检查完整流程。',
        only_parameter_change='Input roster and associated fresh shared SMPLify-X fit:6->16. Gaussian optimization and image-domain rules unchanged.',
        hypothesis='Broader observed directions reduce sparse-view holes; pose/mask inconsistency may instead worsen visible surfaces.',
        causal_scope='Combined selected-input and newly fitted geometry effect, not a pure Gaussian view-count ablation.',
        fixed='SH0;ROI2x;original loss/learning-rate amplitudes;18k time profile;paired background;face/head offset protection;reference105.',
        comparison_budget=18000, configured_schedule_steps=18000, snapshots=[1000,10000,18000],
        new_pose_fitting=True, source_fitted_pose_reused_for_control=False, from_scratch=True,
        optimizer_state_reused=False, gaussian_parameters_reused=False,
        criteria='Review all16source views at1000; no new gross visible limb/face collapse or relative folds. Continue same live optimizer to18000 only after actual review. Final all16 plus continuous novel views and paired native export.',
        budget_note='One candidate maximum18000steps and2400seconds including review. No seed/threshold sweep or automatic retry.',
        known_limitation='Fitted directional gaps improve but pose/mask/conditional camera errors remain. Legs behind furniture unknown.',
        quality_accepted=False, model_promoted=False, end_to_end_15min_accepted=False,
        no_automatic_new_experiment=True)
    # Refresh only newly created files; frozen upstream inputs retain their hashes.
    for path in list(p['source_hashes']):
        if Path(path).is_relative_to(OUT):
            p['source_hashes'][path] = gs.sha(path)
    for file in (Path(__file__), Path(native.__file__), Path(zoom.__file__),
                 BASELINE/'adapter.py', BASELINE/'plan.json', FIT/'conclusion.json',
                 gs.HERE/'paired_training_background124.py', OUT/'roster-only-training.diff'):
        p['source_hashes'][str(file)] = gs.sha(file)
    gs.save(OUT/'plan.json', p)
    gs.save(OUT/'validity-review.json', dict(updated=gs.now(), actual_visual_review=False,
        training_admitted=False, source_review=str(HANDOFF/'validity-review.json'),
        records=gs.read(HANDOFF/'validity-review.json')['records'],
        rule=gs.read(HANDOFF/'validity-review.json')['rule'],
        source_mask_unchanged=True, quality_accepted=False))
    gs.save(OUT/'progress.json', dict(updated=gs.now(), state=p['state'], step=0, pid=None))
    print('SIXTEEN_INPUT_PREPARED', flush=True)


def preflight():
    edits = [
        ("dataset=ctx['Images']();results=[];torch.cuda.reset_peak_memory_stats()",
         "dataset=ctx['Images']();results=[];torch.cuda.reset_peak_memory_stats()\n"
         "    from paired_training_background124 import PairedTrainingBackground\n"
         "    paired=PairedTrainingBackground(ctx['render'],ctx['masks'],[r['frame'] for r in ctx['rows']],ctx['target_cache'])"),
        ("output=ctx['render'](cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image",
         "output=paired(cam,model,ctx['pipe'],ctx['background'])['render'];target=cam.original_image"),
        ('assert not model.optimizer.state', 'assert paired.calls==16\n    assert not model.optimizer.state'),
    ]
    fn, diff = reused(zoom.preflight, edits, dict(OUT=OUT, LIMIT=18000), 'sixteen-paired-gpu-preflight')
    (OUT/'preflight-interface.diff').write_text(diff)
    fn('sixteen-preflight')


def train():
    assert not (OUT/'launch.json').exists() and not (OUT/'gaussian-run').exists()
    assert gs.read(OUT/'validity-cuda-preflight.json')['passed']
    assert gs.read(OUT/'validity-review.json')['training_admitted']
    def timeout(*unused):
        raise TimeoutError('Registered2400second budget exhausted; no automatic retry')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(2400)
    try:
        gs.train(SUBJECT)
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    mode = sys.argv[1]
    try:
        {'prepare':prepare, 'preflight':preflight, 'train':train}[mode]()
        if mode == 'train':
            gs.save(OUT/'process-exit.json', dict(updated=gs.now(), exit_code=0))
    except BaseException as error:
        if OUT.exists():
            gs.save(OUT/(mode+'-failure.json'), dict(updated=gs.now(), error=repr(error),
                traceback=traceback.format_exc(), auto_retry=False))
        raise
