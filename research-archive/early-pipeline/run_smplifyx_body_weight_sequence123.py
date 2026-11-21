"""Uniform body-data weight control on the existing chronological original fit.

Keep the fixed shape anchor and original-score fallback. Do not splice the
successful diagnostic frame or apply the rejected regional fallback policy.
"""
import argparse
import copy
import inspect
from pathlib import Path
import run_smplifyx_warmtail_sequence123 as warm
import control_smplifyx_body_weight123_v2 as single

BODY=warm.BODY
PREVIOUS=BODY/'warmtail-guarded-sequence'
OUT=BODY/'body-weight-guarded-sequence'
PLAN=BODY/'body-weight-guarded-sequence-plan.json'
HANDOFF=BODY/'body-weight-guarded-sequence-gpu-handoff.json'
read,save,sha,now=warm.read,warm.save,warm.sha,warm.now


def bind():
    warm.OUT,warm.PLAN,warm.HANDOFF=OUT,PLAN,HANDOFF
    warm.bind()


def prepare():
    assert not OUT.exists() and not PLAN.exists()
    diagnostic=BODY/'worst-region-body-weight-control-v2'
    reviewed=read(diagnostic/'review/visual-review.json')
    assert reviewed['diagnostic_passed'] and reviewed['reviewed_images']==5
    p=copy.deepcopy(read(BODY/'warmtail-guarded-sequence-plan.json'))
    for path,digest in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(path)==digest,path
    p.update(updated=now(),state='prepared',output=str(OUT),
        hypothesis='Uniformly match body/feet observation amplitude to2 for warm-started refinement, to reduce the original late-stage tradeoff against hand/face data. Keep the same original-score selection, not regional frame splicing.',
        single_change_from_failed_full59='Non-anchor body/feet input amplitude1->2 (squared image-data contribution4x).',
        anchor_policy='Existing shape anchor33 retains the original five-stage amplitude1 result; all other58 frames use the same new amplitude2.',
        body_observation_amplitude=dict(anchor=1.,all_other_frames=2.,prior_value=1.),
        score_contract='For candidate selection only, normalize the captured body/feet amplitudes back to1 and replay the original final objective for both saved candidates. The fit objective uses2. Never compare differently weighted totals.',
        causal_scope='One uniform fitter input-weight intervention. Within-run seeds may change as a consequence; the successful single-frame result is not inserted.',
        unchanged_from_previous=['all59 native observations and confidence','fixedK and independent camera initialization','frozen shared beta and anchor33','original stage3/4 schedules, priors, LBFGS and stopping/maxiters30','chronological seed and edge policy','original unmodified-score fallback to independent cache','zero hand/face initialization'],
        risks=['Hand/face detail can regress.','Upstream original-score fallback can still prefer worse anatomy.','Shoe/occluded-hand targets remain imperfect physical evidence.'],
        runtime_budget_seconds=3000,expected_seconds=1800,
        no_automatic_followup=True,automatic_gaussian_training=False,quality_accepted=False,goal_complete=False)
    # The existing worker adapter is unchanged; the extra weight wrapper is
    # isolated here and the original solver files are never edited.
    p['source_adapter_delta']=warm.source()[1]
    for path in [Path(__file__),Path(single.__file__),Path(single.control.__file__),Path(single.control.trace.__file__)]:
        p['source_hashes'][str(path.resolve())]=sha(path)
    for path in [PREVIOUS/'review/closeout.json',PREVIOUS/'review/region-gate-visual-review.json',PREVIOUS/'review/data-term-audit.json',
                 diagnostic/'result.json',diagnostic/'review/comparison.json',diagnostic/'review/visual-review.json']:
        p['input_hashes'][str(path)]=sha(path)
    p['selection_policy']['candidate0']='New uniform body-amplitude2 chronological fit (anchor33 stays original), scored with original amplitude1.'
    save(PLAN,p);bind();warm.forward.temporal.verify_plan()
    print('PREPARED_ALL59_BODY_AMPLITUDE_CONTROL',flush=True)


def worker():
    bind();entry=warm.forward.temporal.entry
    plan=warm.forward.temporal.verify_plan();original=entry.fit_one
    weighted=single.live_config_fit()
    def fit(**kwargs):
        frame=Path(kwargs['result_path']).parent.name
        if frame==plan['anchor']:
            arrays,report=original(**kwargs)
            report.update(body_observation_amplitude=1.,anchor_unchanged=True)
            return arrays,report
        arrays,report=weighted(**kwargs)
        context=entry._guard_context
        weights=context['joint_weights'].clone()
        assert bool((weights[:,[1,9,12]]==0).all())
        active=[i for i in range(25) if i not in (1,9,12)]
        assert bool((weights[:,active]==2).all())
        weights[:,:25]/=2
        context['joint_weights']=weights
        report.update(body_observation_amplitude=2.,original_body_observation_amplitude=1.,
            fit_weighted_objective=report['winning_loss'],selection_uses_original_body_amplitude=True,
            original_body_numeric_weights_unchanged=False,hand_face_stage_values_unchanged=True)
        return arrays,report
    entry.fit_one=fit
    try:warm.worker()
    finally:entry.fit_one=original


def run(pid):
    bind();fn=warm.forward.guarded.run
    scope=dict(fn.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(fn),'<existing-exclusive-all59-body-weight-handoff>','exec'),scope)
    scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker()
