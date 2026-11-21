"""Uniform59 warm-start curriculum on the original fitter and frozen inputs."""
import argparse
import copy
import difflib
import inspect
from pathlib import Path
import run_smplifyx_forward_guard123 as forward

BODY=forward.BODY
PREVIOUS=BODY/'forward-guarded-sequence'
PROBE=BODY/'stage19-warmstart-original-tail-v2'
OUT=BODY/'warmtail-guarded-sequence'
PLAN=BODY/'warmtail-guarded-sequence-plan.json'
HANDOFF=BODY/'warmtail-guarded-sequence-gpu-handoff.json'
read,save,sha,now=forward.read,forward.save,forward.sha,forward.now
FIELDS=['body_pose_prior_weights','shape_weights','expr_weights','hand_pose_prior_weights',
        'jaw_pose_prior_weights','hand_joints_weights','face_joints_weights','coll_loss_weights','data_weights']
original_worker_source=forward.worker_source


def source():
    before,delta=original_worker_source()
    marker='        return c\n'
    insert="""        if active['seed'] is not None:
            if c.get('data_weights') is None:
                c['data_weights']=[1]*5  # Exact upstream implicit default.
            for field in ['body_pose_prior_weights','shape_weights','expr_weights','hand_pose_prior_weights','jaw_pose_prior_weights','hand_joints_weights','face_joints_weights','coll_loss_weights','data_weights']:
                assert len(c[field])==5
                c[field]=c[field][3:]
        return c
"""
    assert before.count(marker)==1
    changed=before.replace(marker,insert)
    compile(changed,'<upstream-warmstart-original-tail-configuration>','exec')
    delta+='\n'+''.join(difflib.unified_diff(before.splitlines(True),changed.splitlines(True),
        fromfile='chronological-original-five-stages',tofile='warmstarted-original-final-two-stages'))
    return changed,delta


def bind():
    forward.OUT,forward.PLAN,forward.HANDOFF=OUT,PLAN,HANDOFF
    forward.bind()


def prepare():
    assert not OUT.exists() and not PLAN.exists()
    old=read(PREVIOUS/'review/closeout.json');assert old['actual_reviewed_frames']==59 and not old['quality_accepted']
    result=read(PROBE/'result.json');assert result['unchanged_camera_shape_topology']
    assert read(PROBE/'review/comparison.json')['numeric_gate_passed']
    assert read(PROBE/'review/visual-review.json')['diagnostic_passed']
    p=copy.deepcopy(read(BODY/'forward-guarded-sequence-plan.json'))
    for path,digest in {**p['source_hashes'],**p['input_hashes']}.items():assert sha(path)==digest,path
    _,delta=source()
    p.update(updated=now(),state='prepared',output=str(OUT),source_adapter_delta=delta,
        hypothesis='The original cold-start prefix erased a valid neighbor pose. For every non-anchor warm-started frame, uniformly use original stages3/4. Keep the predeclared shape anchor33 at the original five stages.',
        single_change_from_failed_full59='Curriculum entry for an existing temporal pose: remove original stages0/1/2, retain all original stage3/4 values.',
        causal_scope='The entire cold-start curriculum prefix is omitted; not an isolated numeric-weight effect. Single-frame success does not establish sequence quality.',
        curriculum=dict(anchor_original_stages=[0,1,2,3,4],all_other_original_stages=[3,4],fields=FIELDS,
            original_defaults=dict(data_weights=[1]*5),no_numeric_weight_tuning=True),
        unchanged_from_previous=['all59 native observations/confidence','fixed K and independent camera initialization','shared beta from33',
            'chronological predecessor/boundary seed policy','original final objective and uniform cached fallback','LBFGS/maxiters30 and fresh optimizers',
            'zero hand/face initialization','original losses and retained-stage weight values'],
        preserved=['original solver source','same59 inputs and world contract','same59 evaluation/region protections','shape reference33'],
        difference_from_probe='All59 use the same conditional stage rule; temporal seeds come from this new pass and can propagate errors. No stored single-frame success is inserted.',
        original_solver_source_unchanged=True,local_curriculum_adaptation=True,
        runtime_budget_seconds=3000,expected_seconds=1500,
        risks=['Weakening cold-start stabilization can propagate an incorrect seed.','The original final objective can still prefer a wrong occluded limb.',
            'A failed whole-sequence gate rejects the candidate; no cherry-picking successful frames.'],
        no_automatic_followup=True,automatic_gaussian_training=False,quality_accepted=False,goal_complete=False)
    p['selection_policy']['candidate0']='Uniform original-tail warm-start candidate; anchor33 original five stages.'
    p['source_hashes'][str(Path(__file__).resolve())]=sha(__file__)
    for path in [PREVIOUS/'review/closeout.json',PROBE/'result.json',PROBE/'review/comparison.json',PROBE/'review/visual-review.json']:
        p['input_hashes'][str(path)]=sha(path)
    save(PLAN,p);bind();forward.temporal.verify_plan()
    print('PREPARED_UNIFORM59_ORIGINAL_WARMTAIL',flush=True)


def worker():
    bind()
    entry=forward.temporal.entry
    old_fit=entry.fit_one
    def fit(**kwargs):
        arrays,report=old_fit(**kwargs)
        anchor=Path(kwargs['result_path']).parent.name=='frame_000033'
        report.update(five_stages=anchor,original_stage_indices=list(range(5)) if anchor else [3,4],
            local_curriculum_adapter=not anchor,stage_values_unchanged=True)
        return arrays,report
    forward.worker_source=source;entry.fit_one=fit
    try:forward.worker()
    finally:entry.fit_one=old_fit;forward.worker_source=original_worker_source


def run(pid):
    bind()
    fn=forward.guarded.run;scope=dict(fn.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(fn),'<existing-exclusive-handoff-uniform59-warmtail>','exec'),scope)
    scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','run']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker()
