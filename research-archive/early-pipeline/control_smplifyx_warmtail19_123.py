"""One registered warm-start curriculum control using upstream final stages."""
import argparse
import inspect
from pathlib import Path
import numpy as np
import trace_smplifyx_stage19_123_v2 as diagnostic

b=diagnostic.base
TRACE=b.OUT
b.OUT=b.BODY/'stage19-warmstart-original-tail'
b.PLAN=b.BODY/'stage19-warmstart-original-tail-plan.json'
b.HANDOFF=b.BODY/'stage19-warmstart-original-tail-gpu-handoff.json'
FIELDS=['body_pose_prior_weights','shape_weights','expr_weights','hand_pose_prior_weights',
        'jaw_pose_prior_weights','hand_joints_weights','face_joints_weights','coll_loss_weights']


def worker_function():
    source=inspect.getsource(b.worker)
    replacements=[
        ("c['side_view_thsh']==25", "c.get('side_view_thsh',25)==25"),
        ("            c.update(_seed_pose_embedding=seed['pose_embedding']", "            for field in FIELDS:\n                assert len(c[field])==5\n                c[field]=c[field][3:]\n            c.update(_seed_pose_embedding=seed['pose_embedding']"),
        ("        assert report['orientation_count']==1 and len(stages)==6", "        assert report['orientation_count']==1 and len(stages)==3\n        report.update(five_stages=False, original_stage_indices=[3,4], local_curriculum_adapter=True)"),
        ("        assert all(parity[k]<=limit for k,limit in p['endpoint_parity'].items()),parity", "        assert all(parity[k]==0 for k in ['camera_translation','betas']),parity\n        assert np.array_equal(arrays['K_integer_centers'],expected['K_integer_centers'])\n        assert np.array_equal(arrays['faces'],expected['faces'])"),
        ("state='original_trajectory_complete_pending_visual_review'", "state='warmstart_original_tail_complete_pending_review'"),
        ("endpoint_parity_passed=True", "unchanged_camera_shape_topology=True, original_stage_indices=[3,4]"),
        ("no_new_training_candidate=True", "diagnostic_candidate_only=True, no_full59_admission=True")]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    scope=dict(b.worker.__globals__,FIELDS=FIELDS)
    exec(compile(source,'<upstream-two-stage-warmstart-configuration>','exec'),scope)
    return scope['worker']


def prepare():
    assert b.read(TRACE/'result.json')['endpoint_parity_passed']
    review=b.read(TRACE/'review/visual-review.json')
    assert review['actual_visual_review'] and review['reviewed_images']==10
    original=b.entry.load_official_config(b.arguments().official_root)
    for field in FIELDS:assert len(original[field])==5
    worker_function();b.prepare();p=b.read(b.PLAN)
    p.update(hypothesis='The original high-prior cold-start prefix erases a supplied neighbor pose. Start from the exact same frame18 body/root and execute only the existing original final two refinement stages. Test final elbow and protections; do not select an intermediate state.',
        single_policy_change='Original curriculum entry index0 -> index3 for this already-seeded diagnostic. No original-stage value changed.',
        causal_scope='Omitting the first three stages changes their complete curriculum, not just one numeric weight. This does not establish a unique hand, camera or observation cause.',
        body_stages=2,original_stage_indices=[3,4],runtime_budget_seconds=300,
        configuration_difference={field:dict(before=original[field],after=original[field][3:]) for field in FIELDS},
        unchanged=['exact frame18 seed', 'camera initialization/complete K', 'frozen beta', 'observations/confidence',
            'original stage3 and4 loss values and terms', 'original LBFGS/stopping/maxiters30', 'zero hand/face initialization'],
        endpoint_parity={'vertices_camera':2e-6,'joints_projected_native':.002,'pose_embedding':2e-6,
            'global_orient':2e-6,'camera_translation':0,'betas':0},
        interpretation_gate='The candidate is expected to differ. Camera translation, K, beta and topology must stay exact. The separate trace has already replayed baseline exactly.',
        criteria=['Final frame19 left elbow <= half the original candidate84.93596px.',
            'Body/feet median residual increase<=1px and face median increase<=0.5px using the same frozen observations.',
            'Actual same-crop body/face/hands/feet inspection shows no new collapse.',
            'Do not adopt this one frame as a full-sequence replacement; a uniform all59 candidate requires separate review.'],
        risk='Removing the early cold-start stabilization may worsen face/hand fitting or local minima.',
        no_automatic_followup=True,quality_accepted=False,goal_complete=False)
    p['source_hashes'][str(Path(__file__).resolve())]=b.sha(__file__)
    p['source_hashes'][str(Path(diagnostic.__file__).resolve())]=b.sha(diagnostic.__file__)
    for file in [TRACE/'result.json',TRACE/'review/analysis.json',TRACE/'review/visual-review.json']:
        p['input_hashes'][str(file)]=b.sha(file)
    b.save(b.PLAN,p);print('PREPARED_ORIGINAL_WARMSTART_TAIL19',flush=True)


def run(pid):
    source=inspect.getsource(b.run);scope=dict(b.run.__globals__,__file__=__file__)
    exec(compile(source,'<existing-gpu-handoff-warmstart-control>','exec'),scope);scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker_function()()
