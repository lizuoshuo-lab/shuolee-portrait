"""Observe the largest registered foot regression; no fitter changes.

Selection is deterministic across the whole roster, not a hand-picked training
replacement. Reuses the existing original-monitor observer and exclusive lease.
"""
import argparse
import inspect
from pathlib import Path
import numpy as np
import trace_smplifyx_stage19_123 as b

PREVIOUS=b.BODY/'warmtail-guarded-sequence'
b.OUT=b.BODY/'worst-region-original-trajectory'
b.PLAN=b.BODY/'worst-region-original-trajectory-plan.json'
b.HANDOFF=b.BODY/'worst-region-original-trajectory-gpu-handoff.json'
FIELDS=['body_pose_prior_weights','shape_weights','expr_weights','hand_pose_prior_weights',
        'jaw_pose_prior_weights','hand_joints_weights','face_joints_weights','coll_loss_weights','data_weights']


def selected_frame():
    rows=b.read(PREVIOUS/'review/criteria-audit.json')['rows']
    return max(rows,key=lambda r:(r['protections']['feet6']['delta_px'],r['frame']))['frame']


b.FRAME=selected_frame()


def worker_function():
    source=inspect.getsource(b.worker)
    replacements=[
        ("c['side_view_thsh']==25", "c.get('side_view_thsh',25)==25"),
        ("            c.update(_seed_pose_embedding=seed['pose_embedding']",
         "            if c.get('data_weights') is None: c['data_weights']=[1]*5\n            for field in FIELDS:\n                assert len(c[field])==5\n                c[field]=c[field][3:]\n            c.update(_seed_pose_embedding=seed['pose_embedding']"),
        ("        assert report['orientation_count']==1 and len(stages)==6",
         "        assert report['orientation_count']==1 and len(stages)==3\n        report.update(five_stages=False, original_stage_indices=[3,4], stage_values_unchanged=True)"),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    scope=dict(b.worker.__globals__,FIELDS=FIELDS)
    exec(compile(source,'<unchanged-original-tail-stage-observer>','exec'),scope)
    return scope['worker']


def prepare():
    assert not b.OUT.exists() and not b.PLAN.exists()
    old=b.read(PREVIOUS/'review/closeout.json');assert old['frames']==59 and not old['quality_accepted']
    assert b.read(PREVIOUS/'review/region-gate-audit.json')['frames']==59
    assert b.read(PREVIOUS/'review/data-term-audit.json')['frames']==59
    args=b.arguments(True);_,rows,_,K,_,bound,provenance=b.sequence.load_inputs(args)
    receipt_path=PREVIOUS/'frames'/b.FRAME/'result.json';receipt=b.read(receipt_path)
    seed=Path(receipt['seed_path']);candidate=PREVIOUS/'frames'/b.FRAME/'candidate-00-temporal.npz'
    assert receipt['objective_selection']['selected_index']==0
    assert receipt['original_stage_indices']==[3,4]
    s=b.load_npz(seed);c=b.load_npz(candidate)
    assert np.array_equal(s['global_orient'],receipt['initial_state']['body_fit_initial_root'])
    assert np.array_equal(K[0],c['K_integer_centers']) and np.array_equal(s['betas'],c['betas'])
    files=[Path(__file__),Path(b.__file__),Path(b.probe.__file__),Path(b.common.__file__),
           Path(b.entry.__file__),Path(b.sequence.__file__),Path(b.probe.cache_static_topology.__code__.co_filename)]
    files += [Path(args.official_root)/p for p in b.entry.PINNED]
    inputs=[receipt_path,seed,candidate,PREVIOUS/'review/closeout.json',PREVIOUS/'review/criteria-audit.json',
            PREVIOUS/'review/region-gate-plan.json',PREVIOUS/'review/region-gate-audit.json',
            PREVIOUS/'review/data-term-plan.json',PREVIOUS/'review/data-term-audit.json',
            Path(args.shared_betas_npz),Path(args.assets_root)/'smplx/SMPLX_NEUTRAL.npz',
            Path(rows[0]['image']),Path(rows[0]['mask'])]
    bound.update({str(p):b.sha(p) for p in inputs})
    worker_function()
    plan=dict(updated=b.now(),state='prepared',frame=b.FRAME,seed_path=str(seed),candidate_path=str(candidate),
        selection_rule='Maximum existing registered feet6 median regression among all59 original winners, frame identity breaks a tie.',
        hypothesis='Determine whether the region regression occurs in original stage3 or4 before deciding on any parameter or active-variable change.',
        fitter_calls=1,orientation_count=1,body_stages=2,original_stage_indices=[3,4],maxiters=30,
        runtime_budget_seconds=300,random_seed=b.read(b.BODY/'warmtail-guarded-sequence-plan.json')['random_seed'],
        unchanged=['Exact historical seed','fullK/camera init/shared beta','native observations/confidence','original stage3/4 values','original LBFGS/stopping'],
        instrumentation='Existing monitor observer copies states only; no loss, optimizer, parameter, or gradient modification.',
        endpoint_parity=dict(vertices_camera=2e-6,joints_projected_native=.002,pose_embedding=2e-6,global_orient=2e-6,camera_translation=1e-7,betas=0),
        interpretation_gate='Require actual saved endpoint replay before interpreting the stage trajectory.',
        stop_conditions=['One deterministic failing case, one unchanged solve.','No parameter sweep or automatic all59 rerun.','Resume exact suspended original Gaussian in finally.'],
        source_hashes={str(p.resolve()):b.sha(p) for p in files},input_hashes=bound,input_check=provenance,native_wh=rows[0]['wh'],
        no_new_training_candidate=True,quality_accepted=False,goal_complete=False)
    b.save(b.PLAN,plan);print('PREPARED_WORST_REGION_ORIGINAL_TRACE',b.FRAME,flush=True)


def run(pid):
    scope=dict(b.run.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(b.run),'<existing-exclusive-stage-trace-lease>','exec'),scope)
    scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker_function()()
