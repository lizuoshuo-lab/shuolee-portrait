"""One body-observation weight control on the deterministically worst foot case.

Uses the original final two stages, loss, LBFGS and stopping. Changes only the
body/feet input joint-weight amplitude1->2; hand/face stage assignments stay
upstream. This is a local parameter intervention, not unchanged official values.
"""
import argparse
import copy
import inspect
from pathlib import Path
import trace_smplifyx_worst_region123 as trace

b=trace.b
REFERENCE=b.OUT
REFERENCE_PLAN=b.PLAN
b.OUT=b.BODY/'worst-region-body-weight-control'
b.PLAN=b.BODY/'worst-region-body-weight-control-plan.json'
b.HANDOFF=b.BODY/'worst-region-body-weight-control-gpu-handoff.json'


def patched_fit():
    source=inspect.getsource(b.entry.fit_one)
    old="jw = torch.ones((1,118), dtype=dtype, device=dev); jw[:,[1,9,12]] = 0"
    new="jw = torch.full((1,118), 2.0, dtype=dtype, device=dev); jw[:,[1,9,12]] = 0"
    assert source.count(old)==1
    source=source.replace(old,new)
    scope=dict(b.entry.fit_one.__globals__)
    exec(compile(source,'<body-observation-amplitude-only-control>','exec'),scope)
    return scope['fit_one']


def worker():
    # Keep the original trace worker, except endpoint equality is now a frozen
    # camera/shape contract; geometry is expected to change in this control.
    source=inspect.getsource(b.worker)
    replacements=[
        ("c['side_view_thsh']==25","c.get('side_view_thsh',25)==25"),
        ("            c.update(_seed_pose_embedding=seed['pose_embedding']",
         "            if c.get('data_weights') is None: c['data_weights']=[1]*5\n            for field in FIELDS:\n                assert len(c[field])==5\n                c[field]=c[field][3:]\n            c.update(_seed_pose_embedding=seed['pose_embedding']"),
        ("        assert report['orientation_count']==1 and len(stages)==6",
         "        assert report['orientation_count']==1 and len(stages)==3\n        report.update(five_stages=False, original_stage_indices=[3,4], body_observation_amplitude=2.0, original_body_observation_amplitude=1.0, body_numeric_weights_unchanged=False)"),
        ("        assert all(parity[k]<=limit for k,limit in p['endpoint_parity'].items()),parity",
         "        assert parity['camera_translation']==0 and parity['betas']==0,parity\n        assert np.array_equal(arrays['K_integer_centers'],expected['K_integer_centers'])\n        assert np.array_equal(arrays['faces'],expected['faces'])"),
        ("state='original_trajectory_complete_pending_visual_review'","state='body_weight_control_complete_pending_visual_review'"),
        ("endpoint_parity_passed=True","unchanged_camera_shape_topology=True, original_body_observation_amplitude=1.0, body_observation_amplitude=2.0"),
    ]
    for old,new in replacements:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    scope=dict(b.worker.__globals__,FIELDS=trace.FIELDS)
    exec(compile(source,'<existing-monitor-single-body-weight-control>','exec'),scope)
    original=b.entry.fit_one;b.entry.fit_one=patched_fit()
    try:scope['worker']()
    finally:b.entry.fit_one=original


def prepare():
    assert not b.OUT.exists() and not b.PLAN.exists()
    r=b.read(REFERENCE/'result.json');assert r['endpoint_parity_passed'] and r['frame']==trace.selected_frame()
    p=copy.deepcopy(b.read(REFERENCE_PLAN));p['source_hashes'][str(Path(__file__).resolve())]=b.sha(__file__)
    for path in [REFERENCE_PLAN,REFERENCE/'result.json',REFERENCE/'trajectory.npz',REFERENCE/'traced-winner.npz']:
        p['input_hashes'][str(path)]=b.sha(path)
    p.update(updated=b.now(),state='prepared',hypothesis='Original final hand/face emphasis trades off body/feet. Bring body/feet amplitude1 to the same value2 already used by original final hand/face; test one fixed candidate without a sweep.',
        single_change='Initial body/feet joint data-weight amplitude1->2, so its squared contribution is4x. Hand and face are overwritten by unchanged original stage schedules.',
        numeric_parameter_change=True,body_amplitude=2.,body_amplitude_before=1.,
        unchanged=['Original stage3/4 schedules and priors','exact seed and zero hand/face start','camera init, completeK, beta and topology','original solver/LBFGS/maxiters30/stopping','all original observations/confidence'],
        criteria=['All59 old criteria remain intact; this one case cannot admit the sequence.',
            'On the fixed diagnostic case body/feet median <= independent baseline+1px, face <= baseline+0.5px.',
            'Preserve exact camera translation, K, beta, topology; actual body/face/hands/feet native crops must show no new collapse.',
            'Report the same original unmodified-weight score separately; objectives with changed weights cannot be compared directly.'],
        risks=['Hands or face may regress when proximal body observations receive greater relative priority.','Shoe detections do not directly observe bare foot geometry.','Higher image weight is not a fix for wrong observations or depth ambiguity.'],
        stop_conditions=['One fixed diagnostic case and one amplitude; no threshold/weight sweep.','No automatic full59 fitting or model promotion.','Resume the exact prior Gaussian process in finally.'],
        no_new_training_candidate=False,diagnostic_candidate_only=True,quality_accepted=False,goal_complete=False)
    patched_fit();b.save(b.PLAN,p)
    print('PREPARED_SINGLE_BODY_WEIGHT_CONTROL',b.FRAME,flush=True)


def run(pid):
    scope=dict(b.run.__globals__,__file__=__file__)
    exec(compile(inspect.getsource(b.run),'<existing-exclusive-body-weight-lease>','exec'),scope);scope['run'](pid)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run','worker']);parser.add_argument('--pid',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='run':run(args.pid)
    else:worker()
